//! Verificador automático del port a Rust.
//!
//! Ejecuta una batería de comprobaciones sobre el entorno local y devuelve un
//! informe PASS/FAIL. Pensado para validar la instalación en otro equipo:
//!
//! ```text
//! reescaller-selftest.exe                 # usa la carpeta actual como proyecto
//! reescaller-selftest.exe --project .     # explícitamente
//! reescaller-selftest.exe --keep          # conserva los archivos generados
//! ```

use reescaller_core::config::{AppConfig, OutputFormat, ResizeMode, Weight};
use reescaller_core::{batch, diagnostics, process, upscale, video};
use std::path::{Path, PathBuf};
use std::process::ExitCode;

/// Resultado de una comprobación.
struct Resultado {
    nombre: String,
    ok: bool,
    detalle: String,
}

impl Resultado {
    fn nuevo(nombre: &str, ok: bool, detalle: impl Into<String>) -> Self {
        Self {
            nombre: nombre.to_string(),
            ok,
            detalle: detalle.into(),
        }
    }
}

fn main() -> ExitCode {
    let mut project = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    let mut keep = false;

    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--project" | "-p" => {
                i += 1;
                if let Some(v) = args.get(i) {
                    project = PathBuf::from(v);
                }
            }
            "--keep" => keep = true,
            "--help" | "-h" => {
                println!("reescaller-selftest [--project <dir>] [--keep]");
                return ExitCode::SUCCESS;
            }
            other => eprintln!("Argumento desconocido: {other}"),
        }
        i += 1;
    }

    println!("──────────────────────────────────────────────────────────────");
    println!(" Verificador automático · Reescaller (Rust)");
    println!("──────────────────────────────────────────────────────────────");
    println!(" Proyecto: {}", project.display());
    println!();

    let mut cfg = AppConfig::load();
    cfg.autodetectar(&project);

    let work = project.join("selftest_tmp");
    let _ = std::fs::create_dir_all(&work);

    let mut results: Vec<Resultado> = Vec::new();

    // ---------------------------------------------------------------- entorno
    let report = diagnostics::doctor(&cfg, &project);
    for check in &report.checks {
        // Los componentes opcionales (rostros con Python) no hacen fallar el verificador.
        results.push(Resultado::nuevo(
            &format!("entorno · {}", check.name),
            check.ok || !check.required,
            if check.ok || check.required {
                check.detail.clone()
            } else {
                format!("{} (opcional)", check.detail)
            },
        ));
    }

    let engine = cfg.realesrgan_exe.clone().filter(|p| p.is_file());
    let ffmpeg = cfg.ffmpeg_exe.clone().filter(|p| p.is_file());

    // --------------------------------------------------------- imagen de test
    let sample = work.join("muestra.png");
    let mut sample_dims = (0u32, 0u32);
    match crear_muestra(&sample) {
        Ok((w, h)) => {
            sample_dims = (w, h);
            results.push(Resultado::nuevo(
                "test · imagen de muestra generada",
                true,
                format!("{}x{}", w, h),
            ));
        }
        Err(e) => results.push(Resultado::nuevo(
            "test · imagen de muestra generada",
            false,
            e.to_string(),
        )),
    }

    let Some(engine_exe) = engine else {
        results.push(Resultado::nuevo(
            "motor · Real-ESRGAN disponible",
            false,
            "no se encontró realesrgan-ncnn-vulkan.exe",
        ));
        imprimir(&results);
        return ExitCode::FAILURE;
    };

    // ------------------------------------------------------- reescalado x1.5
    if sample_dims.0 > 0 {
        let plan = upscale::UpscalePlan {
            engine_scale: 4,
            model: "realesrgan-x4plus".into(),
            target_scale: 1.5,
            improve_percent: 100,
            weight: Weight::Quality { quality: 92 },
            output_format: OutputFormat::Png,
            resize_mode: ResizeMode::Always,
            use_engine: true,
            ..Default::default()
        };
        let salida = work.join("salida_x1.5.png");
        let log = |_: &str| {};
        match upscale::upscale_image(&plan, &engine_exe, &sample, &salida, &log) {
            Ok(r) => {
                let esperado = (
                    (sample_dims.0 as f32 * 1.5).round() as u32,
                    (sample_dims.1 as f32 * 1.5).round() as u32,
                );
                results.push(Resultado::nuevo(
                    "reescalado · escala elegida x1.5 (no la máxima del motor)",
                    r.final_size == esperado,
                    format!(
                        "{}x{} (esperado {}x{})",
                        r.final_size.0, r.final_size.1, esperado.0, esperado.1
                    ),
                ));
            }
            Err(e) => results.push(Resultado::nuevo(
                "reescalado · escala elegida x1.5",
                false,
                e.to_string(),
            )),
        }

        // --------------------------------------------- modo improve_only
        let plan = upscale::UpscalePlan {
            target_scale: 4.0,
            resize_mode: ResizeMode::ImproveOnly,
            output_format: OutputFormat::Png,
            ..plan.clone()
        };
        let plan_always = plan.clone();
        let salida = work.join("salida_mejora.png");
        match upscale::upscale_image(&plan, &engine_exe, &sample, &salida, &log) {
            Ok(r) => results.push(Resultado::nuevo(
                "reescalado · modo 'solo mejora' conserva el tamaño",
                r.final_size == sample_dims,
                format!("{}x{}", r.final_size.0, r.final_size.1),
            )),
            Err(e) => results.push(Resultado::nuevo(
                "reescalado · modo 'solo mejora'",
                false,
                e.to_string(),
            )),
        }

        // --------------------------------------------------- peso objetivo KB
        let plan = upscale::UpscalePlan {
            target_scale: 2.0,
            resize_mode: ResizeMode::Always,
            weight: Weight::TargetSize { target_kb: 30 },
            output_format: OutputFormat::Jpg,
            ..plan_always.clone()
        };
        let salida = work.join("salida_peso.jpg");
        match upscale::upscale_image(&plan, &engine_exe, &sample, &salida, &log) {
            Ok(r) => {
                let kb = r.bytes as f64 / 1024.0;
                let ok = (10.0..=90.0).contains(&kb);
                results.push(Resultado::nuevo(
                    "peso · objetivo 30 KB (búsqueda de calidad)",
                    ok,
                    format!("{kb:.1} KB en {}x{}", r.final_size.0, r.final_size.1),
                ));
            }
            Err(e) => results.push(Resultado::nuevo(
                "peso · objetivo 30 KB",
                false,
                e.to_string(),
            )),
        }

        // ------------------------------------------------------ sin motor IA
        let plan = upscale::UpscalePlan {
            target_scale: 2.0,
            resize_mode: ResizeMode::Always,
            use_engine: false,
            output_format: OutputFormat::Png,
            ..plan_always.clone()
        };
        let salida = work.join("salida_sin_motor.png");
        match upscale::upscale_image(&plan, &engine_exe, &sample, &salida, &log) {
            Ok(r) => results.push(Resultado::nuevo(
                "reescalado · modo rápido sin motor de IA",
                r.final_size == (sample_dims.0 * 2, sample_dims.1 * 2),
                format!("{}x{}", r.final_size.0, r.final_size.1),
            )),
            Err(e) => results.push(Resultado::nuevo(
                "reescalado · modo rápido sin motor",
                false,
                e.to_string(),
            )),
        }
    }

    // ------------------------------------------------------------------ lote
    if sample_dims.0 > 0 {
        let entrada = work.join("lote_entrada");
        let salida = work.join("lote_salida");
        let _ = std::fs::create_dir_all(&entrada);
        for nombre in ["a.png", "b.png"] {
            let _ = std::fs::copy(&sample, entrada.join(nombre));
        }
        let plan = upscale::UpscalePlan {
            target_scale: 1.5,
            output_format: OutputFormat::Png,
            use_engine: false,
            ..Default::default()
        };
        let log = |_: &str| {};
        match batch::process_folder(&plan, &engine_exe, &entrada, &salida, 2, &log) {
            Ok(r) => results.push(Resultado::nuevo(
                "lote · carpeta procesada",
                r.processed.len() == 2 && r.failed.is_empty(),
                format!(
                    "{} correctas, {} fallidas",
                    r.processed.len(),
                    r.failed.len()
                ),
            )),
            Err(e) => results.push(Resultado::nuevo(
                "lote · carpeta procesada",
                false,
                e.to_string(),
            )),
        }
    }

    // ----------------------------------------------------------------- video
    match ffmpeg {
        Some(ffmpeg_exe) => {
            let video_in = work.join("clip.mp4");
            let args = vec![
                "-y".to_string(),
                "-f".to_string(),
                "lavfi".to_string(),
                "-i".to_string(),
                "testsrc=size=320x240:rate=25:duration=2".to_string(),
                "-f".to_string(),
                "lavfi".to_string(),
                "-i".to_string(),
                "sine=frequency=440:duration=2".to_string(),
                "-c:v".to_string(),
                "libx264".to_string(),
                "-pix_fmt".to_string(),
                "yuv420p".to_string(),
                "-c:a".to_string(),
                "aac".to_string(),
                video_in.display().to_string(),
            ];
            if process::run(&ffmpeg_exe, &args).is_ok() {
                results.push(Resultado::nuevo(
                    "video · clip de prueba generado (FFmpeg)",
                    true,
                    "320x240 @25fps con audio",
                ));

                let opts = video::VideoOptions {
                    model: "realesr-animevideov3".into(),
                    engine_scale: 2,
                    target_scale: 1.5,
                    ..Default::default()
                };
                let salida = work.join("clip_mejorado.mp4");
                let log = |_: &str| {};
                match video::process_video(
                    &opts,
                    &ffmpeg_exe,
                    &engine_exe,
                    &video_in,
                    &salida,
                    &log,
                ) {
                    Ok(r) => {
                        let info = sondear_video(&ffmpeg_exe, &salida);
                        let dims_ok = info.as_ref().map(|i| i.dims == (480, 360)).unwrap_or(false);
                        let fps_ok = info
                            .as_ref()
                            .map(|i| i.fps.map(|f| (f - 25.0).abs() < 0.5).unwrap_or(false))
                            .unwrap_or(false);
                        let audio_ok = info.as_ref().map(|i| i.has_audio).unwrap_or(false);
                        results.push(Resultado::nuevo(
                            "video · escala 1.5 aplicada a los frames",
                            dims_ok,
                            match info.as_ref() {
                                Some(i) => format!("{}x{} (esperado 480x360)", i.dims.0, i.dims.1),
                                None => "no se pudo sondear".into(),
                            },
                        ));
                        results.push(Resultado::nuevo(
                            "video · FPS heredado del original (25)",
                            fps_ok,
                            match info.as_ref().and_then(|i| i.fps) {
                                Some(f) => format!("{f}"),
                                None => "desconocido".into(),
                            },
                        ));
                        results.push(Resultado::nuevo(
                            "video · audio original conservado",
                            audio_ok,
                            format!("{} frames", r.frames),
                        ));
                    }
                    Err(e) => results.push(Resultado::nuevo(
                        "video · pipeline completo",
                        false,
                        e.to_string(),
                    )),
                }
            } else {
                results.push(Resultado::nuevo(
                    "video · clip de prueba generado (FFmpeg)",
                    false,
                    "FFmpeg no pudo crear el clip",
                ));
            }
        }
        None => results.push(Resultado::nuevo(
            "video · FFmpeg disponible",
            false,
            "no se encontró ffmpeg.exe",
        )),
    }

    // ------------------------------------------- hardware y compatibilidad
    {
        use reescaller_core::hardware;
        if engine_exe.is_file() {
            hardware::set_engine_probe(engine_exe.clone());
        }
        let inf = hardware::detect_cached();
        let sonda_ok = !inf.vulkan_available || !inf.vulkan_devices.is_empty();
        results.push(Resultado::nuevo(
            "hardware · GPU y Vulkan detectados",
            sonda_ok,
            format!(
                "{} | motor IA: {}",
                inf.summary(),
                match inf.engine_support {
                    reescaller_core::EngineSupport::Full => "completo",
                    reescaller_core::EngineSupport::CpuOnly => "solo CPU",
                }
            ),
        ));
    }

    // -------------------------------------------------- modo fácil (remaster)
    if sample_dims.0 > 0 {
        use reescaller_core::remaster::{self, Preset, RemasterOptions};
        match remaster::analyze(&sample) {
            Ok(a) => {
                let opts = RemasterOptions {
                    preset: Preset::Auto,
                    ..Default::default()
                };
                let rp = remaster::build_plan(&opts, &a);
                let salida = work.join("remaster_auto.png");
                let log = |_: &str| {};
                match upscale::upscale_image(&rp.plan, &engine_exe, &sample, &salida, &log) {
                    Ok(o) => {
                        let esperado = rp.plan.target_dimensions(sample_dims);
                        results.push(Resultado::nuevo(
                            "remaster · modo fácil automático",
                            o.final_size == esperado,
                            format!(
                                "{}x{} -> {}x{} (IA: {})",
                                sample_dims.0,
                                sample_dims.1,
                                o.final_size.0,
                                o.final_size.1,
                                if o.engine_applied { "sí" } else { "no" }
                            ),
                        ));
                    }
                    Err(e) => results.push(Resultado::nuevo(
                        "remaster · modo fácil automático",
                        false,
                        e.to_string(),
                    )),
                }

                // Equipos sin GPU: forzar «solo CPU» debe seguir produciendo la imagen.
                let mut plan_cpu = rp.plan.clone();
                plan_cpu.device = reescaller_core::config::DevicePreference::Cpu;
                let salida_cpu = work.join("remaster_cpu.png");
                match upscale::upscale_image(&plan_cpu, &engine_exe, &sample, &salida_cpu, &log) {
                    Ok(o) => results.push(Resultado::nuevo(
                        "remaster · fallback sin GPU (solo CPU)",
                        !o.engine_applied && salida_cpu.is_file(),
                        format!("{}x{} sin motor de IA", o.final_size.0, o.final_size.1),
                    )),
                    Err(e) => results.push(Resultado::nuevo(
                        "remaster · fallback sin GPU (solo CPU)",
                        false,
                        e.to_string(),
                    )),
                }

                // Motor roto o sin VRAM: un ejecutable que siempre falla no debe tumbar el proceso.
                let falso = work.join("motor_falso.exe");
                let _ = std::fs::write(&falso, b"no soy un ejecutable");
                let salida_roto = work.join("remaster_motor_roto.png");
                match upscale::upscale_image(&rp.plan, &falso, &sample, &salida_roto, &log) {
                    Ok(o) => results.push(Resultado::nuevo(
                        "remaster · motor averiado -> sigue sin IA",
                        !o.engine_applied && salida_roto.is_file(),
                        "la imagen se genera igualmente",
                    )),
                    Err(e) => results.push(Resultado::nuevo(
                        "remaster · motor averiado -> sigue sin IA",
                        false,
                        e.to_string(),
                    )),
                }
            }
            Err(e) => results.push(Resultado::nuevo(
                "remaster · análisis de imagen",
                false,
                e.to_string(),
            )),
        }
    }

    // ------------------------------------------------- interfaz de depuración
    let puerto = std::env::var("REESCALLER_DEBUG_PORT")
        .ok()
        .and_then(|v| v.parse::<u16>().ok())
        .unwrap_or(7878);
    let servidor_activo = conexion_activa(("127.0.0.1", puerto));
    results.push(Resultado::nuevo(
        "interfaz · servidor de depuración",
        servidor_activo,
        if servidor_activo {
            format!("respondiendo en http://127.0.0.1:{puerto}")
        } else {
            "no está en marcha (arrastra `iniciar_debug.bat`)".to_string()
        },
    ));

    if !keep {
        let _ = std::fs::remove_dir_all(&work);
    }

    let ok = imprimir(&results);
    if ok {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    }
}

/// Crea una imagen de prueba con degradado y figuras.
fn crear_muestra(destino: &Path) -> anyhow::Result<(u32, u32)> {
    let (w, h) = (600u32, 400u32);
    let mut buffer = image::RgbImage::new(w, h);
    for (x, y, pixel) in buffer.enumerate_pixels_mut() {
        *pixel = image::Rgb([
            (x * 255 / w) as u8,
            (y * 255 / h) as u8,
            (((x + y) * 255 / (w + h)) as u8),
        ]);
    }
    // Figura de alto contraste para que el reescalado se note.
    for y in 120..280 {
        for x in 180..420 {
            let dx = x as f32 - 300.0;
            let dy = y as f32 - 200.0;
            let distancia = (dx * dx + dy * dy).sqrt() as u32;
            if distancia < 70 {
                buffer.put_pixel(x, y, image::Rgb([220, 40, 40]));
            }
        }
    }
    for i in 0..w {
        for y in 0..4 {
            buffer.put_pixel(i, y, image::Rgb([255, 255, 255]));
        }
    }
    buffer.save(destino)?;
    Ok((w, h))
}

/// Información del video extraída de la salida de FFmpeg.
struct InfoVideo {
    dims: (u32, u32),
    fps: Option<f64>,
    has_audio: bool,
}

fn sondear_video(ffmpeg: &Path, video: &Path) -> Option<InfoVideo> {
    let args = vec![
        "-hide_banner".to_string(),
        "-i".to_string(),
        video.display().to_string(),
    ];
    // `ffmpeg -i` sin archivo de salida termina con código 1 (es normal). Se lee la salida completa directamente:
    // `process::run` resume el stderr de los comandos fallidos y, con FFmpeg 8 (más metadatos), las líneas `Stream` quedaban fuera.
    let texto = match std::process::Command::new(ffmpeg)
        .args(&args)
        .stdin(std::process::Stdio::null())
        .output()
    {
        Ok(o) => format!(
            "{}\n{}",
            String::from_utf8_lossy(&o.stderr),
            String::from_utf8_lossy(&o.stdout)
        ),
        Err(_) => String::new(),
    };

    let mut dims = (0, 0);
    let mut fps = None;
    let mut has_audio = false;

    for linea in texto.lines() {
        if linea.contains("Video:") {
            if let Some((w, h)) = extraer_dims(linea) {
                dims = (w, h);
            }
            if let Some(f) = extraer_fps(linea) {
                fps = Some(f);
            }
        }
        if linea.contains("Audio:") {
            has_audio = true;
        }
    }

    if dims.0 == 0 {
        None
    } else {
        Some(InfoVideo {
            dims,
            fps,
            has_audio,
        })
    }
}

fn extraer_dims(linea: &str) -> Option<(u32, u32)> {
    // Solo se busca después de "Video:" (antes aparecen codigos como "0x1").
    let ventana = match linea.find("Video:") {
        Some(pos) => &linea[pos..],
        None => linea,
    };
    // Busca un token tipo "1280x720".
    let mut inicio = 0;
    while let Some(pos) = ventana.get(inicio..).and_then(|rest| rest.find('x')) {
        let pos = inicio + pos;
        let izquierda = &ventana[..pos];
        let derecha = &ventana[pos + 1..];
        let w: String = izquierda
            .chars()
            .rev()
            .take_while(|c| c.is_ascii_digit())
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect();
        let h: String = derecha.chars().take_while(|c| c.is_ascii_digit()).collect();
        if !w.is_empty() && !h.is_empty() {
            if let (Ok(w), Ok(h)) = (w.parse::<u32>(), h.parse::<u32>()) {
                // Se descartan tokens tipo "0x31637661" (códigos de códec).
                if w > 0 && h > 0 {
                    return Some((w, h));
                }
            }
        }
        inicio = pos + 1;
    }
    None
}

fn extraer_fps(linea: &str) -> Option<f64> {
    let pos = linea.find(" fps")?;
    let antes = &linea[..pos];
    let numero: String = antes
        .chars()
        .rev()
        .take_while(|c| c.is_ascii_digit() || *c == '.')
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect();
    numero.parse::<f64>().ok()
}

fn conexion_activa(addr: (&str, u16)) -> bool {
    let destino = format!("{}:{}", addr.0, addr.1);
    match std::net::ToSocketAddrs::to_socket_addrs(&destino) {
        Ok(mut addrs) => {
            let socket = std::net::TcpStream::connect_timeout(
                &addrs.next().expect("dirección válida"),
                std::time::Duration::from_millis(800),
            );
            socket.is_ok()
        }
        Err(_) => false,
    }
}

fn imprimir(results: &[Resultado]) -> bool {
    let ok = results.iter().filter(|r| r.ok).count();
    let total = results.len();

    println!("──────────────────────────────────────────────────────────────");
    for r in results {
        println!(
            " {} {:<52} {}",
            if r.ok { "PASA" } else { "FALLA" },
            r.nombre,
            r.detalle
        );
    }
    println!("──────────────────────────────────────────────────────────────");
    println!(" Resultado: {ok}/{total} comprobaciones superadas");
    if ok == total {
        println!(" ✔ Todo correcto. El port a Rust funciona en este equipo.");
    } else {
        println!(" ✘ Hay comprobaciones fallidas (revisa el detalle anterior).");
    }
    ok == total
}
