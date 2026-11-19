//! Pipeline de video: extracción de frames → mejora → fusión con audio.
//!
//! Reemplaza a `ejecutar_proceso_video()` del proyecto Python, heredando el FPS
//! real del origen y permitiendo fijar la escala final de los frames.

use crate::config::{OutputFormat, ResizeFilter};
use crate::error::{Error, Result};
use crate::image_ops;
use crate::process;
use image::GenericImageView;
use std::path::{Path, PathBuf};

/// Opciones del proceso de video.
#[derive(Debug, Clone)]
pub struct VideoOptions {
    /// Modelo de Real-ESRGAN.
    pub model: String,
    /// Escala del motor (2 o 4).
    pub engine_scale: u8,
    /// Escala final de los frames respecto al original (1.0 = mismo tamaño).
    pub target_scale: f32,
    /// Formato de los frames intermedios.
    pub frame_format: OutputFormat,
    /// Eliminar temporales al terminar.
    pub delete_temp: bool,
    /// Filtro de reescalado.
    pub resize_filter: ResizeFilter,
    /// Dispositivo preferido para el motor (auto / nvidia / amd / intel / cpu).
    pub device: crate::config::DevicePreference,
    /// Índice de GPU forzado (`-g`).
    pub gpu_index: Option<usize>,
}

impl Default for VideoOptions {
    fn default() -> Self {
        Self {
            model: "realesrgan-x4plus".to_string(),
            engine_scale: 4,
            target_scale: 1.0,
            frame_format: OutputFormat::Jpg,
            delete_temp: true,
            resize_filter: ResizeFilter::Lanczos3,
            device: crate::config::DevicePreference::Auto,
            gpu_index: None,
        }
    }
}

/// Resultado del proceso de video.
#[derive(Debug, Clone)]
pub struct VideoOutcome {
    pub output: PathBuf,
    pub frames: usize,
    pub fps: Option<f64>,
    pub bytes: u64,
}

/// Ejecuta el pipeline completo de video.
pub fn process_video(
    opts: &VideoOptions,
    ffmpeg: &Path,
    engine_exe: &Path,
    input: &Path,
    output: &Path,
    log: &(dyn Fn(&str) + Send + Sync),
) -> Result<VideoOutcome> {
    if !input.is_file() {
        return Err(Error::FileNotFound(input.to_path_buf()));
    }

    let base_dir = output
        .parent()
        .map(|p| p.to_path_buf())
        .unwrap_or_else(|| PathBuf::from("."));
    let stem = input
        .file_stem()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| "video".into());

    // Carpeta de trabajo única (sin el nombre del video: un «%» o un nombre repetido romperían ffmpeg o borrarían carpetas
    // del usuario) que se elimina siempre al terminar, incluso si hay un error a mitad.
    struct Temporal(PathBuf);
    impl Drop for Temporal {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }
    let _ = &stem;
    let marca = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0);
    let raiz = base_dir.join(format!("reescaller_tmp_{}_{marca}", std::process::id()));
    let _guardia = Temporal(raiz.clone());
    let tmp_frames = raiz.join("frames");
    let out_frames = raiz.join("mejorados");
    let numbered = raiz.join("numerados");

    std::fs::create_dir_all(&tmp_frames)?;
    std::fs::create_dir_all(&out_frames)?;

    let cleanup = |paths: &[&Path]| {
        for p in paths {
            let _ = std::fs::remove_dir_all(p);
        }
    };

    // --- Paso 1: extraer frames ---------------------------------------------
    let args = vec![
        "-y".to_string(),
        "-i".to_string(),
        input.display().to_string(),
        "-qscale:v".to_string(),
        "1".to_string(),
        "-qmin".to_string(),
        "1".to_string(),
        "-qmax".to_string(),
        "1".to_string(),
        "-vsync".to_string(),
        "0".to_string(),
        tmp_frames.join("frame%08d.jpg").display().to_string(),
    ];
    process::run_with_log(ffmpeg, &args, log)?;

    // --- Paso 2: mejorar frames ---------------------------------------------
    let args = vec![
        "-i".to_string(),
        tmp_frames.display().to_string(),
        "-o".to_string(),
        out_frames.display().to_string(),
        "-n".to_string(),
        opts.model.clone(),
        "-s".to_string(),
        opts.engine_scale.clamp(2, 4).to_string(),
        "-f".to_string(),
        "jpg".to_string(),
    ];

    // Selección de GPU (igual que en imágenes): sin Vulkan no hay motor y no
    // se puede mejorar el video; se avisa y se continúa con los frames originales.
    let informe = crate::hardware::detect_cached();
    let (indice_gpu, avisos) =
        crate::hardware::resolve_preference(opts.device, informe, opts.gpu_index);
    for aviso in avisos {
        log(&format!("⚠ {aviso}"));
    }
    if !informe.vulkan_available {
        return Err(crate::error::Error::Image(
            "No se detectó Vulkan en este equipo: el motor de mejora de video necesita una GPU \
             con Vulkan (NVIDIA, AMD o Intel Arc/Arcade con driver actualizado)."
                .into(),
        ));
    }
    let gpu_args = crate::hardware::engine_gpu_args(informe, indice_gpu);
    let intentos: Vec<(Vec<String>, &str)> = vec![
        (gpu_args.clone(), ""),
        (
            [gpu_args.clone(), vec!["-t".into(), "100".into()]].concat(),
            "teselas de 100 px",
        ),
        (
            vec!["-t".into(), "32".into()],
            "GPU automática y teselas de 32 px",
        ),
    ];
    let mut resultado: Result<process::CommandOutput> = Err(Error::Image("sin intentos".into()));
    for (extra, motivo) in intentos {
        if !motivo.is_empty() {
            log(&format!("Reintentando con {motivo}..."));
        }
        let mut a = args.clone();
        a.extend(extra);
        resultado = process::run_with_log(engine_exe, &a, log);
        if resultado.is_ok() {
            break;
        }
    }
    if let Err(e) = resultado {
        cleanup(&[&tmp_frames, &out_frames]);
        return Err(e);
    }

    // --- Paso 3: reescalar frames al tamaño final --------------------------
    let mut produced: Vec<PathBuf> = std::fs::read_dir(&out_frames)
        .map(|entries| {
            entries
                .flatten()
                .map(|e| e.path())
                .filter(|p| p.is_file())
                .collect()
        })
        .unwrap_or_default();
    produced.sort();

    if produced.is_empty() {
        cleanup(&[&tmp_frames, &out_frames]);
        return Err(Error::Image("El motor no generó frames de salida.".into()));
    }

    std::fs::create_dir_all(&numbered)?;
    let effective = effective_scale(opts);
    let target_ratio = opts.target_scale / effective;

    for (idx, frame) in produced.iter().enumerate() {
        let target_path = numbered.join(format!("frame{:08}.jpg", idx + 1));
        let needs_resize = (target_ratio - 1.0).abs() > 0.001;
        if needs_resize {
            let img = image_ops::load(frame)?;
            let (w, h) = img.dimensions();
            let nw = ((w as f32 * target_ratio).round() as u32).max(1);
            let nh = ((h as f32 * target_ratio).round() as u32).max(1);
            let resized = image_ops::resize(&img, nw, nh, opts.resize_filter);
            let data = image_ops::encode(
                &resized,
                OutputFormat::Jpg,
                crate::config::Weight::Quality { quality: 95 },
            )?;
            image_ops::save_bytes(&data, &target_path)?;
        } else {
            std::fs::copy(frame, &target_path)?;
        }
    }

    let frames = produced.len();

    // --- Paso 4: fusionar con el audio original ----------------------------
    let fps = process::detect_fps(ffmpeg, input);
    let mut args = vec![
        "-y".to_string(),
        "-i".to_string(),
        numbered.join("frame%08d.jpg").display().to_string(),
        "-i".to_string(),
        input.display().to_string(),
        "-map".to_string(),
        "0:v:0".to_string(),
        "-map".to_string(),
        "1:a:0?".to_string(),
        "-c:a".to_string(),
        "copy".to_string(),
        "-c:v".to_string(),
        "libx264".to_string(),
        "-pix_fmt".to_string(),
        "yuv420p".to_string(),
    ];
    if let Some(fps) = fps {
        args.push("-r".to_string());
        args.push(format_fps(fps));
        log(&format!(
            "FPS del video original detectado: {}",
            format_fps(fps)
        ));
    }
    args.push(output.display().to_string());
    process::run_with_log(ffmpeg, &args, log)?;

    if opts.delete_temp {
        cleanup(&[&tmp_frames, &out_frames, &numbered]);
    }

    let bytes = std::fs::metadata(output).map(|m| m.len()).unwrap_or(0);
    log(&format!(
        "Video guardado: {} ({frames} frames, {:.1} MB)",
        output.display(),
        bytes as f64 / 1_048_576.0
    ));

    Ok(VideoOutcome {
        output: output.to_path_buf(),
        frames,
        fps,
        bytes,
    })
}

/// Formatea un valor de FPS sin ceros sobrantes (23.976, 25, 29.97...).
pub fn format_fps(fps: f64) -> String {
    let mut text = format!("{fps:.6}");
    if text.contains('.') {
        while text.ends_with('0') {
            text.pop();
        }
        if text.ends_with('.') {
            text.pop();
        }
    }
    text
}

fn effective_scale(opts: &VideoOptions) -> f32 {
    if opts.model.starts_with("realesrgan-x4plus") {
        4.0
    } else {
        opts.engine_scale.clamp(2, 4) as f32
    }
}

/// Convierte un video a otro formato (remux sin recodificar).
pub fn convert_video(
    ffmpeg: &Path,
    input: &Path,
    output: &Path,
    log: &(dyn Fn(&str) + Send + Sync),
) -> Result<()> {
    let ext = output
        .extension()
        .map(|e| e.to_string_lossy().to_lowercase())
        .unwrap_or_default();
    let base = |extra: &[&str]| -> Vec<String> {
        let mut a = vec![
            "-y".to_string(),
            "-i".to_string(),
            input.display().to_string(),
            "-map".to_string(),
            "0:v:0".to_string(),
            "-map".to_string(),
            "0:a:0?".to_string(),
        ];
        a.extend(extra.iter().map(|s| s.to_string()));
        a.push(output.display().to_string());
        a
    };

    // 0) formatos que no admiten copia de streams: GIF animado (primeros 20 s) y solo audio MP3.
    if ext == "gif" {
        log("GIF: se usan los primeros 20 s, a 480 px de ancho y 12 fps.");
        let args = vec![
            "-y".to_string(),
            "-t".to_string(),
            "20".to_string(),
            "-i".to_string(),
            input.display().to_string(),
            "-an".to_string(),
            "-vf".to_string(),
            "fps=12,scale=480:-2:flags=lanczos,split[s0][s1];[s0]palettegen=max_colors=128[p];[s1][p]paletteuse=dither=bayer".to_string(),
            "-loop".to_string(),
            "0".to_string(),
            output.display().to_string(),
        ];
        process::run_with_log(ffmpeg, &args, log)?;
        return Ok(());
    }
    if ext == "mp3" {
        log("MP3: se extrae solo el audio.");
        let args = vec![
            "-y".to_string(),
            "-i".to_string(),
            input.display().to_string(),
            "-vn".to_string(),
            "-c:a".to_string(),
            "libmp3lame".to_string(),
            "-q:a".to_string(),
            "2".to_string(),
            output.display().to_string(),
        ];
        process::run_with_log(ffmpeg, &args, log)?;
        return Ok(());
    }

    // 1) copia de streams (instantánea y sin pérdida) salvo en WebM, que solo admite VP8/VP9/AV1 + Vorbis/Opus.
    if ext != "webm" {
        match process::run_with_log(ffmpeg, &base(&["-c", "copy"]), log) {
            Ok(_) => return Ok(()),
            Err(e) => log(&format!(
                "La copia directa no es posible para este formato ({e}); se recodifica."
            )),
        }
    } else {
        log("WebM requiere VP9 + Opus: se recodifica el video.");
    }

    // 2) recodificación con los códecs adecuados al contenedor
    let extra: Vec<&str> = match ext.as_str() {
        "webm" => vec![
            "-c:v",
            "libvpx-vp9",
            "-crf",
            "32",
            "-b:v",
            "0",
            "-deadline",
            "good",
            "-cpu-used",
            "4",
            "-row-mt",
            "1",
            "-c:a",
            "libopus",
            "-b:a",
            "128k",
        ],
        "avi" => vec![
            "-c:v",
            "mpeg4",
            "-q:v",
            "3",
            "-c:a",
            "libmp3lame",
            "-q:a",
            "3",
        ],
        "mp4" | "mov" | "m4v" => vec![
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "20",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-movflags",
            "+faststart",
        ],
        _ => vec![
            "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p", "-c:a",
            "aac", "-b:a", "192k",
        ],
    };
    process::run_with_log(ffmpeg, &base(&extra), log)?;
    Ok(())
}

/// Recorta un video (copia de streams) entre un instante y una duración.
pub fn trim_video(
    ffmpeg: &Path,
    input: &Path,
    output: &Path,
    start: &str,
    duration: &str,
    log: &(dyn Fn(&str) + Send + Sync),
) -> Result<()> {
    let args = vec![
        "-y".to_string(),
        "-ss".to_string(),
        start.to_string(),
        "-i".to_string(),
        input.display().to_string(),
        "-t".to_string(),
        duration.to_string(),
        "-c".to_string(),
        "copy".to_string(),
        output.display().to_string(),
    ];
    process::run_with_log(ffmpeg, &args, log)?;
    Ok(())
}
