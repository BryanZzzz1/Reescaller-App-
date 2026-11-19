//! Manejadores HTTP de la interfaz de depuración.

use crate::state::State;
use base64::Engine;
use reescaller_core::{
    batch,
    config::{DevicePreference, OutputFormat, ResizeFilter, ResizeMode, Weight},
    diagnostics, face, hardware, image_ops, remaster, upscale, video, AppConfig,
};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

/// Plan de reescalado recibido desde la UI.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct PlanDto {
    pub engine_scale: u8,
    pub model: String,
    pub target_scale: f32,
    pub improve_percent: u8,
    pub weight: Weight,
    pub output_format: OutputFormat,
    pub resize_mode: ResizeMode,
    pub resize_filter: ResizeFilter,
    pub sharpen: u8,
    pub use_engine: bool,
    /// Dispositivo preferido: auto / nvidia / amd / intel / cpu.
    pub device: DevicePreference,
    /// Índice de GPU forzado (`-g`).
    pub gpu_index: Option<usize>,
}

impl Default for PlanDto {
    fn default() -> Self {
        PlanDto {
            engine_scale: 4,
            model: "realesrgan-x4plus".to_string(),
            target_scale: 2.0,
            improve_percent: 100,
            weight: Weight::Quality { quality: 92 },
            output_format: OutputFormat::Png,
            resize_mode: ResizeMode::Always,
            resize_filter: ResizeFilter::Lanczos3,
            sharpen: 0,
            use_engine: true,
            device: DevicePreference::Auto,
            gpu_index: None,
        }
    }
}

impl From<&PlanDto> for upscale::UpscalePlan {
    fn from(dto: &PlanDto) -> Self {
        upscale::UpscalePlan {
            engine_scale: dto.engine_scale,
            model: dto.model.clone(),
            target_scale: dto.target_scale,
            improve_percent: dto.improve_percent,
            weight: dto.weight,
            output_format: dto.output_format,
            resize_mode: dto.resize_mode,
            resize_filter: dto.resize_filter,
            sharpen: dto.sharpen,
            use_engine: dto.use_engine,
            device: dto.device,
            gpu_index: dto.gpu_index,
        }
    }
}

/// Petición de reescalado.
#[derive(Deserialize)]
pub struct UpscaleRequest {
    pub input: String,
    #[serde(default)]
    pub output: Option<String>,
    #[serde(default)]
    pub plan: PlanDto,
}

/// Petición de lote.
#[derive(Deserialize)]
pub struct BatchRequest {
    pub input: String,
    #[serde(default)]
    pub output: String,
    #[serde(default)]
    pub plan: PlanDto,
    #[serde(default = "default_workers")]
    pub workers: usize,
}

fn default_workers() -> usize {
    2
}

/// Petición de video.
#[derive(Deserialize)]
pub struct VideoRequest {
    pub input: String,
    pub output: String,
    #[serde(default)]
    pub model: String,
    #[serde(default = "default_engine")]
    pub engine_scale: u8,
    #[serde(default = "default_target")]
    pub target_scale: f32,
    #[serde(default)]
    pub device: DevicePreference,
}

fn default_engine() -> u8 {
    4
}
fn default_target() -> f32 {
    1.0
}

/// Petición de rostros.
#[derive(Deserialize)]
pub struct FaceRequest {
    pub input: String,
    #[serde(default)]
    pub output: Option<String>,
    #[serde(default = "default_face_model")]
    pub model: String,
    /// `portrait` (retrato completo: pelo, ropa y fondo con el motor general + rostros) o `face` (solo rostros).
    #[serde(default = "default_face_scope")]
    pub scope: String,
    /// Fotos oscuras: `auto` (aclara solo si hace falta), `on` (siempre) u `off`.
    #[serde(default = "default_face_light")]
    pub light: String,
    /// Fuerza de la restauración de rostros, 0..=100.
    #[serde(default = "default_face_strength")]
    pub strength: u32,
}

/// Petición de conversión de video.
#[derive(Deserialize)]
pub struct ConvertRequest {
    pub input: String,
    #[serde(default)]
    pub output: Option<String>,
    /// Formato de salida: mp4, avi, mov, mkv, webm...
    #[serde(default = "default_format")]
    pub format: String,
}

fn default_format() -> String {
    "mp4".to_string()
}

/// Petición de recorte de video.
#[derive(Deserialize)]
pub struct TrimRequest {
    pub input: String,
    #[serde(default)]
    pub output: Option<String>,
    /// Instante de inicio (hh:mm:ss).
    #[serde(default)]
    pub start: String,
    /// Duración en segundos.
    #[serde(default)]
    pub duration: String,
}

fn default_face_light() -> String {
    "auto".to_string()
}

fn default_face_strength() -> u32 {
    100
}

fn default_face_scope() -> String {
    "portrait".to_string()
}

fn default_face_model() -> String {
    "gfpgan".to_string()
}

/// Petición de subida de archivo.
#[derive(Deserialize)]
pub struct UploadRequest {
    /// Nombre del archivo.
    pub name: String,
    /// Contenido en base64 (sin cabecera data:).
    pub data: String,
}

/// Respuesta estándar.
/// Copia las líneas registradas durante la operación.
fn collect_logs(logs: &Mutex<Vec<String>>) -> Vec<String> {
    logs.lock().map(|buf| buf.clone()).unwrap_or_default()
}

fn ok_response(payload: serde_json::Value, logs: Vec<String>) -> serde_json::Value {
    json!({ "ok": true, "result": payload, "logs": logs })
}

fn err_response(message: &str, logs: Vec<String>) -> serde_json::Value {
    json!({ "ok": false, "error": message, "logs": logs })
}

/// Ejecuta el pipeline de reescalado.
pub fn handle_upscale(state: &State, req: UpscaleRequest) -> serde_json::Value {
    let logs: Mutex<Vec<String>> = Mutex::new(Vec::new());
    let log = |msg: &str| {
        state.push_log(msg);
        if let Ok(mut buf) = logs.lock() {
            buf.push(msg.to_string());
        }
    };

    let input = PathBuf::from(&req.input);
    if !input.is_file() {
        return err_response(
            &format!("No existe el archivo: {}", input.display()),
            collect_logs(&logs),
        );
    }

    let cfg = state.config.lock().unwrap().clone();
    let engine = match cfg.realesrgan_exe.clone() {
        Some(p) if p.is_file() => p,
        _ => {
            return err_response(
                "Real-ESRGAN no está configurado (se autodetecta al arrancar).",
                collect_logs(&logs),
            )
        }
    };

    let plan: upscale::UpscalePlan = (&req.plan).into();
    let output = match req.output.filter(|s| !s.trim().is_empty()) {
        Some(o) => PathBuf::from(o),
        // Igual que el modo fácil: junto al original o, si viene de una subida/pegado, en Imágenes\Reescaller.
        None => remaster_output(state, &input, None, plan.output_format, "_mejorado"),
    };

    match upscale::upscale_image(&plan, &engine, &input, &output, &log) {
        Ok(outcome) => ok_response(
            json!({
                "output": outcome.output.display().to_string(),
                "originalWidth": outcome.original_size.0,
                "originalHeight": outcome.original_size.1,
                "finalWidth": outcome.final_size.0,
                "finalHeight": outcome.final_size.1,
                "bytes": outcome.bytes,
                "engineApplied": outcome.engine_applied,
                "engineScale": outcome.engine_scale,
                "weight": format!("{:?}", outcome.weight),
            }),
            collect_logs(&logs),
        ),
        Err(e) => err_response(&e.to_string(), collect_logs(&logs)),
    }
}

/// Ejecuta el lote.
pub fn handle_batch(state: &State, req: BatchRequest) -> serde_json::Value {
    let logs: Mutex<Vec<String>> = Mutex::new(Vec::new());
    let log = |msg: &str| {
        state.push_log(msg);
        if let Ok(mut buf) = logs.lock() {
            buf.push(msg.to_string());
        }
    };

    let input = PathBuf::from(&req.input);
    // Sin carpeta de salida: las imágenes mejoradas se guardan dentro de la propia carpeta, en «Mejoradas».
    let output = if req.output.trim().is_empty() {
        input.join("Mejoradas")
    } else {
        PathBuf::from(&req.output)
    };
    if !input.is_dir() {
        return err_response(
            &format!("No existe la carpeta: {}", input.display()),
            collect_logs(&logs),
        );
    }

    let cfg = state.config.lock().unwrap().clone();
    let engine = match cfg.realesrgan_exe.clone() {
        Some(p) if p.is_file() => p,
        _ => return err_response("Real-ESRGAN no está configurado.", collect_logs(&logs)),
    };

    let plan: upscale::UpscalePlan = (&req.plan).into();
    match batch::process_folder(
        &plan,
        &engine,
        &input,
        &output,
        req.workers.clamp(1, 16),
        &log,
    ) {
        Ok(outcome) => ok_response(
            json!({
                "processed": outcome.processed.iter().map(|o| o.output.display().to_string()).collect::<Vec<_>>(),
                "failed": outcome.failed.iter().map(|(p, e)| json!({"file": p.display().to_string(), "error": e})).collect::<Vec<_>>(),
            }),
            collect_logs(&logs),
        ),
        Err(e) => err_response(&e.to_string(), collect_logs(&logs)),
    }
}

/// Ejecuta el pipeline de video.
pub fn handle_video(state: &State, req: VideoRequest) -> serde_json::Value {
    let logs: Mutex<Vec<String>> = Mutex::new(Vec::new());
    let log = |msg: &str| {
        state.push_log(msg);
        if let Ok(mut buf) = logs.lock() {
            buf.push(msg.to_string());
        }
    };

    let input = PathBuf::from(&req.input);
    let output = PathBuf::from(&req.output);
    if !input.is_file() {
        return err_response(
            &format!("No existe el video: {}", input.display()),
            collect_logs(&logs),
        );
    }

    let cfg = state.config.lock().unwrap().clone();
    let (ffmpeg, engine) = match (cfg.ffmpeg_exe.clone(), cfg.realesrgan_exe.clone()) {
        (Some(f), Some(e)) if f.is_file() && e.is_file() => (f, e),
        _ => {
            return err_response(
                "FFmpeg o Real-ESRGAN no están configurados.",
                collect_logs(&logs),
            )
        }
    };

    let opts = video::VideoOptions {
        model: req.model.clone(),
        engine_scale: req.engine_scale,
        target_scale: req.target_scale,
        device: req.device,
        gpu_index: cfg.gpu_index,
        ..Default::default()
    };

    match video::process_video(&opts, &ffmpeg, &engine, &input, &output, &log) {
        Ok(outcome) => ok_response(
            json!({
                "output": outcome.output.display().to_string(),
                "frames": outcome.frames,
                "fps": outcome.fps,
                "bytes": outcome.bytes,
            }),
            collect_logs(&logs),
        ),
        Err(e) => err_response(&e.to_string(), collect_logs(&logs)),
    }
}

/// Restaura rostros a través del puente de Python.
pub fn handle_face(state: &State, req: FaceRequest) -> serde_json::Value {
    let logs: Mutex<Vec<String>> = Mutex::new(Vec::new());
    let log = |msg: &str| {
        state.push_log(msg);
        if let Ok(mut buf) = logs.lock() {
            buf.push(msg.to_string());
        }
    };

    let input = PathBuf::from(&req.input);
    if !input.is_file() {
        return err_response(
            &format!("No existe la imagen: {}", input.display()),
            collect_logs(&logs),
        );
    }

    let model = match face::FaceModel::parse(&req.model) {
        Some(m) => m,
        None => {
            return err_response(
                &format!("Modelo desconocido: {}", req.model),
                collect_logs(&logs),
            )
        }
    };

    let cfg = state.config.lock().unwrap().clone();
    let output = match req.output.filter(|s| !s.trim().is_empty()) {
        Some(o) => PathBuf::from(o),
        None => upscale::default_output_path(&input, OutputFormat::Png, "_rostros"),
    };

    // El puente de Python solo se prepara si hace falta (no hay motor nativo).
    let script_abs = ensure_bridge_script().unwrap_or_default();

    // Retrato completo: primero el motor general amplía TODA la imagen (pelo, ropa, fondo) y luego se restauran los
    // rostros encima. Con imágenes grandes o sin motor/Vulkan se hace solo la parte de rostros.
    let mut fuente = input.clone();
    let mut temporal: Option<PathBuf> = None;
    let mut temporal_luz: Option<PathBuf> = None;
    reescaller_core::face_onnx::set_strength(req.strength.min(100) as f32 / 100.0);

    // Fotos oscuras: se aclaran (y se limpia un poco el ruido) antes de restaurar.
    if req.light != "off" {
        if let Ok(img) = image_ops::load(&input) {
            if let Some(clara) = image_ops::lift_dark(&img, req.light != "on") {
                let tmp = std::env::temp_dir().join(format!(
                    "reescaller_luz_{}_{}.png",
                    std::process::id(),
                    crate::state::now_ms()
                ));
                if clara.save(&tmp).is_ok() {
                    log("Foto oscura: se aclara y se reduce el ruido antes de restaurar.");
                    fuente = tmp.clone();
                    temporal_luz = Some(tmp);
                }
            }
        }
    }
    let entrada_retrato = fuente.clone();
    let mut escala_rostros = 2u32;
    let mut retrato = false;
    if req.scope != "face" {
        let (w, h) = image::image_dimensions(&input).unwrap_or((0, 0));
        let informe = reescaller_core::hardware::detect_cached();
        match cfg.realesrgan_exe.clone().filter(|p| p.is_file()) {
            Some(motor) if informe.vulkan_available && cfg.device != reescaller_core::config::DevicePreference::Cpu && (w as u64 * h as u64) <= 6_000_000 => {
                let tmp = std::env::temp_dir().join(format!("reescaller_retrato_{}_{}.png", std::process::id(), crate::state::now_ms()));
                // Imágenes pequeñas (lado largo < 640 px) se amplían ×4; el resto ×2.
                let escala = if w.max(h) < 640 && (w as u64 * h as u64 * 16) <= 48_000_000 { 4.0 } else { 2.0 };
                log(&format!("Retrato completo: ampliación ×{escala}."));
                let plan = upscale::UpscalePlan {
                    target_scale: escala,
                    device: cfg.device,
                    gpu_index: cfg.gpu_index,
                    ..Default::default()
                };
                log("Retrato completo: mejorando toda la imagen (pelo, ropa y fondo)…");
                match upscale::upscale_image(&plan, &motor, &entrada_retrato, &tmp, &log) {
                    Ok(o) if o.engine_applied => {
                        fuente = tmp.clone();
                        temporal = Some(tmp);
                        escala_rostros = 1;
                        retrato = true;
                    }
                    Ok(_) => log("⚠ El motor general no se aplicó: solo se restauran los rostros."),
                    Err(e) => log(&format!("⚠ No se pudo mejorar la imagen completa ({e}): solo se restauran los rostros.")),
                }
            }
            _ => log("Retrato completo no disponible (imagen muy grande, sin motor o sin Vulkan): solo se restauran los rostros."),
        }
    }

    let resultado = face::restore_scaled(
        &cfg,
        &state.project_dir,
        &script_abs,
        model,
        &fuente,
        &output,
        escala_rostros,
        &log,
    );
    if let Some(t) = temporal {
        let _ = std::fs::remove_file(t);
    }
    if let Some(t) = temporal_luz {
        let _ = std::fs::remove_file(t);
    }
    match resultado {
        Ok(outcome) => {
            log(&format!("Motor: {} · {}", outcome.backend, outcome.detail));
            let detalle = if retrato {
                format!("Retrato completo · {}", outcome.detail)
            } else {
                outcome.detail.clone()
            };
            ok_response(
                json!({
                    "output": output.display().to_string(),
                    "backend": outcome.backend,
                    "detail": detalle,
                }),
                collect_logs(&logs),
            )
        }
        Err(e) => err_response(&e.to_string(), collect_logs(&logs)),
    }
}

/// Convierte un video a otro formato (remux, sin recodificar).
pub fn handle_convert(state: &State, req: ConvertRequest) -> serde_json::Value {
    let logs: Mutex<Vec<String>> = Mutex::new(Vec::new());
    let log = |msg: &str| {
        state.push_log(msg);
        if let Ok(mut buf) = logs.lock() {
            buf.push(msg.to_string());
        }
    };

    let input = PathBuf::from(&req.input);
    if !input.is_file() {
        return err_response(
            &format!("No existe el video: {}", input.display()),
            collect_logs(&logs),
        );
    }

    let cfg = state.config.lock().unwrap().clone();
    let ffmpeg = match cfg.ffmpeg_exe.clone().filter(|p| p.is_file()) {
        Some(f) => f,
        None => return err_response("FFmpeg no está configurado.", collect_logs(&logs)),
    };

    let formato = req.format.trim().to_lowercase();
    let output = match req.output.filter(|s| !s.trim().is_empty()) {
        Some(o) => PathBuf::from(o),
        None => {
            let stem = input
                .file_stem()
                .map(|s| s.to_string_lossy().to_string())
                .unwrap_or_else(|| "video".into());
            input
                .parent()
                .unwrap_or_else(|| std::path::Path::new("."))
                .join(format!("{stem}_convertido.{formato}"))
        }
    };

    match video::convert_video(&ffmpeg, &input, &output, &log) {
        Ok(()) => ok_response(
            json!({ "output": output.display().to_string(), "bytes": std::fs::metadata(&output).map(|m| m.len()).unwrap_or(0) }),
            collect_logs(&logs),
        ),
        Err(e) => err_response(&e.to_string(), collect_logs(&logs)),
    }
}

/// Recorta un video entre un instante y una duración.
pub fn handle_trim(state: &State, req: TrimRequest) -> serde_json::Value {
    let logs: Mutex<Vec<String>> = Mutex::new(Vec::new());
    let log = |msg: &str| {
        state.push_log(msg);
        if let Ok(mut buf) = logs.lock() {
            buf.push(msg.to_string());
        }
    };

    let input = PathBuf::from(&req.input);
    if !input.is_file() {
        return err_response(
            &format!("No existe el video: {}", input.display()),
            collect_logs(&logs),
        );
    }
    if req.start.trim().is_empty() || req.duration.trim().is_empty() {
        return err_response(
            "Indica el inicio y la duración del recorte.",
            collect_logs(&logs),
        );
    }

    let cfg = state.config.lock().unwrap().clone();
    let ffmpeg = match cfg.ffmpeg_exe.clone().filter(|p| p.is_file()) {
        Some(f) => f,
        None => return err_response("FFmpeg no está configurado.", collect_logs(&logs)),
    };

    let output = match req.output.filter(|s| !s.trim().is_empty()) {
        Some(o) => PathBuf::from(o),
        None => {
            let stem = input
                .file_stem()
                .map(|s| s.to_string_lossy().to_string())
                .unwrap_or_else(|| "video".into());
            input
                .parent()
                .unwrap_or_else(|| std::path::Path::new("."))
                .join(format!("{stem}_recortado.mp4"))
        }
    };

    match video::trim_video(
        &ffmpeg,
        &input,
        &output,
        req.start.trim(),
        req.duration.trim(),
        &log,
    ) {
        Ok(()) => ok_response(
            json!({ "output": output.display().to_string(), "bytes": std::fs::metadata(&output).map(|m| m.len()).unwrap_or(0) }),
            collect_logs(&logs),
        ),
        Err(e) => err_response(&e.to_string(), collect_logs(&logs)),
    }
}

/// Guarda un archivo subido (base64) en la carpeta temporal de la app.
pub fn handle_upload(state: &State, req: UploadRequest) -> serde_json::Value {
    let _ = state;
    let dir = State::uploads_dir();
    if let Err(e) = std::fs::create_dir_all(&dir) {
        return err_response(&e.to_string(), Vec::new());
    }
    let nombre: String = req
        .name
        .rsplit(['\\', '/'])
        .next()
        .unwrap_or("imagen.png")
        .chars()
        .map(|c| {
            if c.is_alphanumeric() || matches!(c, '.' | '-' | '_' | ' ') {
                c
            } else {
                '_'
            }
        })
        .collect();
    let nombre = nombre.trim_matches(|c| c == '.' || c == ' ').to_string();
    let nombre = if nombre.is_empty() {
        "imagen.png".to_string()
    } else {
        nombre
    };
    let ext = Path::new(&nombre)
        .extension()
        .map(|e| e.to_string_lossy().to_lowercase())
        .unwrap_or_default();
    if !["png", "jpg", "jpeg", "webp", "bmp", "tif", "tiff", "gif"].contains(&ext.as_str()) {
        return err_response(
            "Solo se pueden subir imágenes (png, jpg, webp, bmp, tif, gif).",
            Vec::new(),
        );
    }
    let path = dir.join(format!(
        "{}_{}",
        crate::state::now_ms() % 100_000_000,
        nombre
    ));

    let clean = req
        .data
        .split_once(',')
        .map(|(_, rest)| rest.to_string())
        .unwrap_or(req.data.clone());
    let bytes = match base64::engine::general_purpose::STANDARD.decode(clean.as_bytes()) {
        Ok(b) => b,
        Err(e) => return err_response(&format!("Base64 inválido: {e}"), Vec::new()),
    };
    if image::guess_format(&bytes).is_err() {
        return err_response("El archivo no parece una imagen válida.", Vec::new());
    }
    if let Err(e) = std::fs::write(&path, &bytes) {
        return err_response(&e.to_string(), Vec::new());
    }
    ok_response(json!({ "path": path.display().to_string() }), Vec::new())
}

/// Compatibilidad de los modelos con el hardware (por GPU) según la preferencia actual.
pub fn handle_compat(state: &State) -> serde_json::Value {
    let cfg = state.config.lock().unwrap().clone();
    let informe = hardware::detect_cached();
    let rep = reescaller_core::compat::check(informe, cfg.device, cfg.gpu_index);
    ok_response(serde_json::to_value(&rep).unwrap_or(json!({})), Vec::new())
}

/// Prueba real (imagen diminuta) de cada modelo del motor en la GPU elegida.
pub fn handle_compat_test(state: &State) -> serde_json::Value {
    let cfg = state.config.lock().unwrap().clone();
    let informe = hardware::detect_cached();
    let Some(engine) = cfg.realesrgan_exe.clone().filter(|p| p.is_file()) else {
        return err_response("El motor de IA no está configurado.", Vec::new());
    };
    if cfg.device == reescaller_core::config::DevicePreference::Cpu || !informe.vulkan_available {
        return err_response(
            "No hay una GPU compatible con Vulkan: no se puede probar el motor de IA.",
            Vec::new(),
        );
    }
    let (indice, _) = hardware::resolve_preference(cfg.device, informe, cfg.gpu_index);
    let gpu_args = hardware::engine_gpu_args(informe, indice);
    let work = std::env::temp_dir().join("reescaller_compat");
    let res = reescaller_core::compat::probe_models(&engine, &gpu_args, &work);
    ok_response(json!({ "results": res }), Vec::new())
}

/// Lee una imagen del portapapeles del sistema (la guarda como PNG en la carpeta de subidas) o devuelve la
/// ruta del archivo de imagen copiado/cortado.
pub fn handle_clipboard_image() -> serde_json::Value {
    use crate::clipboard::Clip;
    let guardar = |png: Vec<u8>| -> serde_json::Value {
        let dir = State::uploads_dir();
        if let Err(e) = std::fs::create_dir_all(&dir) {
            return err_response(&e.to_string(), Vec::new());
        }
        let path = dir.join(format!(
            "{}_pegado.png",
            crate::state::now_ms() % 100_000_000
        ));
        match std::fs::write(&path, &png) {
            Ok(()) => ok_response(json!({ "path": path.display().to_string() }), Vec::new()),
            Err(e) => err_response(&e.to_string(), Vec::new()),
        }
    };
    match crate::clipboard::read() {
        Ok(Some(Clip::File(p))) => {
            ok_response(json!({ "path": p.display().to_string() }), Vec::new())
        }
        Ok(Some(Clip::Png(bytes))) => guardar(bytes),
        Ok(Some(Clip::Bmp(bytes))) => match image::load_from_memory(&bytes) {
            Ok(img) => {
                let mut png = Vec::new();
                match img.write_to(&mut std::io::Cursor::new(&mut png), image::ImageFormat::Png) {
                    Ok(()) => guardar(png),
                    Err(e) => err_response(
                        &format!("No se pudo convertir la imagen pegada: {e}"),
                        Vec::new(),
                    ),
                }
            }
            Err(e) => err_response(
                &format!("No se pudo leer la imagen pegada: {e}"),
                Vec::new(),
            ),
        },
        Ok(None) => err_response("No hay ninguna imagen en el portapapeles.", Vec::new()),
        Err(e) => err_response(&e, Vec::new()),
    }
}

/// Lista el contenido de una carpeta para el explorador de archivos.
pub fn handle_browse(state: &State, path: &str) -> serde_json::Value {
    let target = if path.is_empty() {
        state.project_dir.clone()
    } else {
        PathBuf::from(path)
    };
    let mut entries: Vec<serde_json::Value> = Vec::new();
    if let Ok(dir) = std::fs::read_dir(&target) {
        for entry in dir.flatten() {
            let p = entry.path();
            let is_dir = p.is_dir();
            let name = p
                .file_name()
                .map(|n| n.to_string_lossy().to_string())
                .unwrap_or_default();
            if name.starts_with('.') {
                continue;
            }
            let size = entry.metadata().map(|m| m.len()).unwrap_or(0);
            entries.push(json!({
                "name": name,
                "path": p.display().to_string(),
                "isDir": is_dir,
                "size": size,
            }));
        }
    }
    entries.sort_by(|a, b| {
        let ad = a["isDir"].as_bool().unwrap_or(false);
        let bd = b["isDir"].as_bool().unwrap_or(false);
        bd.cmp(&ad).then_with(|| {
            a["name"]
                .as_str()
                .unwrap_or("")
                .cmp(b["name"].as_str().unwrap_or(""))
        })
    });

    let parent = target.parent().map(|p| p.display().to_string());
    ok_response(
        json!({
            "path": target.display().to_string(),
            "parent": parent,
            "entries": entries,
        }),
        Vec::new(),
    )
}

/// Devuelve una miniatura en base64 (PNG) de una imagen.
pub fn handle_preview(path: &str, max_side: u32) -> serde_json::Value {
    if es_ruta_red(path) {
        return err_response("No se admiten rutas de red.", Vec::new());
    }
    let max_side = max_side.min(4096);
    let file = Path::new(path);
    if !file.is_file() {
        return err_response(&format!("No existe: {path}"), Vec::new());
    }
    match image_ops::load(file) {
        Ok(img) => {
            let thumb = image_ops::thumbnail(&img, max_side.max(64));
            let data =
                image_ops::encode(&thumb, OutputFormat::Png, Weight::Quality { quality: 90 });
            match data {
                Ok(bytes) => ok_response(
                    json!({
                        "data": base64::engine::general_purpose::STANDARD.encode(&bytes),
                        "mime": "image/png",
                        "width": thumb.width(),
                        "height": thumb.height(),
                        "bytes": bytes.len(),
                    }),
                    Vec::new(),
                ),
                Err(e) => err_response(&e.to_string(), Vec::new()),
            }
        }
        Err(e) => err_response(&e.to_string(), Vec::new()),
    }
}

/// Devuelve la configuración actual.
pub fn handle_get_config(state: &State) -> serde_json::Value {
    let cfg = state.config.lock().unwrap().clone();
    ok_response(serde_json::to_value(&cfg).unwrap_or(json!({})), Vec::new())
}

/// Guarda la configuración.
pub fn handle_set_config(state: &State, cfg: AppConfig) -> serde_json::Value {
    let mut cfg = cfg;
    // Solo se aceptan programas con el nombre esperado: la configuración no puede apuntar a cualquier ejecutable.
    let nombre = |p: &Option<PathBuf>| {
        p.as_ref()
            .and_then(|p| p.file_name())
            .map(|n| n.to_string_lossy().to_lowercase())
    };
    if let Some(n) = nombre(&cfg.realesrgan_exe) {
        if !(n.contains("realesrgan") && n.ends_with(".exe")) {
            return err_response(
                "El motor de IA debe ser realesrgan-ncnn-vulkan.exe.",
                Vec::new(),
            );
        }
    }
    if let Some(n) = nombre(&cfg.ffmpeg_exe) {
        if n != "ffmpeg.exe" {
            return err_response("FFmpeg debe ser ffmpeg.exe.", Vec::new());
        }
    }
    if let Some(n) = nombre(&cfg.python_exe) {
        if !(n.starts_with("python") && n.ends_with(".exe")) {
            return err_response("Python debe ser python.exe.", Vec::new());
        }
    }
    cfg.validar();
    let saved = cfg.save();
    *state.config.lock().unwrap() = cfg.clone();
    ok_response(
        json!({
            "saved": saved.is_ok(),
            "path": AppConfig::config_path().display().to_string(),
            "config": serde_json::to_value(&cfg).unwrap_or(json!({})),
        }),
        Vec::new(),
    )
}

/// Diagnóstico del entorno.
pub fn handle_doctor(state: &State) -> serde_json::Value {
    // El diagnóstico del puente necesita el script en disco.
    let _ = ensure_bridge_script();
    let cfg = state.config.lock().unwrap().clone();
    let report = diagnostics::doctor(&cfg, &state.project_dir);
    ok_response(
        serde_json::to_value(&report).unwrap_or(json!({})),
        Vec::new(),
    )
}

/// Informe de hardware: CPU, RAM, GPU (NVIDIA/AMD/Intel) y Vulkan.
pub fn handle_hardware(state: &State, refresh: bool) -> serde_json::Value {
    let cfg = state.config.lock().unwrap().clone();
    let informe = if refresh {
        hardware::refresh()
    } else {
        hardware::detect_cached()
    };
    let (indice, avisos) = hardware::resolve_preference(cfg.device, informe, cfg.gpu_index);
    let gpu = indice.and_then(|i| informe.gpus.iter().find(|g| g.index == i));

    ok_response(
        json!({
            "informe": informe,
            "configurado": cfg.device,
            "gpuElegida": gpu,
            "avisos": avisos,
            "notas": informe.notes,
            "sugerencia": match informe.engine_support {
                hardware::EngineSupport::Full => "El motor de mejora por IA está operativo.".to_string(),
                hardware::EngineSupport::CpuOnly =>
                    "Sin Vulkan: el motor de IA se desactiva y se usa el reescalado por CPU.".to_string(),
            },
        }),
        Vec::new(),
    )
}

// ===========================================================================
// Modo fácil, selectores nativos y utilidades de escritorio
// ===========================================================================

/// Petición del modo fácil de remasterización.
#[derive(Deserialize)]
pub struct RemasterRequest {
    /// Imagen o carpeta de imágenes.
    pub input: String,
    /// Carpeta de salida (opcional).
    #[serde(default)]
    pub output_dir: Option<String>,
    #[serde(default)]
    pub options: remaster::RemasterOptions,
    /// Solo analiza y devuelve el plan, sin procesar.
    #[serde(default)]
    pub dry_run: bool,
    /// Solo convertir el formato (sin IA ni cambio de tamaño).
    #[serde(default)]
    pub convert_to: Option<OutputFormat>,
}

/// Devuelve `ruta` si no existe; si existe añade « (2)», « (3)»...
fn unique_path(ruta: PathBuf) -> PathBuf {
    if !ruta.exists() {
        return ruta;
    }
    let stem = ruta
        .file_stem()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| "imagen".into());
    let ext = ruta
        .extension()
        .map(|e| e.to_string_lossy().to_string())
        .unwrap_or_default();
    let dir = ruta.parent().map(|p| p.to_path_buf()).unwrap_or_default();
    for n in 2..1000 {
        let cand = dir.join(format!("{stem} ({n}).{ext}"));
        if !cand.exists() {
            return cand;
        }
    }
    ruta
}

fn plan_json(a: &remaster::Analysis, p: &remaster::RemasterPlan) -> serde_json::Value {
    let destino = p.plan.target_dimensions((a.width, a.height));
    json!({
        "analysis": a,
        "explanation": p.explanation,
        "model": p.plan.model,
        "targetScale": p.plan.target_scale,
        "improvePercent": p.plan.improve_percent,
        "sharpen": p.plan.sharpen,
        "format": p.plan.output_format.as_str(),
        "weight": p.plan.weight.describe(),
        "quality": match p.plan.weight {
            Weight::Quality { quality } => Some(quality),
            _ => None,
        },
        "resizeMode": p.plan.resize_mode.as_str(),
        "useEngine": p.plan.use_engine,
        "finalWidth": destino.0,
        "finalHeight": destino.1,
    })
}

/// Nombre de salida de una imagen del modo fácil.
fn remaster_output(
    state: &State,
    input: &Path,
    dir: Option<&str>,
    fmt: OutputFormat,
    suffix: &str,
) -> PathBuf {
    let carpeta = match dir.filter(|d| !d.trim().is_empty()) {
        Some(d) => PathBuf::from(d),
        None => {
            if input.starts_with(State::uploads_dir()) {
                State::default_outputs_dir()
            } else {
                input
                    .parent()
                    .map(|p| p.to_path_buf())
                    .unwrap_or_else(|| state.project_dir.clone())
            }
        }
    };
    let _ = std::fs::create_dir_all(&carpeta);
    let stem = input
        .file_stem()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| "imagen".into());
    unique_path(carpeta.join(format!("{stem}{suffix}.{}", fmt.extension())))
}

/// Plan del modo fácil; con `convert` solo cambia el formato.
fn plan_for(
    opts: &remaster::RemasterOptions,
    a: &remaster::Analysis,
    convert: Option<OutputFormat>,
) -> remaster::RemasterPlan {
    let mut rp = remaster::build_plan(opts, a);
    if let Some(fmt) = convert {
        let p = &mut rp.plan;
        p.use_engine = false;
        p.resize_mode = ResizeMode::ImproveOnly;
        p.target_scale = 1.0;
        p.improve_percent = 100;
        p.sharpen = 0;
        p.output_format = fmt;
        p.weight = Weight::Quality { quality: 92 };
        rp.explanation = vec![format!(
            "Conversión a {} sin cambiar el tamaño ni aplicar IA.",
            fmt.as_str().to_uppercase()
        )];
    }
    rp
}

fn sufijo(convert: Option<OutputFormat>) -> &'static str {
    if convert.is_some() {
        "_convertido"
    } else {
        "_mejorado"
    }
}

fn outcome_json(o: &upscale::UpscaleOutcome) -> serde_json::Value {
    json!({
        "output": o.output.display().to_string(),
        "originalWidth": o.original_size.0,
        "originalHeight": o.original_size.1,
        "finalWidth": o.final_size.0,
        "finalHeight": o.final_size.1,
        "bytes": o.bytes,
        "engineApplied": o.engine_applied,
        "engineScale": o.engine_scale,
    })
}

/// Remasterización en un clic: analiza, decide y procesa una imagen o carpeta.
pub fn handle_remaster(state: &State, req: RemasterRequest) -> serde_json::Value {
    let logs: Mutex<Vec<String>> = Mutex::new(Vec::new());
    let log = |msg: &str| {
        state.push_log(msg);
        if let Ok(mut buf) = logs.lock() {
            buf.push(msg.to_string());
        }
    };

    let convert = req.convert_to;
    let cfg = state.config.lock().unwrap().clone();
    let mut opts = req.options.clone();
    opts.device = cfg.device;
    opts.gpu_index = cfg.gpu_index;

    let input = PathBuf::from(req.input.trim());
    let engine = match cfg.realesrgan_exe.clone().filter(|p| p.is_file()) {
        Some(p) => p,
        None => {
            return err_response(
                "No se encontró el motor de IA (realesrgan-ncnn-vulkan.exe). Revisa la pestaña Sistema.",
                collect_logs(&logs),
            )
        }
    };

    // ---------------------------------------------------------------- carpeta
    if input.is_dir() {
        let files = reescaller_core::batch::list_images(&input);
        if files.is_empty() {
            return err_response(
                "La carpeta no contiene imágenes (png, jpg, webp, bmp, tif).",
                collect_logs(&logs),
            );
        }
        let out_dir = match req.output_dir.as_deref().filter(|d| !d.trim().is_empty()) {
            Some(d) => PathBuf::from(d),
            None => input.join("Mejoradas"),
        };
        if req.dry_run {
            let muestra = remaster::analyze(&files[0]).ok();
            let plan = muestra
                .as_ref()
                .map(|a| plan_json(a, &plan_for(&opts, a, convert)));
            return ok_response(
                json!({
                    "kind": "folder",
                    "count": files.len(),
                    "outputDir": out_dir.display().to_string(),
                    "sample": plan
                }),
                Vec::new(),
            );
        }
        if let Err(e) = std::fs::create_dir_all(&out_dir) {
            return err_response(
                &format!("No se pudo crear la carpeta de salida: {e}"),
                collect_logs(&logs),
            );
        }

        let informe = hardware::detect_cached();
        let workers = informe.recommended_workers.clamp(1, 4).min(files.len());
        let total = files.len();
        log(&format!(
            "Remasterizando {total} imágenes ({workers} en paralelo)..."
        ));
        let siguiente = std::sync::atomic::AtomicUsize::new(0);
        let hechas = std::sync::atomic::AtomicUsize::new(0);
        let items: Mutex<Vec<serde_json::Value>> = Mutex::new(Vec::new());
        let carpeta_salida = out_dir.display().to_string();
        let sink = |m: &str| state.push_log(m);

        std::thread::scope(|scope| {
            for _ in 0..workers {
                scope.spawn(|| loop {
                    let i = siguiente.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                    let Some(file) = files.get(i) else { break };
                    let resultado =
                        remaster::analyze(file)
                            .map_err(|e| e.to_string())
                            .and_then(|a| {
                                let rp = plan_for(&opts, &a, convert);
                                let salida = remaster_output(
                                    state,
                                    file,
                                    Some(&carpeta_salida),
                                    rp.plan.output_format,
                                    sufijo(convert),
                                );
                                upscale::upscale_image(&rp.plan, &engine, file, &salida, &sink)
                                    .map_err(|e| e.to_string())
                            });
                    let n = hechas.fetch_add(1, std::sync::atomic::Ordering::SeqCst) + 1;
                    state.push_log(&format!("Progreso: {n}/{total}"));
                    let item = match resultado {
                        Ok(o) => {
                            let mut v = outcome_json(&o);
                            v["input"] = json!(file.display().to_string());
                            v["ok"] = json!(true);
                            v
                        }
                        Err(e) => {
                            json!({ "input": file.display().to_string(), "ok": false, "error": e })
                        }
                    };
                    items.lock().unwrap().push(item);
                });
            }
        });
        let items = items.into_inner().unwrap();
        let ok_count = items.iter().filter(|v| v["ok"] == json!(true)).count();
        return ok_response(
            json!({
                "kind": "folder",
                "count": total,
                "okCount": ok_count,
                "failCount": total - ok_count,
                "outputDir": out_dir.display().to_string(),
                "items": items,
            }),
            collect_logs(&logs),
        );
    }

    // ----------------------------------------------------------------- imagen
    if !input.is_file() {
        return err_response(
            &format!("No existe: {}", input.display()),
            collect_logs(&logs),
        );
    }
    let analysis = match remaster::analyze(&input) {
        Ok(a) => a,
        Err(e) => {
            return err_response(
                &format!("No se pudo leer la imagen: {e}"),
                collect_logs(&logs),
            );
        }
    };
    let rp = plan_for(&opts, &analysis, convert);
    let mut resumen = plan_json(&analysis, &rp);
    if req.dry_run {
        resumen["kind"] = json!("image");
        return ok_response(resumen, Vec::new());
    }

    let salida = remaster_output(
        state,
        &input,
        req.output_dir.as_deref(),
        rp.plan.output_format,
        sufijo(convert),
    );
    for linea in &rp.explanation {
        log(linea);
    }
    match upscale::upscale_image(&rp.plan, &engine, &input, &salida, &log) {
        Ok(o) => {
            let mut v = outcome_json(&o);
            v["kind"] = json!("image");
            v["plan"] = resumen;
            v["outputDir"] = json!(salida
                .parent()
                .map(|p| p.display().to_string())
                .unwrap_or_default());
            ok_response(v, collect_logs(&logs))
        }
        Err(e) => err_response(&e.to_string(), collect_logs(&logs)),
    }
}

/// Ejecuta un script de PowerShell en un hilo STA (necesario para los diálogos).
fn powershell_sta(script: &str) -> Option<String> {
    let mut cmd = std::process::Command::new("powershell.exe");
    cmd.args([
        "-NoProfile",
        "-NonInteractive",
        "-STA",
        "-ExecutionPolicy",
        "Bypass",
        "-Command",
        &format!("[Console]::OutputEncoding=[Text.Encoding]::UTF8; {script}"),
    ])
    .stdin(std::process::Stdio::null());
    reescaller_core::process::sin_ventana(&mut cmd);
    let out = cmd.output().ok()?;
    Some(String::from_utf8_lossy(&out.stdout).to_string())
}

/// Abre el diálogo nativo de Windows para elegir archivos o carpetas.
pub fn handle_pick(kind: &str, multi: bool) -> serde_json::Value {
    let owner = "$f=New-Object System.Windows.Forms.Form;$f.TopMost=$true;";
    let script = match kind {
        "folder" => format!(
            "Add-Type -AssemblyName System.Windows.Forms;{owner}\
             $d=New-Object System.Windows.Forms.FolderBrowserDialog;\
             $d.Description='Elige una carpeta';$d.ShowNewFolderButton=$true;\
             if($d.ShowDialog($f) -eq 'OK'){{$d.SelectedPath}}"
        ),
        other => {
            let filtro = match other {
                "video" => "Vídeos|*.mp4;*.mkv;*.avi;*.mov;*.webm;*.m4v|Todos los archivos|*.*",
                "model" => "Modelos (*.pth, *.ckpt)|*.pth;*.ckpt|Todos los archivos|*.*",
                "exe" => "Programas (*.exe)|*.exe|Todos los archivos|*.*",
                _ => "Imágenes|*.png;*.jpg;*.jpeg;*.webp;*.bmp;*.tif;*.tiff|Todos los archivos|*.*",
            };
            format!(
                "Add-Type -AssemblyName System.Windows.Forms;{owner}\
                 $d=New-Object System.Windows.Forms.OpenFileDialog;\
                 $d.Filter='{filtro}';$d.Multiselect=${};\
                 if($d.ShowDialog($f) -eq 'OK'){{$d.FileNames}}",
                if multi { "true" } else { "false" }
            )
        }
    };
    match powershell_sta(&script) {
        Some(texto) => {
            let paths: Vec<String> = texto
                .lines()
                .map(|l| l.trim().to_string())
                .filter(|l| !l.is_empty())
                .collect();
            ok_response(json!({ "paths": paths }), Vec::new())
        }
        None => err_response(
            "No se pudo abrir el selector de archivos de Windows.",
            Vec::new(),
        ),
    }
}

/// Muestra un archivo en el Explorador de Windows (o abre la carpeta).
/// Extensiones de medios que la app puede abrir con el programa predeterminado. Nunca se abre un ejecutable o script.
const ABRIR_PERMITIDAS: [&str; 15] = [
    "png", "jpg", "jpeg", "webp", "bmp", "tif", "tiff", "gif", "mp4", "mkv", "mov", "avi", "webm",
    "mp3", "m4v",
];

/// Rutas de red (`\\servidor\recurso`): pueden filtrar credenciales y bloquear la app.
fn es_ruta_red(p: &str) -> bool {
    let n = p.replace('/', "\\");
    n.starts_with("\\\\")
}

pub fn handle_reveal(path: &str, open: bool) -> serde_json::Value {
    // Explorer exige barras invertidas; la ruta puede llegar mezclada («C:/a\\b»).
    let norm = path.replace('/', "\\");
    if es_ruta_red(&norm) {
        return err_response("No se admiten rutas de red.", Vec::new());
    }
    let p = PathBuf::from(&norm);
    if !p.exists() {
        return err_response(&format!("No existe: {path}"), Vec::new());
    }
    if open && p.is_file() {
        let ext = p
            .extension()
            .map(|e| e.to_string_lossy().to_lowercase())
            .unwrap_or_default();
        if !ABRIR_PERMITIDAS.contains(&ext.as_str()) {
            return err_response("Solo se pueden abrir imágenes, videos o audio.", Vec::new());
        }
    }
    let mut cmd = std::process::Command::new("explorer.exe");
    if p.is_file() && !open {
        // `/select,"ruta"`: las comillas deben ir solo alrededor de la ruta. Con
        // `arg()` Rust entrecomillaría todo el argumento cuando hay espacios y
        // Explorer abriría «Documentos» en vez de seleccionar el archivo.
        #[cfg(windows)]
        {
            use std::os::windows::process::CommandExt;
            cmd.raw_arg(format!("/select,\"{norm}\""));
        }
        #[cfg(not(windows))]
        cmd.arg(format!("/select,{norm}"));
    } else {
        cmd.arg(&norm);
    }
    match cmd.spawn() {
        Ok(_) => ok_response(json!({ "path": norm }), Vec::new()),
        Err(e) => err_response(&e.to_string(), Vec::new()),
    }
}

/// Script de Python del puente de rostros, embebido en el ejecutable.
const BRIDGE_SCRIPT: &str = include_str!("../../../../scripts/restore_face.py");

/// Escribe el puente en la carpeta de datos del usuario y devuelve su ruta.
fn ensure_bridge_script() -> std::io::Result<PathBuf> {
    let dir = State::data_dir().join("bridge");
    std::fs::create_dir_all(&dir)?;
    let path = dir.join("restore_face.py");
    let actual = std::fs::read_to_string(&path).unwrap_or_default();
    if actual != BRIDGE_SCRIPT {
        std::fs::write(&path, BRIDGE_SCRIPT)?;
    }
    Ok(path)
}

/// Vuelve a buscar motor, FFmpeg, modelos y Python, y guarda lo encontrado.
pub fn handle_rescan(state: &State) -> serde_json::Value {
    let mut cfg = state.config.lock().unwrap().clone();
    cfg.autodetectar(&state.project_dir);
    let _ = cfg.save();
    *state.config.lock().unwrap() = cfg.clone();
    ok_response(serde_json::to_value(&cfg).unwrap_or(json!({})), Vec::new())
}
