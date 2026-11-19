//! Procesamiento por lotes de una carpeta de imágenes.

use crate::config::{OutputFormat, ResizeMode, Weight};
use crate::error::Result;
use crate::upscale::{upscale_image, UpscaleOutcome, UpscalePlan};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

/// Extensiones de imagen soportadas.
pub const IMAGE_EXTENSIONS: &[&str] = &["png", "jpg", "jpeg", "webp", "bmp", "tif", "tiff"];

/// Lista las imágenes de una carpeta (no recursivo).
pub fn list_images(dir: &Path) -> Vec<PathBuf> {
    let mut files: Vec<PathBuf> = Vec::new();
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_file() {
                if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
                    if IMAGE_EXTENSIONS.contains(&ext.to_lowercase().as_str()) {
                        files.push(path);
                    }
                }
            }
        }
    }
    files.sort();
    files
}

/// Resultado del lote.
#[derive(Debug, Clone)]
pub struct BatchOutcome {
    pub processed: Vec<UpscaleOutcome>,
    pub failed: Vec<(PathBuf, String)>,
}

/// Procesa todas las imágenes de `input_dir` guardando en `output_dir`.
pub fn process_folder(
    plan: &UpscalePlan,
    engine_exe: &Path,
    input_dir: &Path,
    output_dir: &Path,
    max_workers: usize,
    log: &(dyn Fn(&str) + Send + Sync),
) -> Result<BatchOutcome> {
    std::fs::create_dir_all(output_dir)?;
    let files = list_images(input_dir);
    if files.is_empty() {
        log("No se encontraron imágenes en la carpeta de entrada.");
        return Ok(BatchOutcome {
            processed: Vec::new(),
            failed: Vec::new(),
        });
    }

    let total = files.len();
    log(&format!(
        "Procesando {total} imágenes con {max_workers} worker(s)..."
    ));

    let counter = Arc::new(AtomicUsize::new(0));
    let processed: Arc<Mutex<Vec<UpscaleOutcome>>> = Arc::new(Mutex::new(Vec::new()));
    let failed: Arc<Mutex<Vec<(PathBuf, String)>>> = Arc::new(Mutex::new(Vec::new()));
    let log_shared = Arc::new(Mutex::new(Vec::<String>::new()));

    let workers = max_workers.clamp(1, 16);
    let chunk_size = files.len().div_ceil(workers);

    std::thread::scope(|scope| {
        for chunk in files.chunks(chunk_size) {
            let processed = Arc::clone(&processed);
            let failed = Arc::clone(&failed);
            let counter = Arc::clone(&counter);
            let log_shared = Arc::clone(&log_shared);
            scope.spawn(move || {
                for file in chunk {
                    let output = output_in(output_dir, file, plan.output_format);
                    let local_log = |msg: &str| {
                        if let Ok(mut buf) = log_shared.lock() {
                            buf.push(msg.to_string());
                        }
                    };
                    match upscale_image(plan, engine_exe, file, &output, &local_log) {
                        Ok(outcome) => processed.lock().unwrap().push(outcome),
                        Err(e) => {
                            local_log(&format!("ERROR {}: {e}", file.display()));
                            failed.lock().unwrap().push((file.clone(), e.to_string()));
                        }
                    }
                    let done = counter.fetch_add(1, Ordering::SeqCst) + 1;
                    local_log(&format!("Progreso: {done}/{total}"));
                    if let Ok(mut buf) = log_shared.lock() {
                        for line in buf.iter() {
                            log(line);
                        }
                        buf.clear();
                    }
                }
            });
        }
    });

    let mut processed = Arc::try_unwrap(processed).unwrap().into_inner().unwrap();
    let failed = Arc::try_unwrap(failed).unwrap().into_inner().unwrap();
    processed.sort_by(|a, b| a.output.cmp(&b.output));

    log(&format!(
        "Lote terminado: {} correctas, {} fallidas",
        processed.len(),
        failed.len()
    ));
    for (file, err) in &failed {
        log(&format!("  - {}: {err}", file.display()));
    }

    Ok(BatchOutcome { processed, failed })
}

fn output_in(output_dir: &Path, input: &Path, format: OutputFormat) -> PathBuf {
    let stem = input
        .file_stem()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| "imagen".into());
    output_dir.join(format!("{stem}_mejorado.{}", format.extension()))
}

/// Genera el plan por defecto a partir de parámetros sueltos.
pub fn plan_from(
    engine_scale: u8,
    model: &str,
    target_scale: f32,
    improve_percent: u8,
    weight: Weight,
    output_format: OutputFormat,
    resize_mode: ResizeMode,
) -> UpscalePlan {
    UpscalePlan {
        engine_scale,
        model: model.to_string(),
        target_scale,
        improve_percent,
        weight,
        output_format,
        resize_mode,
        ..Default::default()
    }
}
