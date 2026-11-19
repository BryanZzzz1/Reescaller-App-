//! Pipeline de reescalado controlado.
//!
//! Reemplaza a `ejecutar_realesrgan_unico()` del proyecto Python, pero con
//! control del **cuándo** y **cuánto** se reescala y del **peso** final.
//!
//! Flujo:
//! 1. El motor (Real-ESRGAN) escala a su máxima capacidad (x2 o x4).
//! 2. El resultado se reescala al tamaño final elegido por el usuario
//!    (porcentaje sobre el original), en lugar de quedarse en el máximo.
//! 3. Se mezcla con el original según el porcentaje de mejora.
//! 4. Se aplica nitidez opcional.
//! 5. Se codifica con el "peso" elegido (calidad fija o tamaño objetivo).

use crate::config::{DevicePreference, OutputFormat, ResizeFilter, ResizeMode, Weight};
use crate::error::{Error, Result};
use crate::hardware;
use crate::image_ops;
use crate::process;
use image::GenericImageView;
use std::path::{Path, PathBuf};

/// Plan de reescalado de una imagen.
#[derive(Debug, Clone)]
pub struct UpscalePlan {
    /// Escala deseada del motor (2 o 4). El motor solo admite 2, 3 y 4.
    pub engine_scale: u8,
    /// Modelo de Real-ESRGAN.
    pub model: String,
    /// Escala final respecto al original (1.0 = mismo tamaño, 2.0 = x2).
    pub target_scale: f32,
    /// Porcentaje de mejora (100 = solo la imagen mejorada).
    pub improve_percent: u8,
    /// Peso de la imagen final.
    pub weight: Weight,
    /// Formato de salida.
    pub output_format: OutputFormat,
    /// Cuándo reescalar.
    pub resize_mode: ResizeMode,
    /// Filtro de interpolación.
    pub resize_filter: ResizeFilter,
    /// Nitidez extra (0..=100).
    pub sharpen: u8,
    /// Usar el motor de IA. Si es `false` solo se reescala (modo rápido).
    pub use_engine: bool,
    /// Dispositivo preferido para el motor (auto / nvidia / amd / intel / cpu).
    pub device: DevicePreference,
    /// Índice de GPU forzado (`-g`). `None` = automática.
    pub gpu_index: Option<usize>,
}

impl Default for UpscalePlan {
    fn default() -> Self {
        Self {
            engine_scale: 4,
            model: "realesrgan-x4plus".to_string(),
            target_scale: 2.0,
            improve_percent: 100,
            weight: Weight::default(),
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

impl UpscalePlan {
    /// Escala real que se puede pedir al motor según el modelo.
    ///
    /// `realesrgan-x4plus` solo escala x4; `realesr-animevideov3` admite 2, 3 y 4.
    pub fn effective_engine_scale(&self) -> u8 {
        let requested = self.engine_scale.clamp(2, 4);
        if self.model.starts_with("realesrgan-x4plus") {
            4
        } else {
            requested
        }
    }

    /// Tamaño final de la imagen según el modo de reescalado.
    pub fn target_dimensions(&self, original: (u32, u32)) -> (u32, u32) {
        let requested = image_ops::scaled_size(original, self.target_scale);
        match self.resize_mode {
            ResizeMode::Always => requested,
            ResizeMode::OnlyIfNeeded => {
                if original.0 < requested.0 {
                    requested
                } else {
                    original
                }
            }
            ResizeMode::ImproveOnly => original,
        }
    }
}

/// Resultado de procesar una imagen.
#[derive(Debug, Clone)]
pub struct UpscaleOutcome {
    pub output: PathBuf,
    pub original_size: (u32, u32),
    pub final_size: (u32, u32),
    pub bytes: u64,
    pub engine_applied: bool,
    pub engine_scale: u8,
    /// Peso realmente aplicado (puede ajustarse en modo `TargetSize`).
    pub weight: Weight,
    /// Índice de GPU usada (si el motor se aplicó).
    pub gpu_index: Option<usize>,
}

/// Procesa una imagen con el plan dado.
pub fn upscale_image(
    plan: &UpscalePlan,
    engine_exe: &Path,
    input: &Path,
    output: &Path,
    log: &(dyn Fn(&str) + Send + Sync),
) -> Result<UpscaleOutcome> {
    if !input.is_file() {
        return Err(Error::FileNotFound(input.to_path_buf()));
    }

    let original = image_ops::load(input)?;
    let original_size = original.dimensions();
    let target = plan.target_dimensions(original_size);

    log(&format!(
        "Entrada: {} ({}x{})",
        input.display(),
        original_size.0,
        original_size.1
    ));
    log(&format!(
        "Objetivo: {}x{} | modo: {} | mejora: {}% | peso: {}",
        target.0,
        target.1,
        plan.resize_mode.as_str(),
        plan.improve_percent,
        plan.weight.describe()
    ));

    // --- Paso 1: motor de IA -------------------------------------------------
    let engine_output = output.with_extension(format!(
        "{}_engine.{}",
        output
            .file_stem()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| "temp".into()),
        if plan.use_engine { "png" } else { "tmp" }
    ));

    let mut engine_applied = false;
    let mut engine_scale = plan.effective_engine_scale();

    // --- Selección de GPU ---------------------------------------------------
    // Se consulta el hardware una sola vez (está cacheado) y se decide:
    //  · CPU explícito  -> no hay motor de IA.
    //  · sin Vulkan     -> tampoco hay motor de IA (aviso claro al usuario).
    //  · gpu concreta   -> se pasa `-g <índice>` al motor.
    let informe = hardware::detect_cached();
    let mut usar_motor = plan.use_engine;
    let mut indice_gpu = plan.gpu_index;

    if usar_motor {
        if plan.device == DevicePreference::Cpu {
            log("⚠ Dispositivo configurado como «solo CPU»: se desactiva la mejora por IA.");
            usar_motor = false;
        } else if !informe.vulkan_available {
            log("⚠ No se detectó Vulkan en este equipo: se desactiva la mejora por IA y se trabaja por CPU.");
            usar_motor = false;
        } else {
            let (idx, avisos) = hardware::resolve_preference(plan.device, informe, plan.gpu_index);
            indice_gpu = idx;
            for aviso in avisos {
                log(&format!("⚠ {aviso}"));
            }
            if let Some(g) = informe.gpus.iter().find(|g| Some(g.index) == idx) {
                log(&format!(
                    "GPU seleccionada: {} (índice {})",
                    g.describe(),
                    g.index
                ));
            }
        }
    }

    if usar_motor {
        let base_args = vec![
            "-i".to_string(),
            input.display().to_string(),
            "-o".to_string(),
            engine_output.display().to_string(),
            "-s".to_string(),
            engine_scale.to_string(),
            "-n".to_string(),
            plan.model.clone(),
        ];
        let gpu_args = hardware::engine_gpu_args(informe, indice_gpu);

        // Intentos: lo planeado -> teselas más pequeñas (poca VRAM) -> GPU
        // automática con teselas mínimas. Si todo falla se sigue sin IA.
        let mut intentos: Vec<(Vec<String>, &str)> = vec![(gpu_args.clone(), "")];
        intentos.push((
            [gpu_args.clone(), vec!["-t".into(), "100".into()]].concat(),
            "teselas de 100 px",
        ));
        intentos.push((
            vec!["-t".into(), "32".into()],
            "GPU automática y teselas de 32 px",
        ));

        let mut ultimo_error: Option<Error> = None;
        for (extra, motivo) in intentos {
            if !motivo.is_empty() {
                log(&format!("Reintentando con {motivo}..."));
            }
            let mut args = base_args.clone();
            args.extend(extra);
            match process::run_with_log(engine_exe, &args, log) {
                Ok(_) if engine_output.is_file() => {
                    engine_applied = true;
                    ultimo_error = None;
                    break;
                }
                Ok(_) => {
                    ultimo_error = Some(Error::Image(format!(
                        "El motor no generó el archivo {}",
                        engine_output.display()
                    )));
                }
                Err(e) => ultimo_error = Some(e),
            }
        }

        if let Some(e) = ultimo_error {
            log(&format!(
                "⚠ El motor de IA no pudo procesar la imagen ({e}). Se continúa solo con reescalado de alta calidad (sin IA)."
            ));
            let _ = std::fs::remove_file(&engine_output);
            engine_scale = 1;
        } else {
            let engine_img = image_ops::load(&engine_output)?;
            let engine_size = engine_img.dimensions();
            log(&format!(
                "Motor aplicado: escala x{} -> {}x{}",
                engine_scale, engine_size.0, engine_size.1
            ));
        }
    } else {
        engine_scale = 1;
    }

    // --- Paso 2: reescalado al tamaño elegido --------------------------------
    let resultado = (|| -> Result<UpscaleOutcome> {
        let base_image = match (engine_applied, image_ops::load(&engine_output)) {
            (true, Ok(img)) => img,
            (true, Err(e)) => {
                let _ = std::fs::remove_file(&engine_output);
                return Err(e);
            }
            (false, _) => original.clone(),
        };
        let base_resized = image_ops::resize(&base_image, target.0, target.1, plan.resize_filter);

        // --- Paso 3: mezcla con el original (porcentaje de mejora) ----------------
        let original_resized = image_ops::resize(&original, target.0, target.1, plan.resize_filter);
        let improved = image_ops::blend(&original_resized, &base_resized, plan.improve_percent);

        // --- Paso 4: nitidez opcional -------------------------------------------
        let final_image = image_ops::sharpen(&improved, plan.sharpen);

        // --- Paso 5: peso de la imagen final -------------------------------------
        let data = image_ops::encode(&final_image, plan.output_format, plan.weight)?;

        image_ops::save_bytes(&data, output)?;
        if engine_applied {
            let _ = std::fs::remove_file(&engine_output);
        }

        log(&format!(
            "Guardado: {} ({:.1} KB, {}x{})",
            output.display(),
            data.len() as f64 / 1024.0,
            target.0,
            target.1
        ));

        Ok(UpscaleOutcome {
            output: output.to_path_buf(),
            original_size,
            final_size: target,
            bytes: data.len() as u64,
            engine_applied,
            engine_scale,
            weight: plan.weight,
            gpu_index: indice_gpu,
        })
    })();

    // Si algo falla, no se dejan temporales sueltos en disco.
    if resultado.is_err() && engine_applied {
        let _ = std::fs::remove_file(&engine_output);
    }
    resultado
}

/// Calcula el nombre de salida por defecto para una entrada.
pub fn default_output_path(input: &Path, output_format: OutputFormat, suffix: &str) -> PathBuf {
    let stem = input
        .file_stem()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| "imagen".into());
    let dir = input.parent().unwrap_or_else(|| Path::new("."));
    dir.join(format!("{stem}{suffix}.{}", output_format.extension()))
}
