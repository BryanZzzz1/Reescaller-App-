//! Restauración de rostros mediante puente a Python (GFPGAN / RestoreFormer++).
//!
//! Los modelos de rostros siguen siendo PyTorch. La app en Rust los invoca a
//! través de un script auxiliar, de forma equivalente a como la app Python
//! llamaba a las librerías directamente. En una fase posterior estos pesos
//! pueden exportarse a ONNX para ejecutarse nativamente (ver `docs/`).

use crate::config::{AppConfig, DevicePreference};
use crate::error::{Error, Result};
use crate::face_onnx;
use crate::process;
use std::path::{Path, PathBuf};

/// Modelos de rostros soportados por el puente.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FaceModel {
    Gfpgan,
    RestoreFormer,
}

impl FaceModel {
    pub fn as_str(&self) -> &'static str {
        match self {
            FaceModel::Gfpgan => "gfpgan",
            FaceModel::RestoreFormer => "restoreformer",
        }
    }

    /// Archivo ONNX equivalente (carpeta `models/onnx`).
    pub fn onnx_file(&self) -> &'static str {
        match self {
            FaceModel::Gfpgan => "gfpgan_v14.onnx",
            FaceModel::RestoreFormer => "restoreformer_pp.onnx",
        }
    }

    pub fn parse(value: &str) -> Option<Self> {
        match value.to_lowercase().as_str() {
            "gfpgan" | "gfpganv1.4" => Some(FaceModel::Gfpgan),
            "restoreformer" | "restoreformer++" | "rf++" => Some(FaceModel::RestoreFormer),
            _ => None,
        }
    }
}

/// Script auxiliar que hace el trabajo en Python.
pub fn bridge_script() -> PathBuf {
    // scripts/restore_face.py, relativo a la raíz del proyecto.
    Path::new("scripts").join("restore_face.py").to_path_buf()
}

/// Restaura los rostros de una imagen.
pub fn restore_faces(
    python: &Path,
    script: &Path,
    model: FaceModel,
    model_path: &Path,
    input: &Path,
    output: &Path,
    log: &(dyn Fn(&str) + Send + Sync),
) -> Result<()> {
    if !python.is_file() {
        return Err(Error::ExecutableNotFound(python.to_path_buf()));
    }
    if !script.is_file() {
        return Err(Error::FileNotFound(script.to_path_buf()));
    }
    if !model_path.is_file() {
        return Err(Error::FileNotFound(model_path.to_path_buf()));
    }

    let args = vec![
        script.display().to_string(),
        "--model".to_string(),
        model.as_str().to_string(),
        "--weights".to_string(),
        model_path.display().to_string(),
        "--input".to_string(),
        input.display().to_string(),
        "--output".to_string(),
        output.display().to_string(),
    ];

    let result = process::run_with_log(python, &args, log);
    if result.is_ok() && !output.is_file() {
        return Err(Error::Image(
            "El puente de rostros terminó sin generar el archivo de salida.".into(),
        ));
    }
    result.map(|_| ())
}

/// Resultado de `restore`: qué motor se usó.
#[derive(Debug, Clone)]
pub struct Outcome {
    /// `onnx` o `python`.
    pub backend: &'static str,
    pub detail: String,
}

/// Restaura rostros eligiendo el motor según `cfg.face_backend`:
/// `auto` usa ONNX Runtime (sin Python) si están los modelos y, si falla o no están, el puente de Python.
pub fn restore(
    cfg: &AppConfig,
    base_dir: &Path,
    bridge_script: &Path,
    model: FaceModel,
    input: &Path,
    output: &Path,
    log: &(dyn Fn(&str) + Send + Sync),
) -> Result<Outcome> {
    restore_scaled(cfg, base_dir, bridge_script, model, input, output, 2, log)
}

/// Igual que [`restore`], pero con el factor de ampliación de la imagen completa (1 = sin ampliar; útil cuando
/// la imagen ya viene ampliada por el motor general).
#[allow(clippy::too_many_arguments)]
pub fn restore_scaled(
    cfg: &AppConfig,
    base_dir: &Path,
    bridge_script: &Path,
    model: FaceModel,
    input: &Path,
    output: &Path,
    upscale: u32,
    log: &(dyn Fn(&str) + Send + Sync),
) -> Result<Outcome> {
    let backend = cfg.face_backend.to_lowercase();
    let onnx_dir = if backend != "python" {
        face_onnx::find_onnx_dir(base_dir).filter(|d| face_onnx::supports(d, model))
    } else {
        None
    };
    if backend == "onnx" && onnx_dir.is_none() {
        return Err(Error::Message(
            "Faltan los modelos de rostros (carpeta models\\onnx).".into(),
        ));
    }
    if let Some(dir) = onnx_dir {
        let prefer_gpu = cfg.device != DevicePreference::Cpu;
        match face_onnx::restore_faces_onnx(&dir, model, input, output, prefer_gpu, upscale, log) {
            Ok(r) => {
                return Ok(Outcome {
                    backend: "onnx",
                    detail: format!(
                        "{} · {} cara(s) · {}x{}",
                        r.device, r.faces, r.width, r.height
                    ),
                })
            }
            Err(e) => {
                if backend == "onnx" || e.to_string().contains("No se encontraron rostros") {
                    return Err(e);
                }
                log(&format!(
                    "El motor nativo falló ({e}); se prueba el puente de Python."
                ));
            }
        }
    }

    let python = cfg.python_exe.clone().filter(|p| p.is_file()).ok_or_else(|| {
        Error::Message(
            "No hay motor de rostros disponible: faltan los modelos de rostros (carpeta models\\onnx). \
             Revisa Ajustes → Equipo y GPU."
                .into(),
        )
    })?;
    let weights = match model {
        FaceModel::Gfpgan => cfg.gfpgan_model.clone(),
        FaceModel::RestoreFormer => cfg.restoreformer_model.clone(),
    }
    .filter(|p| p.is_file())
    .ok_or_else(|| Error::Message("Modelo de rostros no encontrado.".into()))?;
    restore_faces(&python, bridge_script, model, &weights, input, output, log)?;
    Ok(Outcome {
        backend: "python",
        detail: "puente de Python (torch)".into(),
    })
}
