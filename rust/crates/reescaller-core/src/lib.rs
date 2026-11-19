//! # reescaller-core
//!
//! Núcleo en Rust de la *Herramienta de Mejora Visual*.
//!
//! Módulos:
//! - [`config`]: configuración persistente + autodetección de recursos.
//! - [`process`]: ejecución de Real-ESRGAN / FFmpeg / Python.
//! - [`image_ops`]: reescalado, mezcla de mejora, nitidez y codificación con peso.
//! - [`upscale`]: pipeline de reescalado controlado (x2, x4, %, peso).
//! - [`batch`]: procesado por lotes de carpetas.
//! - [`video`]: pipeline de video (extract → enhance → mux).
//! - [`face`]: puente a Python para GFPGAN / RestoreFormer++.
//! - [`diagnostics`]: comprobación del entorno.
//! - [`remaster`]: modo fácil (presets + análisis automático de la imagen).
//! - [`hardware`]: autodetección de CPU, RAM, GPU (NVIDIA/AMD/Intel) y Vulkan.

pub mod batch;
pub mod compat;
pub mod config;
pub mod diagnostics;
pub mod error;
pub mod face;
pub mod face_onnx;
pub mod hardware;
pub mod image_ops;
pub mod process;
pub mod remaster;
pub mod upscale;
pub mod video;

pub use config::{AppConfig, OutputFormat, ResizeFilter, ResizeMode, Weight};
pub use error::{Error, Result};
pub use hardware::{EngineSupport, GpuInfo, GpuVendor, HardwareReport};
pub use upscale::{UpscaleOutcome, UpscalePlan};

/// Versión del núcleo.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
