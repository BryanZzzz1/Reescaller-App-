//! Configuración persistente y autodetección de recursos.
//!
//! Equivalente Rust de `class Config` + `verificar_dependencias_inicio()`.
//! Se guarda en JSON: `%APPDATA%\Reescaller\config.json` (Windows) o
//! `~/.config/Reescaller/config.json` (Linux/macOS).

use crate::error::{Error, Result};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// Formato de salida de la imagen.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum OutputFormat {
    Png,
    Jpg,
}

impl OutputFormat {
    pub fn as_str(&self) -> &'static str {
        match self {
            OutputFormat::Png => "png",
            OutputFormat::Jpg => "jpg",
        }
    }

    pub fn extension(&self) -> &'static str {
        self.as_str()
    }
}

/// Filtro de interpolación usado al achievable la escala final deseada.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum ResizeFilter {
    /// Máxima calidad, más lento (recomendado).
    #[default]
    Lanczos3,
    CatmullRom,
    Bilinear,
    Nearest,
}

impl ResizeFilter {
    pub fn as_str(&self) -> &'static str {
        match self {
            ResizeFilter::Lanczos3 => "lanczos3",
            ResizeFilter::CatmullRom => "catmull_rom",
            ResizeFilter::Bilinear => "bilinear",
            ResizeFilter::Nearest => "nearest",
        }
    }
}

/// Cuándo se debe reescalar la imagen.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum ResizeMode {
    /// Siempre aplica la escala final elegida.
    #[default]
    Always,
    /// Solo escala si la imagen original es más pequeña que la destino.
    OnlyIfNeeded,
    /// No cambia el tamaño: solo mejora la calidad (la imagen se reescala
    /// internamente para melhorar y luego se devuelve al tamaño original).
    ImproveOnly,
}

impl ResizeMode {
    pub fn as_str(&self) -> &'static str {
        match self {
            ResizeMode::Always => "always",
            ResizeMode::OnlyIfNeeded => "only_if_needed",
            ResizeMode::ImproveOnly => "improve_only",
        }
    }
}

/// Dispositivo de cálculo preferido para el motor de IA.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum DevicePreference {
    /// Elegir automáticamente la mejor GPU disponible.
    #[default]
    Auto,
    Nvidia,
    Amd,
    Intel,
    /// No usar la GPU: solo reescalado por CPU.
    Cpu,
}

impl DevicePreference {
    pub fn as_str(&self) -> &'static str {
        match self {
            DevicePreference::Auto => "auto",
            DevicePreference::Nvidia => "nvidia",
            DevicePreference::Amd => "amd",
            DevicePreference::Intel => "intel",
            DevicePreference::Cpu => "cpu",
        }
    }

    /// Etiqueta legible para la interfaz.
    pub fn label(&self) -> &'static str {
        match self {
            DevicePreference::Auto => "Automático",
            DevicePreference::Nvidia => "NVIDIA",
            DevicePreference::Amd => "AMD",
            DevicePreference::Intel => "Intel",
            DevicePreference::Cpu => "Solo CPU",
        }
    }
}

/// "Peso" de la imagen final: cuánto pesa (calidad / tamaño) el archivo de salida.
///
/// - `Quality`: calidad fija de codificación (JPEG 1..=100, PNG = nivel de compresión 0..=9).
/// - `TargetSize`: se busca automáticamente la calidad que más se acerca al tamaño objetivo.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "mode", rename_all = "snake_case")]
pub enum Weight {
    Quality {
        /// Calidad JPEG (1..=100). Para PNG se usa como nivel de compresión invertido.
        quality: u8,
    },
    TargetSize {
        /// Tamaño objetivo del archivo en kilobytes.
        target_kb: u32,
    },
}

impl Default for Weight {
    fn default() -> Self {
        Weight::Quality { quality: 92 }
    }
}

impl Weight {
    pub fn describe(&self) -> String {
        match self {
            Weight::Quality { quality } => format!("calidad {quality}"),
            Weight::TargetSize { target_kb } => format!("peso objetivo {target_kb} KB"),
        }
    }
}

/// Configuración completa de la aplicación.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppConfig {
    /// Ruta a `realesrgan-ncnn-vulkan.exe`.
    pub realesrgan_exe: Option<PathBuf>,
    /// Ruta a `ffmpeg.exe`.
    pub ffmpeg_exe: Option<PathBuf>,
    /// Ruta a `GFPGANv1.4.pth`.
    pub gfpgan_model: Option<PathBuf>,
    /// Ruta a `RestoreFormer++.ckpt`.
    pub restoreformer_model: Option<PathBuf>,
    /// Ruta al intérprete de Python usado como puente para los modelos de rostros.
    pub python_exe: Option<PathBuf>,

    /// Escala del motor (2 o 4). El motor solo sabe x2 y x4.
    pub engine_scale: u8,
    /// Modelo de Real-ESRGAN (`realesrgan-x4plus`, ...).
    pub model: String,
    /// Escala final deseada respecto al original (1.0 = mismo tamaño, 4.0 = x4).
    pub target_scale: f32,
    /// Porcentaje de mejora aplicado sobre la imagen original (0..=100).
    pub improve_percent: u8,
    /// Peso de la imagen final.
    pub weight: Weight,
    /// Formato de salida.
    pub output_format: OutputFormat,
    /// Cuándo reescalar.
    pub resize_mode: ResizeMode,
    /// Filtro de interpolación.
    pub resize_filter: ResizeFilter,
    /// Nitidez extra tras el reescalado (0..=100).
    pub sharpen: u8,
    /// Eliminar temporales tras procesar video.
    pub delete_temp: bool,
    /// Tema de la interfaz.
    pub theme: String,
    /// Trabajos simultáneos en modo lote.
    pub max_workers: usize,
    /// Dispositivo preferido para el motor de IA (auto / nvidia / amd / intel / cpu).
    pub device: DevicePreference,
    /// Índice de GPU forzado para el motor (`-g`). `None` = automática.
    pub gpu_index: Option<usize>,
    /// Carpeta por defecto de salida.
    pub default_output_dir: Option<PathBuf>,
    /// Motor de rostros: `auto` (ONNX si está disponible, si no Python), `onnx` o `python`.
    #[serde(default = "default_face_backend")]
    pub face_backend: String,
    /// Idioma de la interfaz (`auto` = el del sistema).
    #[serde(default = "default_language")]
    pub language: String,
    /// Fondos de los temas con colores en movimiento.
    #[serde(default = "default_true")]
    pub animated_theme: bool,
    /// Colores del tema personalizado (JSON: base + colores claro/oscuro). Vacío si no se usa.
    #[serde(default)]
    pub custom_theme: String,
}

fn default_language() -> String {
    "auto".to_string()
}

fn default_true() -> bool {
    true
}

fn default_face_backend() -> String {
    "auto".to_string()
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            realesrgan_exe: None,
            ffmpeg_exe: None,
            gfpgan_model: None,
            restoreformer_model: None,
            python_exe: None,
            engine_scale: 4,
            model: "realesrgan-x4plus".to_string(),
            target_scale: 2.0,
            improve_percent: 100,
            weight: Weight::default(),
            output_format: OutputFormat::Png,
            resize_mode: ResizeMode::Always,
            resize_filter: ResizeFilter::Lanczos3,
            sharpen: 0,
            delete_temp: true,
            theme: "Dark".to_string(),
            max_workers: 2,
            device: DevicePreference::Auto,
            gpu_index: None,
            default_output_dir: None,
            face_backend: default_face_backend(),
            language: default_language(),
            animated_theme: true,
            custom_theme: String::new(),
        }
    }
}

impl AppConfig {
    /// Carpeta donde vive `config.json`.
    pub fn config_dir() -> PathBuf {
        if let Some(dir) = dirs::config_dir() {
            dir.join("Reescaller")
        } else {
            PathBuf::from(".")
        }
    }

    /// Ruta del archivo de configuración.
    pub fn config_path() -> PathBuf {
        Self::config_dir().join("config.json")
    }

    /// Carga la configuración. Si no existe, devuelve la de por defecto.
    pub fn load() -> Self {
        let path = Self::config_path();
        match std::fs::read_to_string(&path) {
            Ok(data) => match serde_json::from_str::<AppConfig>(&data) {
                Ok(mut cfg) => {
                    cfg.validar();
                    cfg
                }
                Err(e) => {
                    log::warn!("config.json inválido ({e}); se usan los valores por defecto");
                    Self::default()
                }
            },
            Err(_) => Self::default(),
        }
    }

    /// Guarda la configuración en disco.
    pub fn save(&self) -> Result<()> {
        let path = Self::config_path();
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let data = serde_json::to_string_pretty(self)?;
        std::fs::write(&path, data)?;
        log::info!("Configuración guardada en {}", path.display());
        Ok(())
    }

    /// Ajusta valores fuera de rango.
    pub fn validar(&mut self) {
        self.engine_scale = if self.engine_scale == 2 { 2 } else { 4 };
        self.target_scale = self.target_scale.clamp(0.25, 8.0);
        self.improve_percent = self.improve_percent.min(100);
        self.sharpen = self.sharpen.min(100);
        self.max_workers = self.max_workers.clamp(1, 16);
        if !matches!(
            self.face_backend.to_lowercase().as_str(),
            "auto" | "onnx" | "python"
        ) {
            self.face_backend = default_face_backend();
        }
        if let Some(idx) = self.gpu_index {
            if idx > 15 {
                self.gpu_index = None;
            }
        }
        if let Weight::Quality { quality } = self.weight {
            self.weight = Weight::Quality {
                quality: quality.clamp(1, 100),
            };
        }
    }

    /// Busca ejecutables y modelos en `base_dir` y `base_dir/models`.
    ///
    /// Devuelve `true` si se detectó al menos el motor de Real-ESRGAN.
    pub fn autodetectar(&mut self, base_dir: &Path) -> bool {
        let models_dir = base_dir.join("models");

        fn buscar(bases: &[&Path], nombres: &[&str]) -> Option<PathBuf> {
            for base in bases {
                for nombre in nombres {
                    let cand = base.join(nombre);
                    if cand.is_file() {
                        return Some(cand);
                    }
                }
            }
            None
        }

        // Orden de búsqueda: junto a la app, carpetas superiores (p. ej. el proyecto
        // que contiene `dist\`) y la carpeta de modelos del usuario.
        let mut propias: Vec<PathBuf> = vec![models_dir.clone(), base_dir.to_path_buf()];
        let mut ancestro = base_dir.parent();
        for _ in 0..3 {
            if let Some(a) = ancestro {
                propias.push(a.join("models"));
                propias.push(a.to_path_buf());
                ancestro = a.parent();
            }
        }
        if let Some(d) = dirs::data_local_dir() {
            propias.push(d.join("Reescaller").join("models"));
        }
        let bases: Vec<&Path> = propias.iter().map(|p| p.as_path()).collect();

        // El motor y FFmpeg que van junto a la app tienen prioridad sobre una ruta guardada de otra instalación
        // (por ejemplo, una carpeta de pruebas anterior). Solo si no hay copia propia se respeta/busca el resto.
        let locales: &[&Path] = &bases[..2.min(bases.len())];
        let motor = ["realesrgan-ncnn-vulkan.exe", "realesrgan-ncnn-vulkan"];
        let ffmpeg = ["ffmpeg.exe", "ffmpeg"];
        if let Some(local) = buscar(locales, &motor) {
            self.realesrgan_exe = Some(local);
        } else if self
            .realesrgan_exe
            .as_ref()
            .map(|p| !p.is_file())
            .unwrap_or(true)
        {
            self.realesrgan_exe = buscar(&bases, &motor);
        }
        if let Some(local) = buscar(locales, &ffmpeg) {
            self.ffmpeg_exe = Some(local);
        } else if self
            .ffmpeg_exe
            .as_ref()
            .map(|p| !p.is_file())
            .unwrap_or(true)
        {
            self.ffmpeg_exe = buscar(&bases, &ffmpeg);
        }
        if self
            .gfpgan_model
            .as_ref()
            .map(|p| !p.is_file())
            .unwrap_or(true)
        {
            self.gfpgan_model = buscar(&bases, &["GFPGANv1.4.pth"]);
        }
        if self
            .restoreformer_model
            .as_ref()
            .map(|p| !p.is_file())
            .unwrap_or(true)
        {
            self.restoreformer_model = buscar(&bases, &["RestoreFormer++.ckpt"]);
        }
        if self
            .python_exe
            .as_ref()
            .map(|p| !p.is_file())
            .unwrap_or(true)
        {
            self.python_exe = buscar(&bases, &["venv/Scripts/python.exe", "venv/bin/python"]);
        }

        self.realesrgan_exe.is_some()
    }

    /// Ejecutable de Real-ESRGAN o error.
    pub fn realesrgan(&self) -> Result<&Path> {
        self.realesrgan_exe
            .as_ref()
            .filter(|p| p.is_file())
            .map(|p| p.as_path())
            .ok_or_else(|| {
                Error::ExecutableNotFound(self.realesrgan_exe.clone().unwrap_or_default())
            })
    }

    /// Ejecutable de FFmpeg o error.
    pub fn ffmpeg(&self) -> Result<&Path> {
        self.ffmpeg_exe
            .as_ref()
            .filter(|p| p.is_file())
            .map(|p| p.as_path())
            .ok_or_else(|| Error::ExecutableNotFound(self.ffmpeg_exe.clone().unwrap_or_default()))
    }
}
