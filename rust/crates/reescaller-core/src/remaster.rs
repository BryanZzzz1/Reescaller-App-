//! Modo fácil de remasterización.
//!
//! El usuario solo elige una imagen (o carpeta), un *preset* y, si quiere, el
//! tamaño final. Este módulo analiza la imagen y construye el [`UpscalePlan`]
//! completo (modelo, escala, mezcla, nitidez, formato y peso) con una
//! explicación en lenguaje llano de lo que se va a hacer.

use crate::config::{DevicePreference, OutputFormat, ResizeFilter, ResizeMode, Weight};
use crate::error::Result;
use crate::image_ops;
use crate::upscale::UpscalePlan;
use image::GenericImageView;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::path::Path;

/// Perfil de remasterización.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum Preset {
    /// Analiza la imagen y elige todo automáticamente.
    #[default]
    Auto,
    /// Fotografías: detalle natural y algo de nitidez.
    Photo,
    /// Anime, dibujos e ilustraciones: líneas limpias, colores planos.
    Anime,
    /// Ligera para web y redes: tamaño contenido y archivo pequeño.
    Web,
}

/// Fuerza de la mejora.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum Intensity {
    Soft,
    #[default]
    Normal,
    Strong,
}

impl Intensity {
    pub fn improve_percent(&self) -> u8 {
        match self {
            Intensity::Soft => 55,
            Intensity::Normal => 85,
            Intensity::Strong => 100,
        }
    }
}

/// Qué tamaño final quiere el usuario.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Default)]
#[serde(tag = "mode", rename_all = "snake_case")]
pub enum SizeChoice {
    /// Se decide según el tamaño de la imagen.
    #[default]
    Auto,
    /// Mismo tamaño, solo mejora de calidad.
    Same,
    /// Multiplicar por un factor (1.5, 2, 3, 4...).
    Factor { factor: f32 },
}

/// Petición del modo fácil.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct RemasterOptions {
    pub preset: Preset,
    pub intensity: Intensity,
    pub size: SizeChoice,
    pub device: DevicePreference,
    pub gpu_index: Option<usize>,
}

impl Default for RemasterOptions {
    fn default() -> Self {
        Self {
            preset: Preset::Auto,
            intensity: Intensity::Normal,
            size: SizeChoice::Auto,
            device: DevicePreference::Auto,
            gpu_index: None,
        }
    }
}

/// Resultado del análisis de una imagen.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Analysis {
    pub width: u32,
    pub height: u32,
    pub megapixels: f32,
    /// Colores distintos por cada 1000 píxeles muestreados: bajo = ilustración.
    pub color_density: f32,
    /// `true` si parece un dibujo / anime.
    pub looks_illustration: bool,
    pub is_jpeg: bool,
}

/// Límite de seguridad del resultado (megapíxeles) para no agotar la memoria.
const MAX_OUTPUT_MP: f32 = 64.0;
/// Lado largo máximo del resultado.
const MAX_SIDE: u32 = 16_384;

/// Analiza una imagen para decidir los parámetros automáticos.
pub fn analyze(path: &Path) -> Result<Analysis> {
    let img = image_ops::load(path)?;
    let (w, h) = img.dimensions();
    let muestra = image_ops::thumbnail(&img, 128).to_rgb8();
    let mut colores: HashSet<u16> = HashSet::new();
    for p in muestra.pixels() {
        // 5 bits por canal.
        let c = ((p[0] as u16 >> 3) << 10) | ((p[1] as u16 >> 3) << 5) | (p[2] as u16 >> 3);
        colores.insert(c);
    }
    let total = (muestra.width() * muestra.height()).max(1) as f32;
    let densidad = colores.len() as f32 / total * 1000.0;
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_lowercase())
        .unwrap_or_default();
    Ok(Analysis {
        width: w,
        height: h,
        megapixels: (w as f32 * h as f32) / 1_000_000.0,
        color_density: densidad,
        // Las fotos reales tienen miles de colores en una muestra de 128 px;
        // los dibujos planos, unos pocos cientos.
        looks_illustration: colores.len() < 600,
        is_jpeg: ext == "jpg" || ext == "jpeg",
    })
}

/// Factor automático: lleva el lado largo hacia ~3000 px sin pasar de ×4.
fn auto_factor(a: &Analysis) -> f32 {
    let lado = a.width.max(a.height) as f32;
    if lado >= 3840.0 {
        return 1.0;
    }
    let ideal = (3000.0 / lado).clamp(1.0, 4.0);
    // Se redondea a factores "limpios".
    [1.0f32, 1.5, 2.0, 3.0, 4.0]
        .into_iter()
        .min_by(|x, y| {
            (x - ideal)
                .abs()
                .partial_cmp(&(y - ideal).abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .unwrap_or(2.0)
}

/// Limita el factor para que el resultado no sea gigantesco.
fn limitar(a: &Analysis, factor: f32) -> f32 {
    let mut f = factor.clamp(0.25, 8.0);
    let lado = a.width.max(a.height) as f32;
    if lado * f > MAX_SIDE as f32 {
        f = MAX_SIDE as f32 / lado;
    }
    let mp = a.megapixels * f * f;
    if mp > MAX_OUTPUT_MP {
        f *= (MAX_OUTPUT_MP / mp).sqrt();
    }
    f.max(0.25)
}

/// Plan resultante + explicación para mostrar al usuario.
#[derive(Debug, Clone)]
pub struct RemasterPlan {
    pub plan: UpscalePlan,
    pub explanation: Vec<String>,
}

/// Construye el plan completo del modo fácil.
pub fn build_plan(opts: &RemasterOptions, a: &Analysis) -> RemasterPlan {
    let mut why: Vec<String> = Vec::new();

    // --- Modelo según preset / análisis -------------------------------------
    let (model, tipo) = match opts.preset {
        Preset::Anime => ("realesrgan-x4plus-anime", "ilustración"),
        Preset::Photo => ("realesrgan-x4plus", "foto"),
        Preset::Auto | Preset::Web => {
            if a.looks_illustration {
                ("realesrgan-x4plus-anime", "ilustración")
            } else {
                ("realesrgan-x4plus", "foto")
            }
        }
    };
    if opts.preset == Preset::Auto || opts.preset == Preset::Web {
        why.push(format!(
            "Detectado: {tipo} de {}×{} ({:.1} MP)",
            a.width, a.height, a.megapixels
        ));
    }

    // --- Tamaño final ---------------------------------------------------------
    let mut resize_mode = ResizeMode::Always;
    let mut factor = match opts.size {
        SizeChoice::Same => {
            resize_mode = ResizeMode::ImproveOnly;
            1.0
        }
        SizeChoice::Factor { factor } => factor,
        SizeChoice::Auto => auto_factor(a),
    };
    if opts.preset == Preset::Web && opts.size == SizeChoice::Auto {
        // Para web: nunca más de 2048 px de lado largo.
        let lado = a.width.max(a.height) as f32;
        factor = factor.min(2048.0 / lado).max(1.0);
    }
    if factor <= 1.001 && resize_mode == ResizeMode::Always {
        resize_mode = ResizeMode::ImproveOnly;
        factor = 1.0;
    }
    let limitado = limitar(a, factor);
    if (limitado - factor).abs() > 0.001 {
        why.push("Tamaño reducido para no agotar la memoria del equipo.".to_string());
    }
    factor = limitado;

    let destino = image_ops::scaled_size((a.width, a.height), factor);
    if resize_mode == ResizeMode::ImproveOnly {
        why.push(format!(
            "Mismo tamaño ({}×{}), solo se mejora la calidad.",
            a.width, a.height
        ));
    } else {
        why.push(format!(
            "Se amplía ×{:.2} → {}×{}.",
            factor, destino.0, destino.1
        ));
    }

    // --- Mezcla y nitidez -----------------------------------------------------
    let improve = opts.intensity.improve_percent();
    let sharpen = match (opts.preset, tipo) {
        (Preset::Web, _) => 10,
        (_, "ilustración") => 0,
        _ => match opts.intensity {
            Intensity::Soft => 0,
            Intensity::Normal => 10,
            Intensity::Strong => 20,
        },
    };

    // --- Formato y peso --------------------------------------------------------
    let (output_format, weight) = if opts.preset == Preset::Web {
        (OutputFormat::Jpg, Weight::Quality { quality: 82 })
    } else if a.is_jpeg {
        (OutputFormat::Jpg, Weight::Quality { quality: 94 })
    } else {
        (OutputFormat::Png, Weight::Quality { quality: 92 })
    };
    why.push(format!(
        "Modelo «{model}», mejora {improve}%, salida {} ({}).",
        output_format.as_str().to_uppercase(),
        weight.describe()
    ));

    let plan = UpscalePlan {
        engine_scale: 4,
        model: model.to_string(),
        target_scale: factor,
        improve_percent: improve,
        weight,
        output_format,
        resize_mode,
        resize_filter: ResizeFilter::Lanczos3,
        sharpen,
        use_engine: true,
        device: opts.device,
        gpu_index: opts.gpu_index,
    };
    RemasterPlan {
        plan,
        explanation: why,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn analisis(w: u32, h: u32, ilus: bool, jpeg: bool) -> Analysis {
        Analysis {
            width: w,
            height: h,
            megapixels: (w as f32 * h as f32) / 1e6,
            color_density: 0.0,
            looks_illustration: ilus,
            is_jpeg: jpeg,
        }
    }

    #[test]
    fn auto_pequena_amplia_x4() {
        let r = build_plan(
            &RemasterOptions::default(),
            &analisis(640, 480, false, true),
        );
        assert_eq!(r.plan.target_scale, 4.0);
        assert_eq!(r.plan.model, "realesrgan-x4plus");
        assert_eq!(r.plan.output_format, OutputFormat::Jpg);
    }

    #[test]
    fn auto_grande_solo_mejora() {
        let r = build_plan(
            &RemasterOptions::default(),
            &analisis(4000, 3000, false, false),
        );
        assert_eq!(r.plan.resize_mode, ResizeMode::ImproveOnly);
        assert_eq!(r.plan.target_scale, 1.0);
    }

    #[test]
    fn auto_elige_modelo_anime_para_ilustraciones() {
        let r = build_plan(
            &RemasterOptions::default(),
            &analisis(800, 800, true, false),
        );
        assert_eq!(r.plan.model, "realesrgan-x4plus-anime");
        assert_eq!(r.plan.sharpen, 0);
    }

    #[test]
    fn nunca_supera_el_limite_de_memoria() {
        let opts = RemasterOptions {
            size: SizeChoice::Factor { factor: 8.0 },
            ..Default::default()
        };
        let r = build_plan(&opts, &analisis(6000, 4000, false, false));
        let (w, h) = image_ops::scaled_size((6000, 4000), r.plan.target_scale);
        assert!(w <= MAX_SIDE && h <= MAX_SIDE);
        assert!((w as f32 * h as f32) / 1e6 <= MAX_OUTPUT_MP + 0.5);
    }

    #[test]
    fn preset_web_limita_a_2048_y_jpg() {
        let opts = RemasterOptions {
            preset: Preset::Web,
            ..Default::default()
        };
        let r = build_plan(&opts, &analisis(1000, 800, false, false));
        assert!(r.plan.target_scale <= 2.05);
        assert_eq!(r.plan.output_format, OutputFormat::Jpg);
    }

    #[test]
    fn mismo_tamano_respeta_la_eleccion() {
        let opts = RemasterOptions {
            size: SizeChoice::Same,
            ..Default::default()
        };
        let r = build_plan(&opts, &analisis(500, 500, false, false));
        assert_eq!(r.plan.resize_mode, ResizeMode::ImproveOnly);
    }
}
