//! Compatibilidad de los modelos de IA con el hardware del equipo.
//!
//! Antes de usar un modelo conviene saber si la GPU lo aguanta. Este módulo clasifica cada GPU (generación y
//! memoria de video) y dice, modelo por modelo, si es **compatible**, **limitado** (funciona, pero lento o con
//! poca memoria) o **no compatible**. La base mínima soportada es **NVIDIA GeForce serie 600 en adelante** y
//! **AMD Radeon serie 400/500 (Polaris) en adelante**; lo anterior no está garantizado.
//!
//! Las razones se devuelven como códigos con parámetros para que la interfaz las muestre traducidas.

use crate::config::DevicePreference;
use crate::hardware::{self, GpuInfo, GpuVendor, HardwareReport};
use serde::Serialize;

/// Veredicto para un modelo.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Level {
    /// Va bien.
    Ok,
    /// Funciona, pero con limitaciones.
    Limited,
    /// No se puede usar con este hardware.
    No,
}

/// ¿La GPU alcanza la base mínima soportada?
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Minimum {
    Yes,
    No,
    Unknown,
}

/// Motivo (código + parámetros) que la interfaz traduce.
#[derive(Debug, Clone, Serialize)]
pub struct Reason {
    pub code: &'static str,
    pub params: Vec<String>,
}

fn reason(code: &'static str, params: &[String]) -> Reason {
    Reason {
        code,
        params: params.to_vec(),
    }
}

/// Compatibilidad de un modelo.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelCompat {
    /// Identificador interno (`realesrgan-x4plus`, `face-gfpgan`…).
    pub id: &'static str,
    /// `engine` (Real-ESRGAN por Vulkan) o `face` (rostros por DirectML/CPU).
    pub group: &'static str,
    pub level: Level,
    pub reasons: Vec<Reason>,
}

/// Informe de una GPU (o del modo solo CPU).
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct GpuCompat {
    pub index: Option<usize>,
    pub name: String,
    /// Generación detectada (`Kepler`, `Polaris`, `RDNA 2`…) o vacío.
    pub generation: String,
    pub minimum: Minimum,
    pub vram_mb: Option<u32>,
    pub vram_is_minimum: bool,
    pub dedicated: bool,
    pub notes: Vec<Reason>,
    pub models: Vec<ModelCompat>,
}

/// Informe completo.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CompatReport {
    /// GPU que la app usará (o el modo solo CPU).
    pub used: GpuCompat,
    /// Resto de GPU detectadas.
    pub others: Vec<GpuCompat>,
    pub vulkan_available: bool,
    pub minimum_nvidia: &'static str,
    pub minimum_amd: &'static str,
    /// Peor nivel entre los modelos de motor (imágenes/video).
    pub engine_overall: Level,
}

/// Datos de un modelo: (id, grupo, VRAM mínima MB, VRAM recomendada MB).
const MODELS: [(&str, &str, u32, u32); 5] = [
    ("realesrgan-x4plus", "engine", 1024, 2048),
    ("realesrgan-x4plus-anime", "engine", 768, 1536),
    ("realesr-animevideov3", "engine", 512, 1024),
    ("face-gfpgan", "face", 1024, 2048),
    ("face-restoreformer", "face", 2048, 3072),
];

/// Primer número de 3 o 4 cifras que aparece tras `clave` en `texto` (mayúsculas).
fn numero_tras(texto: &str, clave: &str) -> Option<u32> {
    let mut resto = texto;
    while let Some(pos) = resto.find(clave) {
        let tras = &resto[pos + clave.len()..];
        let digitos: String = tras
            .trim_start_matches([' ', '-'])
            .chars()
            .take_while(|c| c.is_ascii_digit())
            .collect();
        if digitos.len() >= 3 {
            if let Ok(n) = digitos.parse() {
                return Some(n);
            }
        }
        resto = &resto[pos + clave.len()..];
    }
    None
}

/// Clasifica una GPU por su nombre: (generación, ¿alcanza el mínimo?).
pub fn classify(vendor: GpuVendor, name: &str) -> (String, Minimum) {
    let n = name.to_uppercase();
    match vendor {
        GpuVendor::Nvidia => {
            if n.contains("RTX") {
                let num = numero_tras(&n, "RTX");
                let gen = match num {
                    Some(v) if v >= 5000 => "Blackwell",
                    Some(v) if v >= 4000 => "Ada Lovelace",
                    Some(v) if v >= 3000 => "Ampere",
                    _ => "Turing",
                };
                return (gen.into(), Minimum::Yes);
            }
            if n.contains("TITAN") {
                return ("Titan".into(), Minimum::Yes);
            }
            if let Some(v) = numero_tras(&n, "GTX") {
                return match v {
                    1600..=1699 => ("Turing".into(), Minimum::Yes),
                    1000..=1199 => ("Pascal".into(), Minimum::Yes),
                    900..=999 => ("Maxwell".into(), Minimum::Yes),
                    600..=799 => ("Kepler".into(), Minimum::Yes),
                    _ => ("Fermi o anterior".into(), Minimum::No),
                };
            }
            if let Some(v) = numero_tras(&n, "GT ") {
                return match v {
                    1000..=1099 => ("Pascal".into(), Minimum::Yes),
                    // Las GT 6xx/7xx mezclan Fermi y Kepler: lo decide Vulkan.
                    600..=799 => ("Kepler o Fermi".into(), Minimum::Unknown),
                    _ => ("Fermi o anterior".into(), Minimum::No),
                };
            }
            if let Some(v) = numero_tras(&n, "MX") {
                let gen = if v >= 350 {
                    "Turing"
                } else if v >= 150 {
                    "Pascal"
                } else {
                    "Maxwell"
                };
                return (gen.into(), Minimum::Yes);
            }
            if n.contains("QUADRO K")
                || n.contains("QUADRO M")
                || n.contains("QUADRO P")
                || n.contains("QUADRO RTX")
            {
                return ("Quadro".into(), Minimum::Yes);
            }
            (String::new(), Minimum::Unknown)
        }
        GpuVendor::Amd => {
            if let Some(v) = numero_tras(&n, "RX") {
                let gen = match v {
                    7000..=9999 => "RDNA 3/4",
                    6000..=6999 => "RDNA 2",
                    5000..=5999 => "RDNA",
                    400..=599 => "Polaris",
                    _ => "",
                };
                return if gen.is_empty() {
                    (String::new(), Minimum::No)
                } else {
                    (gen.into(), Minimum::Yes)
                };
            }
            if n.contains("VEGA") || n.contains("RADEON VII") {
                return ("Vega".into(), Minimum::Yes);
            }
            if n.contains("RADEON PRO") || n.contains("FIREPRO") {
                return (String::new(), Minimum::Unknown);
            }
            if numero_tras(&n, "R9").is_some()
                || numero_tras(&n, "R7").is_some()
                || numero_tras(&n, "R5").is_some()
                || numero_tras(&n, "HD ").is_some()
            {
                return ("GCN anterior a Polaris".into(), Minimum::No);
            }
            (String::new(), Minimum::Unknown)
        }
        GpuVendor::Intel => {
            if n.contains("ARC") {
                ("Arc".into(), Minimum::Yes)
            } else if n.contains("IRIS XE") {
                ("Xe".into(), Minimum::Yes)
            } else {
                (String::new(), Minimum::Unknown)
            }
        }
        _ => (String::new(), Minimum::Unknown),
    }
}

fn peor(nuevo: Level, r: Reason, level: &mut Level, reasons: &mut Vec<Reason>) {
    if nuevo > *level {
        *level = nuevo;
    }
    reasons.push(r);
}

/// Informe de una GPU concreta.
fn gpu_report(g: &GpuInfo, informe: &HardwareReport, vulkan_gpu: bool) -> GpuCompat {
    let (generation, minimum) = classify(g.vendor, &g.name);
    let mut notes = Vec::new();
    let vram = g.vram_mb;
    let vram_txt = vram.map(|v| v.to_string()).unwrap_or_default();

    if minimum == Minimum::No {
        notes.push(reason("below_min", &[]));
    } else if matches!(
        generation.as_str(),
        "Kepler" | "Kepler o Fermi" | "Fermi o anterior"
    ) {
        notes.push(reason("old_gen", &[]));
    }
    if !g.dedicated {
        notes.push(reason("shared_mem", &[]));
    }

    let mut models = Vec::new();
    for (id, group, vram_min, vram_rec) in MODELS {
        let mut level = Level::Ok;
        let mut reasons: Vec<Reason> = Vec::new();
        if group == "engine" {
            if !informe.vulkan_available {
                peor(
                    Level::No,
                    reason("no_vulkan", &[]),
                    &mut level,
                    &mut reasons,
                );
            } else if !vulkan_gpu {
                peor(
                    Level::No,
                    reason("gpu_no_vulkan", &[]),
                    &mut level,
                    &mut reasons,
                );
            }
            if level != Level::No && minimum == Minimum::No {
                peor(
                    Level::Limited,
                    reason("below_min", &[]),
                    &mut level,
                    &mut reasons,
                );
            }
        } else {
            // Rostros: DirectML en cualquier GPU con DirectX 12; si no, CPU (más lento).
            if !(informe.vulkan_available || g.dedicated) || minimum == Minimum::No {
                peor(
                    Level::Limited,
                    reason("cpu_fallback", &[]),
                    &mut level,
                    &mut reasons,
                );
            }
        }
        if let Some(v) = vram {
            // Windows limita la VRAM informada a 4 GB: si es «4 GB o más», no se penaliza.
            if !g.vram_is_minimum || v < 4096 {
                if v < vram_min {
                    let nivel = if group == "engine" {
                        Level::No
                    } else {
                        Level::Limited
                    };
                    let code = if group == "engine" {
                        "vram_low"
                    } else {
                        "cpu_fallback"
                    };
                    peor(
                        nivel,
                        reason(code, std::slice::from_ref(&vram_txt)),
                        &mut level,
                        &mut reasons,
                    );
                } else if v < vram_rec {
                    peor(
                        Level::Limited,
                        reason("vram_tight", std::slice::from_ref(&vram_txt)),
                        &mut level,
                        &mut reasons,
                    );
                }
            }
        } else if !g.dedicated && level == Level::Ok {
            peor(
                Level::Limited,
                reason("shared_mem", &[]),
                &mut level,
                &mut reasons,
            );
        }
        models.push(ModelCompat {
            id,
            group,
            level,
            reasons,
        });
    }

    GpuCompat {
        index: Some(g.index),
        name: g.name.clone(),
        generation,
        minimum,
        vram_mb: g.vram_mb,
        vram_is_minimum: g.vram_is_minimum,
        dedicated: g.dedicated,
        notes,
        models,
    }
}

/// Informe del modo «solo CPU» / sin GPU utilizable.
fn cpu_report(informe: &HardwareReport) -> GpuCompat {
    let mut models = Vec::new();
    for (id, group, _, _) in MODELS {
        let (level, code) = if group == "engine" {
            (Level::No, "no_gpu")
        } else {
            (Level::Limited, "cpu_fallback")
        };
        models.push(ModelCompat {
            id,
            group,
            level,
            reasons: vec![reason(code, &[])],
        });
    }
    GpuCompat {
        index: None,
        name: informe.cpu.clone(),
        generation: String::new(),
        minimum: Minimum::Unknown,
        vram_mb: None,
        vram_is_minimum: false,
        dedicated: false,
        notes: Vec::new(),
        models,
    }
}

/// Calcula la compatibilidad según el hardware y la preferencia de dispositivo.
pub fn check(
    informe: &HardwareReport,
    preferencia: DevicePreference,
    gpu_manual: Option<usize>,
) -> CompatReport {
    let (elegida, _) = hardware::resolve_preference(preferencia, informe, gpu_manual);
    let vulkan_de = |g: &GpuInfo| informe.vulkan_devices.iter().any(|d| d.index == g.index);
    let todas: Vec<GpuCompat> = informe
        .gpus
        .iter()
        .map(|g| gpu_report(g, informe, vulkan_de(g)))
        .collect();

    let usada_pos = elegida.and_then(|i| informe.gpus.iter().position(|g| g.index == i));
    let (used, others) = match usada_pos {
        Some(p) => {
            let used = todas[p].clone();
            let others = todas
                .iter()
                .enumerate()
                .filter(|(i, _)| *i != p)
                .map(|(_, g)| g.clone())
                .collect();
            (used, others)
        }
        None => (cpu_report(informe), todas),
    };
    let engine_overall = used
        .models
        .iter()
        .filter(|m| m.group == "engine")
        .map(|m| m.level)
        .max()
        .unwrap_or(Level::No);
    CompatReport {
        used,
        others,
        vulkan_available: informe.vulkan_available,
        minimum_nvidia: "GeForce GTX 600",
        minimum_amd: "Radeon RX 400",
        engine_overall,
    }
}

/// Resultado de una prueba real de un modelo.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ProbeResult {
    pub id: &'static str,
    pub ok: bool,
    pub ms: u64,
    pub error: String,
}

/// Prueba de verdad cada modelo del motor: escala una imagen diminuta (48×48) y comprueba que sale el
/// tamaño esperado. `gpu_args` son los argumentos `-g N` ya resueltos. Cada prueba tiene un límite de 60 s.
pub fn probe_models(
    engine: &std::path::Path,
    gpu_args: &[String],
    work: &std::path::Path,
) -> Vec<ProbeResult> {
    use std::process::{Command, Stdio};
    let _ = std::fs::create_dir_all(work);
    let entrada = work.join("compat_in.png");
    let mut img = image::RgbImage::new(48, 48);
    for (x, y, p) in img.enumerate_pixels_mut() {
        *p = image::Rgb([(x * 5) as u8, (y * 5) as u8, 140]);
    }
    let _ = img.save(&entrada);

    let casos: [(&'static str, &str, u32); 3] = [
        ("realesrgan-x4plus", "realesrgan-x4plus", 4),
        ("realesrgan-x4plus-anime", "realesrgan-x4plus-anime", 4),
        ("realesr-animevideov3", "realesr-animevideov3", 2),
    ];
    let mut out = Vec::new();
    for (id, modelo, escala) in casos {
        let salida = work.join(format!("compat_{id}.png"));
        let _ = std::fs::remove_file(&salida);
        let t0 = std::time::Instant::now();
        let mut cmd = Command::new(engine);
        cmd.args([
            "-i",
            &entrada.display().to_string(),
            "-o",
            &salida.display().to_string(),
            "-s",
            &escala.to_string(),
            "-n",
            modelo,
        ])
        .args(gpu_args)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null());
        crate::process::sin_ventana(&mut cmd);
        let mut resultado = ProbeResult {
            id,
            ok: false,
            ms: 0,
            error: String::new(),
        };
        match cmd.spawn() {
            Err(e) => resultado.error = e.to_string(),
            Ok(mut hijo) => {
                let limite = std::time::Duration::from_secs(60);
                let estado = loop {
                    match hijo.try_wait() {
                        Ok(Some(s)) => break Some(s),
                        Ok(None) if t0.elapsed() > limite => {
                            let _ = hijo.kill();
                            let _ = hijo.wait();
                            break None;
                        }
                        Ok(None) => std::thread::sleep(std::time::Duration::from_millis(100)),
                        Err(_) => break None,
                    }
                };
                match estado {
                    None => resultado.error = "timeout".into(),
                    Some(s) if !s.success() => {
                        resultado.error = format!("exit {}", s.code().unwrap_or(-1))
                    }
                    Some(_) => match image::open(&salida) {
                        Ok(im) if im.width() == 48 * escala && im.height() == 48 * escala => {
                            resultado.ok = true
                        }
                        Ok(im) => {
                            resultado.error =
                                format!("tamaño inesperado {}x{}", im.width(), im.height())
                        }
                        Err(e) => resultado.error = format!("salida ilegible: {e}"),
                    },
                }
            }
        }
        resultado.ms = t0.elapsed().as_millis() as u64;
        let _ = std::fs::remove_file(&salida);
        out.push(resultado);
    }
    let _ = std::fs::remove_file(&entrada);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clasifica_nvidia() {
        assert_eq!(
            classify(GpuVendor::Nvidia, "NVIDIA GeForce GTX 650 Ti").1,
            Minimum::Yes
        );
        assert_eq!(
            classify(GpuVendor::Nvidia, "NVIDIA GeForce GTX 680").0,
            "Kepler"
        );
        assert_eq!(
            classify(GpuVendor::Nvidia, "NVIDIA GeForce GTX 1060 6GB").0,
            "Pascal"
        );
        assert_eq!(
            classify(GpuVendor::Nvidia, "NVIDIA GeForce GTX 1660 SUPER").0,
            "Turing"
        );
        assert_eq!(
            classify(GpuVendor::Nvidia, "NVIDIA GeForce RTX 4070").0,
            "Ada Lovelace"
        );
        assert_eq!(
            classify(GpuVendor::Nvidia, "NVIDIA GeForce GTX 480").1,
            Minimum::No
        );
        assert_eq!(
            classify(GpuVendor::Nvidia, "NVIDIA GeForce GTX 560 Ti").1,
            Minimum::No
        );
        assert_eq!(
            classify(GpuVendor::Nvidia, "NVIDIA GeForce GT 710").1,
            Minimum::Unknown
        );
        assert_eq!(
            classify(GpuVendor::Nvidia, "NVIDIA GeForce GT 1030").1,
            Minimum::Yes
        );
    }

    #[test]
    fn clasifica_amd() {
        assert_eq!(
            classify(GpuVendor::Amd, "AMD Radeon RX 580 Series").0,
            "Polaris"
        );
        assert_eq!(
            classify(GpuVendor::Amd, "AMD Radeon RX 470").1,
            Minimum::Yes
        );
        assert_eq!(classify(GpuVendor::Amd, "AMD Radeon RX 6600").0, "RDNA 2");
        assert_eq!(
            classify(GpuVendor::Amd, "AMD Radeon RX 7900 XTX").0,
            "RDNA 3/4"
        );
        assert_eq!(classify(GpuVendor::Amd, "AMD Radeon R9 290").1, Minimum::No);
        assert_eq!(
            classify(GpuVendor::Amd, "AMD Radeon HD 7970").1,
            Minimum::No
        );
        assert_eq!(
            classify(GpuVendor::Amd, "AMD Radeon RX Vega 56").1,
            Minimum::Yes
        );
    }
}
