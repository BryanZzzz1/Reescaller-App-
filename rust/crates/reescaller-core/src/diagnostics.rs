//! Diagnóstico del entorno (equivalente a `verificar_dependencias_inicio()`).

use crate::config::{AppConfig, DevicePreference};
use crate::hardware::{self, EngineSupport, GpuVendor};
use crate::process;
use std::path::Path;

/// Estado de un componente.
#[derive(Debug, Clone, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Check {
    pub name: String,
    pub ok: bool,
    pub detail: String,
    pub required: bool,
}

/// Resultado del diagnóstico.
#[derive(Debug, Clone, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct DoctorReport {
    pub checks: Vec<Check>,
    pub all_required_ok: bool,
}

fn add(checks: &mut Vec<Check>, name: &str, ok: bool, detail: String, required: bool) {
    checks.push(Check {
        name: name.to_string(),
        ok,
        detail,
        required,
    });
}

/// Ejecuta todas las comprobaciones.
pub fn doctor(cfg: &AppConfig, base_dir: &Path) -> DoctorReport {
    let mut checks: Vec<Check> = Vec::new();

    match cfg.realesrgan_exe.as_ref() {
        Some(p) if p.is_file() => add(
            &mut checks,
            "Real-ESRGAN",
            true,
            p.display().to_string(),
            true,
        ),
        Some(p) => add(
            &mut checks,
            "Real-ESRGAN",
            false,
            format!("{} (no encontrado)", p.display()),
            true,
        ),
        None => add(
            &mut checks,
            "Real-ESRGAN",
            false,
            "no configurado".into(),
            true,
        ),
    }

    // El motor depende de VCOMP140.DLL (OpenMP de Visual C++). En un PC limpio
    // falta si no se instaló el "Visual C++ Redistributable": sin ella el motor
    // ni arranca y parecería que no hay GPU.
    if let Some(motor) = cfg.realesrgan_exe.as_ref().filter(|p| p.is_file()) {
        let junto_al_motor = motor
            .parent()
            .map(|d| d.join("vcomp140.dll"))
            .filter(|p| p.is_file());
        let en_sistema = std::env::var("SystemRoot")
            .map(|r| {
                std::path::PathBuf::from(r)
                    .join("System32")
                    .join("vcomp140.dll")
            })
            .ok()
            .filter(|p| p.is_file());
        match junto_al_motor.or(en_sistema) {
            Some(p) => add(&mut checks, "Runtime Visual C++ (vcomp140)", true, p.display().to_string(), true),
            None => add(
                &mut checks,
                "Runtime Visual C++ (vcomp140)",
                false,
                "falta vcomp140.dll: instala «Microsoft Visual C++ Redistributable 2015-2022 (x64)»                  o copia vcomp140.dll junto a realesrgan-ncnn-vulkan.exe"
                    .into(),
                true,
            ),
        }
    }

    match cfg.ffmpeg_exe.as_ref() {
        Some(p) if p.is_file() => add(&mut checks, "FFmpeg", true, p.display().to_string(), true),
        Some(p) => add(
            &mut checks,
            "FFmpeg",
            false,
            format!("{} (no encontrado)", p.display()),
            true,
        ),
        None => add(&mut checks, "FFmpeg", false, "no configurado".into(), true),
    }

    for (name, path, required) in [
        ("Modelo GFPGAN", &cfg.gfpgan_model, false),
        ("Modelo RestoreFormer++", &cfg.restoreformer_model, false),
    ] {
        match path.as_ref() {
            Some(p) if p.is_file() => {
                add(&mut checks, name, true, p.display().to_string(), required)
            }
            Some(p) => add(
                &mut checks,
                name,
                false,
                format!("{} (no encontrado)", p.display()),
                required,
            ),
            None => add(&mut checks, name, false, "no configurado".into(), required),
        }
    }

    match cfg.python_exe.as_ref() {
        Some(p) if p.is_file() => {
            // Se valida con el propio script del puente (incluye el parche de torchvision).
            // Junto al proyecto o, en la app instalada, la copia que escribe la propia app.
            let copia_usuario = dirs::data_local_dir()
                .map(|d| d.join("Reescaller").join("bridge").join("restore_face.py"));
            let propia = base_dir.join("scripts").join("restore_face.py");
            let script = if propia.is_file() {
                propia
            } else {
                copia_usuario.filter(|p| p.is_file()).unwrap_or(propia)
            };
            let args = vec![script.display().to_string(), "--selftest".to_string()];
            let (ok, detail) = match process::run(p, &args) {
                Ok(out) => (true, out.stdout.trim().to_string()),
                Err(e) => (false, e.to_string()),
            };
            add(
                &mut checks,
                "Puente Python (torch+gfpgan)",
                ok,
                if detail.is_empty() {
                    p.display().to_string()
                } else {
                    detail
                },
                false,
            );
        }
        _ => add(
            &mut checks,
            "Puente Python (torch+gfpgan)",
            false,
            "no configurado".into(),
            false,
        ),
    }

    // Motor de rostros nativo (ONNX Runtime): sin Python ni torch.
    match crate::face_onnx::find_onnx_dir(base_dir) {
        Some(dir) => {
            let modelos: Vec<&str> = crate::face_onnx::available_models(&dir)
                .iter()
                .map(|m| m.as_str())
                .collect();
            add(
                &mut checks,
                "Restauración de rostros",
                true,
                format!(
                    "motor integrado · {} · modelos: {}",
                    dir.display(),
                    modelos.join(", ")
                ),
                false,
            );
        }
        None => add(
            &mut checks,
            "Restauración de rostros",
            false,
            "no instalada (faltan los modelos en la carpeta models\\onnx)".into(),
            false,
        ),
    }

    // Modelos de Real-ESRGAN disponibles junto al ejecutable.
    if let Some(exe) = cfg.realesrgan_exe.as_ref() {
        if let Some(dir) = exe.parent() {
            let models_dir = dir.join("models");
            let found: Vec<String> = std::fs::read_dir(&models_dir)
                .map(|entries| {
                    entries
                        .flatten()
                        .filter_map(|e| {
                            let p = e.path();
                            let ext = p.extension()?.to_str()?;
                            if ext == "param" {
                                Some(p.file_stem()?.to_string_lossy().to_string())
                            } else {
                                None
                            }
                        })
                        .collect()
                })
                .unwrap_or_default();
            add(
                &mut checks,
                "Modelos ncnn locales",
                !found.is_empty(),
                if found.is_empty() {
                    format!("ninguno en {}", models_dir.display())
                } else {
                    found.join(", ")
                },
                false,
            );
        }
    }

    // Comprobación de permisos de escritura en la carpeta del proyecto.
    let probe = base_dir.join(".reescaller_write_test.tmp");
    let writable = std::fs::write(&probe, b"ok").is_ok();
    let _ = std::fs::remove_file(&probe);
    add(
        &mut checks,
        "Escritura en la carpeta del proyecto",
        writable,
        base_dir.display().to_string(),
        false,
    );

    // --- Hardware: GPU, Vulkan y CPU ----------------------------------------
    let informe = hardware::detect_cached();

    add(
        &mut checks,
        "CPU",
        true,
        format!(
            "{} ({} núcleos lógicos{})",
            informe.cpu,
            informe.logical_cores,
            match informe.physical_cores {
                Some(c) => format!(", {c} físicos"),
                None => String::new(),
            }
        ),
        false,
    );

    add(
        &mut checks,
        "Memoria RAM",
        informe.ram_total_mb > 0,
        format!(
            "{:.1} GB totales, {:.1} GB libres",
            informe.ram_total_mb as f64 / 1024.0,
            informe.ram_available_mb as f64 / 1024.0
        ),
        false,
    );

    if informe.gpus.is_empty() {
        add(
            &mut checks,
            "GPU",
            false,
            "no se detectó ninguna GPU (se usará el modo CPU)".to_string(),
            false,
        );
    } else {
        for gpu in &informe.gpus {
            add(
                &mut checks,
                &format!("GPU {}", gpu.index),
                true,
                format!(
                    "{} · driver {}{}",
                    gpu.describe(),
                    if gpu.driver.is_empty() {
                        "desconocido"
                    } else {
                        &gpu.driver
                    },
                    if gpu.dedicated {
                        " · dedicada"
                    } else {
                        " · integrada"
                    }
                ),
                false,
            );
        }
    }

    add(
        &mut checks,
        "Vulkan (motor de IA)",
        informe.vulkan_available,
        informe.vulkan_detail.clone(),
        false,
    );

    // Dispositivo configurado y si realmente se puede usar.
    let (indice, avisos) = hardware::resolve_preference(cfg.device, informe, cfg.gpu_index);
    let gpu_elegida = indice.and_then(|i| informe.gpus.iter().find(|g| g.index == i));
    let detalle_dispositivo = match (cfg.device, gpu_elegida) {
        (DevicePreference::Cpu, _) => "solo CPU (mejora por IA desactivada)".to_string(),
        (DevicePreference::Auto, Some(g)) => {
            format!("automático → {} (GPU {})", g.describe(), g.index)
        }
        (DevicePreference::Auto, None) => "automático → sin GPU, modo CPU".to_string(),
        (pref, Some(g)) => format!("{} → {} (GPU {})", pref.label(), g.describe(), g.index),
        (pref, None) => format!("{} → no encontrada, modo CPU", pref.label()),
    };
    add(
        &mut checks,
        "Dispositivo de cálculo",
        true,
        if avisos.is_empty() {
            detalle_dispositivo
        } else {
            format!("{detalle_dispositivo} ({})", avisos.join("; "))
        },
        false,
    );

    let motor_disponible = informe.engine_support == EngineSupport::Full;
    add(
        &mut checks,
        "Motor de mejora por IA",
        motor_disponible,
        if motor_disponible {
            format!(
                "listo (soporte {})",
                informe
                    .gpus
                    .iter()
                    .map(|g| g.vendor.label())
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        } else {
            "sin Vulkan: el motor se desactiva automáticamente y se trabaja por CPU".to_string()
        },
        false,
    );

    // Aviso explícito si el usuario eligió un fabricante que no está presente.
    if cfg.device != DevicePreference::Auto && cfg.device != DevicePreference::Cpu {
        let objetivo = match cfg.device {
            DevicePreference::Nvidia => GpuVendor::Nvidia,
            DevicePreference::Amd => GpuVendor::Amd,
            DevicePreference::Intel => GpuVendor::Intel,
            _ => GpuVendor::Unknown,
        };
        let hay = informe.gpus.iter().any(|g| g.vendor == objetivo);
        add(
            &mut checks,
            &format!("GPU {}", objetivo.label()),
            hay,
            if hay {
                "detectada".to_string()
            } else {
                "no hay ninguna GPU de este fabricante en el equipo".to_string()
            },
            false,
        );
    }

    let all_required_ok = checks.iter().all(|c| !c.required || c.ok);

    DoctorReport {
        checks,
        all_required_ok,
    }
}
