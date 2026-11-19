//! CLI de la Herramienta de Mejora Visual.
//!
//! Ejemplos:
//! ```text
//! reescaller doctor
//! reescaller upscale --input foto.jpg --target 1.5 --weight-quality 88
//! reescaller upscale --input foto.jpg --target-size-kb 400 --format jpg
//! reescaller batch --input carpeta --output salida --target 2 --workers 2
//! reescaller video --input clip.mp4 --output mejor.mp4
//! reescaller face --input cara.jpg --model gfpgan
//! ```

use anyhow::{Context, Result};
use clap::{Parser, Subcommand, ValueEnum};
use reescaller_core::{
    batch,
    config::{AppConfig, DevicePreference},
    diagnostics, face, hardware, upscale, video,
};
use std::path::PathBuf;

#[derive(Parser)]
#[command(
    name = "reescaller",
    version,
    about = "Herramienta de Mejora Visual (Rust)"
)]
struct Cli {
    /// Carpeta base del proyecto (donde están `models/`). Por defecto: la actual.
    #[arg(long, global = true)]
    project: Option<PathBuf>,

    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Comprueba el entorno (ejecutables, modelos, permisos).
    Doctor,

    /// Procesa una imagen.
    Upscale {
        #[arg(long)]
        input: PathBuf,
        /// Archivo de salida. Si se omite se genera junto al original.
        #[arg(long)]
        output: Option<PathBuf>,
        /// Escala del motor (2 o 4).
        #[arg(long, default_value_t = 4)]
        engine: u8,
        /// Modelo de Real-ESRGAN.
        #[arg(long, default_value = "realesrgan-x4plus")]
        model: String,
        /// Escala final respecto al original (1.0 = mismo tamaño, 2.0 = x2).
        #[arg(long, default_value_t = 2.0)]
        target: f32,
        /// Porcentaje de mejora (100 = solo la imagen mejorada).
        #[arg(long, default_value_t = 100)]
        improve: u8,
        /// Calidad de codificación (1-100).
        #[arg(long)]
        weight_quality: Option<u8>,
        /// Peso objetivo del archivo en KB (ajusta la calidad automáticamente).
        #[arg(long)]
        target_size_kb: Option<u32>,
        /// Formato de salida.
        #[arg(long, value_enum, default_value = "png")]
        format: FormatArg,
        /// Cuándo reescalar.
        #[arg(long, value_enum, default_value = "always")]
        mode: ModeArg,
        /// Nitidez extra (0-100).
        #[arg(long, default_value_t = 0)]
        sharpen: u8,
        /// No usar el motor de IA (solo reescalado rápido).
        #[arg(long)]
        no_engine: bool,
        /// Dispositivo del motor: auto, nvidia, amd, intel o cpu.
        #[arg(long, value_enum, default_value = "auto")]
        device: DeviceArg,
        /// Índice de GPU a forzar (`-g`).
        #[arg(long)]
        gpu: Option<usize>,
    },

    /// Procesa todas las imágenes de una carpeta.
    Batch {
        #[arg(long)]
        input: PathBuf,
        #[arg(long)]
        output: PathBuf,
        #[arg(long, default_value_t = 4)]
        engine: u8,
        #[arg(long, default_value = "realesrgan-x4plus")]
        model: String,
        #[arg(long, default_value_t = 2.0)]
        target: f32,
        #[arg(long, default_value_t = 100)]
        improve: u8,
        #[arg(long, default_value_t = 2)]
        workers: usize,
        #[arg(long, value_enum, default_value = "png")]
        format: FormatArg,
        #[arg(long)]
        weight_quality: Option<u8>,
        /// Dispositivo del motor: auto, nvidia, amd, intel o cpu.
        #[arg(long, value_enum, default_value = "auto")]
        device: DeviceArg,
        /// Índice de GPU a forzar (`-g`).
        #[arg(long)]
        gpu: Option<usize>,
    },

    /// Mejora un video completo.
    Video {
        #[arg(long)]
        input: PathBuf,
        #[arg(long)]
        output: PathBuf,
        #[arg(long, default_value = "realesrgan-x4plus")]
        model: String,
        #[arg(long, default_value_t = 4)]
        engine: u8,
        /// Escala final de los frames (1.0 = mismo tamaño que el original).
        #[arg(long, default_value_t = 1.0)]
        target: f32,
        /// Dispositivo del motor: auto, nvidia, amd, intel o cpu.
        #[arg(long, value_enum, default_value = "auto")]
        device: DeviceArg,
        /// Índice de GPU a forzar (`-g`).
        #[arg(long)]
        gpu: Option<usize>,
    },

    /// Restaura rostros con GFPGAN o RestoreFormer++.
    Face {
        #[arg(long)]
        input: PathBuf,
        #[arg(long)]
        output: Option<PathBuf>,
        #[arg(long, default_value = "gfpgan")]
        model: String,
        /// Motor: auto | onnx | python.
        #[arg(long)]
        backend: Option<String>,
    },

    /// Comprobación de hardware: CPU, RAM, GPU (NVIDIA/AMD/Intel) y Vulkan.
    Hardware {
        /// Salida en JSON.
        #[arg(long)]
        json: bool,
    },

    /// Muestra la configuración actual.
    Config,
}

#[derive(Copy, Clone, PartialEq, Eq, ValueEnum)]
enum FormatArg {
    Png,
    Jpg,
}

impl From<FormatArg> for reescaller_core::OutputFormat {
    fn from(value: FormatArg) -> Self {
        match value {
            FormatArg::Png => reescaller_core::OutputFormat::Png,
            FormatArg::Jpg => reescaller_core::OutputFormat::Jpg,
        }
    }
}

#[derive(Copy, Clone, PartialEq, Eq, ValueEnum)]
enum ModeArg {
    Always,
    OnlyIfNeeded,
    ImproveOnly,
}

#[derive(Copy, Clone, PartialEq, Eq, ValueEnum)]
enum DeviceArg {
    Auto,
    Nvidia,
    Amd,
    Intel,
    Cpu,
}

impl From<DeviceArg> for DevicePreference {
    fn from(value: DeviceArg) -> Self {
        match value {
            DeviceArg::Auto => DevicePreference::Auto,
            DeviceArg::Nvidia => DevicePreference::Nvidia,
            DeviceArg::Amd => DevicePreference::Amd,
            DeviceArg::Intel => DevicePreference::Intel,
            DeviceArg::Cpu => DevicePreference::Cpu,
        }
    }
}

impl From<ModeArg> for reescaller_core::ResizeMode {
    fn from(value: ModeArg) -> Self {
        match value {
            ModeArg::Always => reescaller_core::ResizeMode::Always,
            ModeArg::OnlyIfNeeded => reescaller_core::ResizeMode::OnlyIfNeeded,
            ModeArg::ImproveOnly => reescaller_core::ResizeMode::ImproveOnly,
        }
    }
}

fn main() -> Result<()> {
    env_logger::init();
    let cli = Cli::parse();

    let project = cli
        .project
        .clone()
        .unwrap_or_else(|| std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")));

    let mut cfg = AppConfig::load();
    cfg.autodetectar(&project);

    match cli.command {
        Command::Doctor => {
            let report = diagnostics::doctor(&cfg, &project);
            for check in &report.checks {
                println!(
                    "{} {} — {}",
                    if check.ok { "[OK]  " } else { "[FALLA]" },
                    check.name,
                    check.detail
                );
            }
            println!("\nConfiguración: {}", AppConfig::config_path().display());
            if !report.all_required_ok {
                std::process::exit(1);
            }
        }

        Command::Config => {
            println!("{}", serde_json::to_string_pretty(&cfg)?);
        }

        Command::Hardware { json } => {
            let informe = hardware::detect_cached();
            if json {
                println!("{}", serde_json::to_string_pretty(informe)?);
            } else {
                println!("Sistema      : {}", informe.os);
                println!("CPU          : {}", informe.cpu);
                println!(
                    "Núcleos      : {} lógicos{}",
                    informe.logical_cores,
                    match informe.physical_cores {
                        Some(c) => format!(", {c} físicos"),
                        None => String::new(),
                    }
                );
                println!(
                    "RAM          : {:.1} GB ({:.1} GB libres)",
                    informe.ram_total_mb as f64 / 1024.0,
                    informe.ram_available_mb as f64 / 1024.0
                );
                if informe.gpus.is_empty() {
                    println!("GPU          : ninguna detectada");
                } else {
                    for gpu in &informe.gpus {
                        println!(
                            "GPU {}        : {} · driver {} · origen {}",
                            gpu.index,
                            gpu.describe(),
                            if gpu.driver.is_empty() {
                                "?"
                            } else {
                                &gpu.driver
                            },
                            gpu.source
                        );
                    }
                }
                println!("Vulkan       : {}", informe.vulkan_detail);
                println!(
                    "Motor IA     : {}",
                    match informe.engine_support {
                        hardware::EngineSupport::Full => "disponible (GPU con Vulkan)",
                        hardware::EngineSupport::CpuOnly => "no disponible, se usará CPU",
                    }
                );
                println!("Lote (hilos) : {}", informe.recommended_workers);
                for nota in &informe.notes {
                    println!("Aviso        : {nota}");
                }
            }
        }

        Command::Upscale {
            input,
            output,
            engine,
            model,
            target,
            improve,
            weight_quality,
            target_size_kb,
            format,
            mode,
            sharpen,
            no_engine,
            device,
            gpu,
        } => {
            let format = format.into();
            let weight = match (target_size_kb, weight_quality) {
                (Some(kb), _) => reescaller_core::Weight::TargetSize { target_kb: kb },
                (None, Some(q)) => reescaller_core::Weight::Quality { quality: q },
                _ => reescaller_core::Weight::default(),
            };

            let plan = upscale::UpscalePlan {
                engine_scale: engine,
                model,
                target_scale: target,
                improve_percent: improve,
                weight,
                output_format: format,
                resize_mode: mode.into(),
                sharpen,
                use_engine: !no_engine,
                device: device.into(),
                gpu_index: gpu,
                ..Default::default()
            };

            let engine_exe = cfg
                .realesrgan_exe
                .clone()
                .context("Real-ESRGAN no está configurado")?;
            let output =
                output.unwrap_or_else(|| upscale::default_output_path(&input, format, "_mejorado"));

            let log = |msg: &str| println!("{msg}");
            let outcome = upscale::upscale_image(&plan, &engine_exe, &input, &output, &log)?;
            println!(
                "\nListo: {} ({}x{} -> {}x{}, {:.1} KB)",
                outcome.output.display(),
                outcome.original_size.0,
                outcome.original_size.1,
                outcome.final_size.0,
                outcome.final_size.1,
                outcome.bytes as f64 / 1024.0
            );
        }

        Command::Batch {
            input,
            output,
            engine,
            model,
            target,
            improve,
            workers,
            format,
            weight_quality,
            device,
            gpu,
        } => {
            let format = format.into();
            let weight = match weight_quality {
                Some(q) => reescaller_core::Weight::Quality { quality: q },
                None => reescaller_core::Weight::default(),
            };
            let plan = upscale::UpscalePlan {
                engine_scale: engine,
                model,
                target_scale: target,
                improve_percent: improve,
                weight,
                output_format: format,
                device: if device == DeviceArg::Auto {
                    cfg.device
                } else {
                    device.into()
                },
                gpu_index: gpu,
                ..Default::default()
            };
            let engine_exe = cfg
                .realesrgan_exe
                .clone()
                .context("Real-ESRGAN no está configurado")?;

            let log = |msg: &str| println!("{msg}");
            let outcome =
                batch::process_folder(&plan, &engine_exe, &input, &output, workers, &log)?;
            println!(
                "\n{} imagen(es) procesadas, {} error(es).",
                outcome.processed.len(),
                outcome.failed.len()
            );
        }

        Command::Video {
            input,
            output,
            model,
            engine,
            target,
            device,
            gpu,
        } => {
            let opts = video::VideoOptions {
                model,
                engine_scale: engine,
                target_scale: target,
                device: if device == DeviceArg::Auto {
                    cfg.device
                } else {
                    device.into()
                },
                gpu_index: gpu,
                ..Default::default()
            };
            let ffmpeg = cfg
                .ffmpeg_exe
                .clone()
                .context("FFmpeg no está configurado")?;
            let engine_exe = cfg
                .realesrgan_exe
                .clone()
                .context("Real-ESRGAN no está configurado")?;
            let log = |msg: &str| println!("{msg}");
            let outcome = video::process_video(&opts, &ffmpeg, &engine_exe, &input, &output, &log)?;
            println!(
                "\nListo: {} ({} frames, fps {:?})",
                outcome.output.display(),
                outcome.frames,
                outcome.fps
            );
        }

        Command::Face {
            input,
            output,
            model,
            backend,
        } => {
            let model = face::FaceModel::parse(&model)
                .with_context(|| format!("Modelo de rostros desconocido: {model}"))?;
            let mut cfg = cfg.clone();
            if let Some(b) = backend {
                cfg.face_backend = b;
                cfg.validar();
            }
            let output = output.unwrap_or_else(|| {
                upscale::default_output_path(&input, reescaller_core::OutputFormat::Png, "_rostros")
            });
            let script = project.join(face::bridge_script());
            let log = |msg: &str| println!("{msg}");
            let r = face::restore(&cfg, &project, &script, model, &input, &output, &log)?;
            println!("Motor: {} · {}", r.backend, r.detail);
            println!("\nListo: {}", output.display());
        }
    }

    Ok(())
}
