//! Ejecución de procesos externos (Real-ESRGAN, FFmpeg, Python).
//!
//! Equivalente Rust de `proceso_ejecutar()` + `detectar_fps()`.
//! Siempre con argumentos separados (nunca `shell`), con streaming de la salida
//! y detección del código de salida.

use crate::error::{Error, Result};
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

/// Evita que aparezca una ventana de consola al lanzar procesos hijo desde la
/// aplicación gráfica (subsistema Windows). No hace nada en otras plataformas.
pub fn sin_ventana(cmd: &mut Command) -> &mut Command {
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        cmd.creation_flags(0x0800_0000); // CREATE_NO_WINDOW
    }
    cmd
}

/// Últimas líneas útiles de la salida de error (FFmpeg imprime un banner enorme que no ayuda al usuario).
fn resumen_stderr(stderr: &str) -> String {
    let lineas: Vec<&str> = stderr
        .lines()
        .map(|l| l.trim())
        .filter(|l| !l.is_empty())
        .collect();
    let cola = &lineas[lineas.len().saturating_sub(5)..];
    let mut texto = cola
        .iter()
        .map(|l| {
            if l.chars().count() > 200 {
                format!("{}…", l.chars().take(200).collect::<String>())
            } else {
                l.to_string()
            }
        })
        .collect::<Vec<_>>()
        .join(" | ");
    if texto.is_empty() {
        texto = "sin detalle".into();
    }
    texto
}

/// Quita la marca «descargado de Internet» (`Zone.Identifier`) de los `.exe` y `.dll` de la app.
///
/// Con esa marca Windows muestra el aviso «Abrir archivo - Advertencia de seguridad» cada vez que la app lanza el
/// motor o FFmpeg. Es lo mismo que «Desbloquear» en las propiedades del archivo. No hace nada fuera de Windows.
pub fn unblock_downloaded(dir: &Path) {
    #[cfg(windows)]
    {
        fn recorre(dir: &Path, nivel: u32) {
            if nivel > 3 {
                return;
            }
            let Ok(entries) = std::fs::read_dir(dir) else {
                return;
            };
            for e in entries.flatten() {
                let p = e.path();
                if p.is_dir() {
                    let nombre = p
                        .file_name()
                        .map(|n| n.to_string_lossy().to_lowercase())
                        .unwrap_or_default();
                    if !matches!(
                        nombre.as_str(),
                        "uploads" | "webview" | "webview2" | "venv" | "target" | "target-dist"
                    ) {
                        recorre(&p, nivel + 1);
                    }
                } else if let Some(ext) = p.extension().and_then(|x| x.to_str()) {
                    if ext.eq_ignore_ascii_case("exe") || ext.eq_ignore_ascii_case("dll") {
                        let _ = std::fs::remove_file(format!("{}:Zone.Identifier", p.display()));
                    }
                }
            }
        }
        recorre(dir, 0);
    }
    #[cfg(not(windows))]
    let _ = dir;
}

/// Resultado de un proceso externo.
#[derive(Debug, Clone)]
pub struct CommandOutput {
    pub program: String,
    pub status: i32,
    pub stdout: String,
    pub stderr: String,
}

impl CommandOutput {
    pub fn success(&self) -> bool {
        self.status == 0
    }
}

/// Callback para recibir líneas enquanto el proceso corre.
pub type LogSink<'a> = &'a (dyn Fn(&str) + Send + Sync);

/// Ejecuta un comando y devuelve la salida.
///
/// `args` se pasa como argumentos separados: nunca se interpreta como comando
/// de shell (equivalente a `shell=False` en Python).
pub fn run(program: &Path, args: &[String]) -> Result<CommandOutput> {
    run_with_log(program, args, &|_| {})
}

/// Ejecuta un comando, reenviando stdout/stderr al callback en vivo.
pub fn run_with_log(program: &Path, args: &[String], log: LogSink<'_>) -> Result<CommandOutput> {
    if !program.is_file() {
        return Err(Error::ExecutableNotFound(program.to_path_buf()));
    }

    log(&format!("> {} {}", program.display(), args.join(" ")));

    let mut comando = Command::new(program);
    comando
        .args(args)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    sin_ventana(&mut comando);
    let mut child = comando.spawn().map_err(|e| {
        Error::Io(std::io::Error::new(
            e.kind(),
            format!("{}: {e}", program.display()),
        ))
    })?;

    let stdout_handle = child.stdout.take();
    let stderr_handle = child.stderr.take();

    // Hilos con alcance: permiten que el callback sea una referencia local.
    let (status, stdout, stderr) = std::thread::scope(|scope| {
        let stdout_thread = scope.spawn(move || {
            let mut buf = String::new();
            if let Some(out) = stdout_handle {
                for line in BufReader::new(out)
                    .lines()
                    .map_while(std::result::Result::ok)
                {
                    buf.push_str(&line);
                    buf.push('\n');
                    log(&line);
                }
            }
            buf
        });

        let stderr_thread = scope.spawn(move || {
            let mut buf = String::new();
            if let Some(err) = stderr_handle {
                for line in BufReader::new(err)
                    .lines()
                    .map_while(std::result::Result::ok)
                {
                    buf.push_str(&line);
                    buf.push('\n');
                    log(&line);
                }
            }
            buf
        });

        let status = match child.wait() {
            Ok(status) => status,
            Err(e) => {
                eprintln!("[proceso] error esperando al proceso: {e}");
                std::process::Command::new("cmd")
                    .args(["/C", "exit 1"])
                    .status()
                    .unwrap_or_else(|_| {
                        use std::os::windows::process::ExitStatusExt;
                        std::process::ExitStatus::from_raw(1)
                    })
            }
        };

        (
            status,
            stdout_thread.join().unwrap_or_default(),
            stderr_thread.join().unwrap_or_default(),
        )
    });

    if !status.success() {
        return Err(Error::CommandFailed {
            program: program.display().to_string(),
            code: status.code().unwrap_or(-1),
            stderr: resumen_stderr(&stderr),
        });
    }

    Ok(CommandOutput {
        program: program.display().to_string(),
        status: status.code().unwrap_or(0),
        stdout,
        stderr,
    })
}

/// Ejecuta un comando y descarta la salida (solo importa el código de salida).
pub fn run_quiet(program: &Path, args: &[String]) -> Result<()> {
    run(program, args).map(|_| ())
}

/// Detecta los FPS reales de un video parseando la salida de FFmpeg.
///
/// Devuelve `None` si no se puede determinar.
pub fn detect_fps(ffmpeg: &Path, video: &Path) -> Option<f64> {
    if !ffmpeg.is_file() || !video.is_file() {
        return None;
    }
    let args = vec![
        "-hide_banner".to_string(),
        "-i".to_string(),
        video.display().to_string(),
    ];
    let mut comando = Command::new(ffmpeg);
    comando.args(&args).stdin(Stdio::null());
    let out = sin_ventana(&mut comando).output().ok()?;
    let text = format!(
        "{}\n{}",
        String::from_utf8_lossy(&out.stderr),
        String::from_utf8_lossy(&out.stdout)
    );

    // "Video: h264 ..., 30000/1001 fps, 30 tbr"
    if let Some(idx) = text.find("Video:") {
        let slice = &text[idx..];
        if let Some(pos) = slice.find(" fps") {
            let window_start = slice[..pos].rfind(' ').map(|i| i + 1).unwrap_or(0);
            let candidate = &slice[window_start..pos];
            if let Ok(v) = candidate.trim().parse::<f64>() {
                if v > 0.0 {
                    return Some(v);
                }
            }
        }
    }
    // Fallback: "23.98 tbr"
    if let Some(idx) = text.find(" tbr") {
        let slice = &text[..idx];
        if let Some(pos) = slice.rfind(' ') {
            if let Ok(v) = slice[pos + 1..].trim().parse::<f64>() {
                if v > 0.0 {
                    return Some(v);
                }
            }
        }
    }
    None
}

/// Devuelve la ruta absoluta de un ejecutable en el PATH o `None`.
pub fn which(program: &str) -> Option<PathBuf> {
    let paths = std::env::var_os("PATH")?;
    let exts = if cfg!(windows) {
        vec![".exe", ".cmd", ".bat", ""]
    } else {
        vec![""]
    };
    for dir in std::env::split_paths(&paths) {
        for ext in &exts {
            let cand = dir.join(format!("{program}{ext}"));
            if cand.is_file() {
                return Some(cand);
            }
        }
    }
    None
}
