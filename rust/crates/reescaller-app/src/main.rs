//! Servidor de depuración de Reescaller (con consola).
//!
//! Igual que la aplicación de escritorio, pero muestra el log en la consola y
//! abre la interfaz en el navegador normal.
//!
//! Uso: `reescaller-debug [--port 7878] [--project <dir>] [--no-abrir]`

use anyhow::Result;
use reescaller_app::{server, state::State, window};
use std::path::PathBuf;
use std::sync::Arc;
use tiny_http::Server;

fn main() -> Result<()> {
    let mut port: u16 = 7878;
    let mut project: Option<PathBuf> = None;
    let mut abrir_navegador = true;

    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--port" | "-p" => {
                i += 1;
                port = args.get(i).and_then(|v| v.parse().ok()).unwrap_or(port);
            }
            "--project" | "-d" => {
                i += 1;
                project = args.get(i).map(PathBuf::from);
            }
            "--no-abrir" => abrir_navegador = false,
            "--help" | "-h" => {
                println!("reescaller-debug [--port 7878] [--project <dir>] [--no-abrir]");
                return Ok(());
            }
            other => anyhow::bail!("Argumento desconocido: {other}"),
        }
        i += 1;
    }

    let project = project.unwrap_or_else(window::find_project_dir);
    let state = Arc::new(State::new(project.clone()));

    let servidor = Server::http(("127.0.0.1", port))
        .map_err(|e| anyhow::anyhow!("No se pudo abrir el puerto {port}: {e}"))?;

    println!("──────────────────────────────────────────────");
    println!(" Reescaller · servidor de depuración");
    println!("──────────────────────────────────────────────");
    println!(" URL:      http://127.0.0.1:{port}");
    println!(" Proyecto: {}", project.display());
    println!(
        " Config:   {}",
        reescaller_core::AppConfig::config_path().display()
    );
    println!("──────────────────────────────────────────────");

    let informe = reescaller_core::hardware::detect_cached();
    println!(" Hardware: {}", informe.summary());
    if let Some(gpu) = informe.primary_gpu() {
        println!(" GPU:      {}", gpu.describe());
    }
    println!(
        " Motor IA: {}",
        match informe.engine_support {
            reescaller_core::EngineSupport::Full => "operativo (GPU con Vulkan)",
            reescaller_core::EngineSupport::CpuOnly => "no disponible, se usará CPU",
        }
    );
    for nota in &informe.notes {
        println!(" Aviso:    {nota}");
    }
    println!("──────────────────────────────────────────────");

    if abrir_navegador {
        let url = format!("http://127.0.0.1:{port}/");
        println!(" Abriendo el navegador: {url}");
        std::thread::spawn(move || {
            std::thread::sleep(std::time::Duration::from_millis(700));
            window::open_in_default_browser(&url);
        });
    }
    println!(" Para salir: Ctrl+C o cerrar esta ventana.");

    server::serve(servidor, state);
    Ok(())
}
