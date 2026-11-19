//! Reescaller · aplicación de escritorio.
//!
//! Un único ejecutable: arranca el servidor local (solo 127.0.0.1), abre su
//! propia ventana y se cierra solo cuando se cierra la ventana.
//!
//! Opciones: `--port N`, `--project <dir>`, `--browser` (usa el navegador
//! predeterminado en vez de la ventana de aplicación).

#![cfg_attr(windows, windows_subsystem = "windows")]

use reescaller_app::{server, state::State, window};
use std::path::PathBuf;
use std::sync::Arc;

/// Puerto fijo preferido: permite detectar una instancia ya abierta.
const PUERTO: u16 = 7879;

fn main() {
    std::panic::set_hook(Box::new(|info| {
        window::log_to_file(&format!("PANIC: {info}"));
    }));

    window::kill_children_on_exit();
    let mut port = PUERTO;
    let mut project: Option<PathBuf> = None;
    let mut browser = false;
    let mut restarted = false;
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--port" => {
                i += 1;
                port = args.get(i).and_then(|v| v.parse().ok()).unwrap_or(port);
            }
            "--project" => {
                i += 1;
                project = args.get(i).map(PathBuf::from);
            }
            "--browser" => browser = true,
            "--restarted" => restarted = true,
            _ => {}
        }
        i += 1;
    }

    // Tras «Reiniciar aplicación»: espera (hasta 20 s) a que la instancia anterior termine y libere el puerto.
    if restarted {
        let t0 = std::time::Instant::now();
        while window::is_reescaller_running(port)
            && t0.elapsed() < std::time::Duration::from_secs(20)
        {
            std::thread::sleep(std::time::Duration::from_millis(250));
        }
    }

    // Segunda ejecución: reutiliza la ventana ya abierta.
    if window::is_reescaller_running(port) {
        if window::request_focus(port) && !browser {
            return;
        }
        abrir(&format!("http://127.0.0.1:{port}/"), browser);
        return;
    }

    let mut enlace = tiny_http::Server::http(("127.0.0.1", port));
    if restarted {
        // Mismo puerto que antes (la interfaz guarda sus preferencias por origen), aunque tarde en liberarse.
        let t0 = std::time::Instant::now();
        while enlace.is_err() && t0.elapsed() < std::time::Duration::from_secs(20) {
            std::thread::sleep(std::time::Duration::from_millis(250));
            enlace = tiny_http::Server::http(("127.0.0.1", port));
        }
    }
    let server = match enlace.or_else(|_| tiny_http::Server::http(("127.0.0.1", 0))) {
        Ok(s) => s,
        Err(e) => {
            window::message_box(
                "Reescaller",
                &format!("No se pudo iniciar el servidor local.\n\n{e}"),
            );
            return;
        }
    };
    let port = server
        .server_addr()
        .to_ip()
        .map(|a| a.port())
        .unwrap_or(port);

    let project = project.unwrap_or_else(window::find_project_dir);
    let state = Arc::new(State::new(project));
    window::log_to_file(&format!("Inicio · puerto {port}"));

    // Precalienta la detección de hardware sin retrasar la ventana.
    std::thread::spawn(|| {
        let _ = reescaller_core::hardware::detect_cached();
    });

    let url = format!("http://127.0.0.1:{port}/");
    let datos = State::data_dir();

    // El servidor corre en segundo plano; la ventana ocupa el hilo principal.
    let estado_srv = Arc::clone(&state);
    let hilo = std::thread::spawn(move || server::serve(server, estado_srv));

    // 1) Ventana nativa sin marco (WebView2). Si el proceso termina al cerrarla.
    if !browser {
        // Se calcula antes y en una sentencia aparte: el bloqueo de la configuración no debe vivir durante toda la ventana.
        let (tema_guardado, tema_custom) = {
            let c = state.config.lock().unwrap();
            (c.theme.clone(), c.custom_theme.clone())
        };
        let colores_icono = Some(reescaller_app::icon::palette_colors_cfg(
            &tema_guardado,
            &tema_custom,
        ));
        match reescaller_app::native::run(
            &url,
            &datos.join("webview2"),
            Arc::clone(&state.focus_request),
            colores_icono,
        ) {
            Ok(()) => return,
            Err(e) => window::log_to_file(&format!("Ventana nativa no disponible: {e}")),
        }
    }

    // 2) Respaldo: ventana de aplicación de Edge/Chrome, o navegador normal.
    let perfil = datos.join("webview");
    let en_ventana = !browser && window::open_app_window(&url, &perfil);
    if !en_ventana {
        window::open_in_default_browser(&url);
    }
    // Solo con ventana propia se cierra sola; con el navegador normal la
    // pestaña puede suspenderse y el usuario sale con el botón «Salir».
    if en_ventana {
        server::spawn_watchdog(Arc::clone(&state), 25_000);
    }
    let _ = hilo.join();
}

fn abrir(url: &str, browser: bool) {
    let perfil = State::data_dir().join("webview");
    if browser || !window::open_app_window(url, &perfil) {
        window::open_in_default_browser(url);
    }
}
