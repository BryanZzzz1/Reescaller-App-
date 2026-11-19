//! Ventana nativa sin marco (tao + WebView2).
//!
//! La interfaz HTML se muestra en un WebView2 dentro de una ventana sin barra
//! de título del sistema: los botones de minimizar / maximizar / cerrar y la
//! zona de arrastre los dibuja la propia interfaz y llegan aquí por IPC.

use tao::dpi::LogicalSize;
use tao::event::{Event, WindowEvent};
use tao::event_loop::{ControlFlow, EventLoopBuilder};
use tao::platform::windows::WindowBuilderExtWindows;
use tao::window::{Icon, WindowBuilder};
use wry::{WebContext, WebViewBuilder};

/// Icono de la aplicación (PNG dentro del .ico, tras la cabecera de 22 bytes).
const ICO: &[u8] = include_bytes!("../assets/reescaller.ico");

#[derive(Debug)]
enum UserEvent {
    Ipc(String),
    Focus,
    Show,
}

fn icono() -> Option<Icon> {
    let png = ICO.get(22..)?;
    let img = image::load_from_memory(png).ok()?.to_rgba8();
    let (w, h) = img.dimensions();
    Icon::from_rgba(img.into_raw(), w, h).ok()
}

/// Abre la ventana y bloquea hasta que se cierra (termina el proceso).
/// Devuelve `Err` solo si no se pudo crear la ventana o el WebView2.
pub fn run(
    url: &str,
    data_dir: &std::path::Path,
    focus: std::sync::Arc<std::sync::atomic::AtomicBool>,
    initial_icon: Option<([u8; 3], [u8; 3])>,
) -> Result<(), String> {
    let event_loop = EventLoopBuilder::<UserEvent>::with_user_event().build();
    let proxy = event_loop.create_proxy();
    let proxy_focus = event_loop.create_proxy();
    std::thread::spawn(move || loop {
        std::thread::sleep(std::time::Duration::from_millis(300));
        if focus.swap(false, std::sync::atomic::Ordering::SeqCst) {
            let _ = proxy_focus.send_event(UserEvent::Focus);
        }
    });

    // Icono con los colores del tema guardado: se aplica al arrancar (ventana y botón de la barra de tareas).
    let icono_ventana = initial_icon
        .and_then(|(a, b)| crate::icon::tao_icon(a, b))
        .or_else(icono);
    // Respaldo: si la interfaz no avisa, la ventana se muestra igualmente a los 4 s.
    let proxy_show = event_loop.create_proxy();
    std::thread::spawn(move || {
        std::thread::sleep(std::time::Duration::from_secs(4));
        let _ = proxy_show.send_event(UserEvent::Show);
    });

    let window = WindowBuilder::new()
        .with_title("Reescaller")
        .with_decorations(false)
        .with_visible(false) // se muestra cuando la interfaz está lista (evita el parpadeo en blanco)
        .with_resizable(false)
        .with_inner_size(LogicalSize::new(840.0, 620.0))
        .with_window_icon(icono_ventana.clone())
        .with_taskbar_icon(icono_ventana)
        .build(&event_loop)
        .map_err(|e| format!("ventana: {e}"))?;

    let _ = std::fs::create_dir_all(data_dir);
    let mut contexto = WebContext::new(Some(data_dir.to_path_buf()));

    let webview = WebViewBuilder::with_web_context(&mut contexto)
        .with_url(url)
        .with_background_color((14, 10, 28, 255))
        .with_ipc_handler(move |req| {
            let _ = proxy.send_event(UserEvent::Ipc(req.body().to_string()));
        })
        .build(&window)
        .map_err(|e| format!("WebView2: {e}"))?;

    event_loop.run(move |event, _, control_flow| {
        *control_flow = ControlFlow::Wait;
        let _keep = &webview;
        match event {
            Event::UserEvent(UserEvent::Ipc(msg)) => match msg.as_str() {
                "drag" => {
                    let _ = window.drag_window();
                }
                "ready" => {
                    window.set_visible(true);
                    // El primer ShowWindow hereda el estado pedido por quien lanzó el proceso (a veces «minimizada»).
                    window.set_minimized(false);
                    window.set_focus();
                }
                "min" => window.set_minimized(true),
                "max" => {
                    if window.is_resizable() {
                        window.set_maximized(!window.is_maximized());
                    }
                }
                "close" => std::process::exit(0),
                other => {
                    // "size:ANCHO,ALTO,R" (píxeles lógicos; R=1 permite redimensionar)
                    if let Some(v) = other.strip_prefix("size:") {
                        let p: Vec<&str> = v.split(',').collect();
                        if let (Some(w), Some(h)) = (
                            p.first().and_then(|x| x.parse::<f64>().ok()),
                            p.get(1).and_then(|x| x.parse::<f64>().ok()),
                        ) {
                            let resizable = p.get(2) == Some(&"1");
                            window.set_maximized(false);
                            if resizable {
                                window.set_resizable(true);
                                window.set_min_inner_size(Some(LogicalSize::new(900.0, 600.0)));
                                window.set_inner_size(LogicalSize::new(w, h));
                            } else {
                                window.set_min_inner_size(None::<LogicalSize<f64>>);
                                window.set_inner_size(LogicalSize::new(w, h));
                                window.set_resizable(false);
                            }
                        }
                    }
                }
            },
            Event::UserEvent(UserEvent::Show) => {
                if !window.is_visible() {
                    window.set_visible(true);
                    window.set_minimized(false);
                    window.set_focus();
                }
            }
            Event::UserEvent(UserEvent::Focus) => {
                window.set_visible(true);
                window.set_minimized(false);
                window.set_focus();
            }
            // Al recuperar el foco, el teclado vuelve a la página (Ctrl+V, atajos…).
            Event::WindowEvent {
                event: WindowEvent::Focused(true),
                ..
            } => {
                let _ = webview.focus();
            }
            Event::WindowEvent {
                event: WindowEvent::CloseRequested,
                ..
            } => {
                std::process::exit(0);
            }
            _ => {}
        }
    });
}
