//! Servidor HTTP local: sirve la interfaz y la API JSON.

use crate::api;
use crate::state::{now_ms, State};
use serde_json::json;
use std::io::Cursor;
use std::sync::Arc;
use tiny_http::{Header, Method, Response, Server};

/// Interfaz embebida en el ejecutable (no hay archivos sueltos que perder).
const INDEX_HTML: &str = include_str!("../assets/index.html");

/// Atiende peticiones para siempre (bloquea el hilo actual).
pub fn serve(server: Server, state: Arc<State>) {
    for request in server.incoming_requests() {
        let state = Arc::clone(&state);
        std::thread::spawn(move || {
            handle(request, &state);
        });
    }
}

/// Cierra la app cuando la interfaz deja de latir (se cerró la ventana).
///
/// No cierra nunca mientras haya un trabajo pesado en curso, y solo actúa si
/// la interfaz llegó a conectarse al menos una vez.
pub fn spawn_watchdog(state: Arc<State>, timeout_ms: u64) {
    std::thread::spawn(move || loop {
        std::thread::sleep(std::time::Duration::from_secs(2));
        let ultimo = state.last_ping_ms.load(std::sync::atomic::Ordering::SeqCst);
        if ultimo == 0 || state.busy() {
            continue;
        }
        if now_ms().saturating_sub(ultimo) > timeout_ms {
            std::process::exit(0);
        }
    });
}

/// Valor de una cabecera (sin distinguir mayúsculas).
fn header_value(request: &tiny_http::Request, name: &str) -> Option<String> {
    request
        .headers()
        .iter()
        .find(|h| h.field.as_str().as_str().eq_ignore_ascii_case(name))
        .map(|h| h.value.as_str().to_string())
}

/// Protección frente a páginas web ajenas que intenten hablar con el servidor
/// local (CSRF y DNS rebinding): el `Host` debe ser local y toda llamada a la
/// API debe llevar una cabecera propia, que un sitio externo no puede enviar
/// sin permiso CORS (y no se concede ninguno).
fn peticion_permitida(request: &tiny_http::Request, token: &str) -> bool {
    let host_ok = match header_value(request, "Host") {
        Some(h) => {
            let h = h.to_lowercase();
            h.starts_with("127.0.0.1:")
                || h.starts_with("localhost:")
                || h == "127.0.0.1"
                || h == "localhost"
        }
        None => false,
    };
    if !host_ok {
        return false;
    }
    // Origin / Sec-Fetch-Site: una página ajena (otro origen) nunca debe poder llamar a la API.
    if let Some(origen) = header_value(request, "Origin") {
        let o = origen.to_lowercase();
        if !(o.starts_with("http://127.0.0.1:")
            || o.starts_with("http://localhost:")
            || o == "null")
        {
            return false;
        }
    }
    if let Some(sitio) = header_value(request, "Sec-Fetch-Site") {
        if !matches!(sitio.to_lowercase().as_str(), "same-origin" | "none") {
            return false;
        }
    }
    let url = request.url();
    if !url.starts_with("/api/") {
        return true;
    }
    let valor = header_value(request, "X-Reescaller");
    // «ping» y «foco» los usa la segunda ejecución de la app (no pueden conocer el token) y no hacen nada delicado.
    let libre = url.starts_with("/api/ping") || url.starts_with("/api/focus");
    match valor {
        Some(v) if libre => !v.is_empty(),
        Some(v) => v == token,
        None => false,
    }
}

/// Tamaño máximo de una petición (las subidas de imagen van en base64).
const MAX_CUERPO: usize = 160 * 1024 * 1024;

fn handle(mut request: tiny_http::Request, state: &State) {
    if !peticion_permitida(&request, &state.token) {
        let _ = request.respond(Response::from_string("Forbidden").with_status_code(403));
        return;
    }
    if request.body_length().unwrap_or(0) > MAX_CUERPO {
        let _ = request.respond(Response::from_string("Payload too large").with_status_code(413));
        return;
    }
    let url = request.url().to_string();
    let method = request.method().clone();
    let mut body = String::new();
    {
        use std::io::Read as _;
        let reader = request.as_reader();
        let mut buffer = String::new();
        if reader
            .take(MAX_CUERPO as u64 + 1)
            .read_to_string(&mut buffer)
            .is_ok()
            && buffer.len() <= MAX_CUERPO
        {
            body = buffer;
        }
    }

    let (path, qs) = match url.split_once('?') {
        Some((p, q)) => (p, q),
        None => (url.as_str(), ""),
    };
    let mut query: Vec<(String, String)> = Vec::new();
    for pair in qs.split('&').filter(|s| !s.is_empty()) {
        if let Some((k, v)) = pair.split_once('=') {
            query.push((k.to_string(), url_decode(v)));
        }
    }

    let response = route(state, &method, path, &query, &body);
    let _ = request.respond(response);
}

/// Decodifica el cuerpo JSON o devuelve la respuesta de error.
fn parse<T: serde::de::DeserializeOwned>(body: &str) -> Result<T, serde_json::Value> {
    serde_json::from_str::<T>(body)
        .map_err(|e| json!({"ok": false, "error": format!("JSON inválido: {e}")}))
}

fn route(
    state: &State,
    method: &Method,
    path: &str,
    query: &[(String, String)],
    body: &str,
) -> Response<Cursor<Vec<u8>>> {
    let get = |k: &str| -> Option<String> {
        query
            .iter()
            .find(|(key, _)| key == k)
            .map(|(_, v)| v.clone())
    };

    macro_rules! post_job {
        ($ty:ty, $handler:path) => {{
            match parse::<$ty>(body) {
                Ok(req) => {
                    let _job = state.begin_job();
                    json_response($handler(state, req))
                }
                Err(e) => json_response(e),
            }
        }};
    }

    match (method, path) {
        (&Method::Get, "/") => {
            // Desarrollo: REESCALLER_UI=<ruta a index.html> recarga la interfaz sin recompilar.
            match std::env::var_os("REESCALLER_UI").and_then(|p| std::fs::read_to_string(p).ok()) {
                Some(html) => html_response(&html, &state.token),
                None => html_response(INDEX_HTML, &state.token),
            }
        }

        // --- Ciclo de vida -------------------------------------------------
        (&Method::Get, "/api/ping") => {
            state.ping();
            json_response(json!({
                "ok": true,
                "app": "reescaller",
                "version": reescaller_core::VERSION,
                "busy": state.busy(),
            }))
        }
        (&Method::Get, "/api/focus") => {
            state
                .focus_request
                .store(true, std::sync::atomic::Ordering::SeqCst);
            json_response(json!({"ok": true}))
        }
        (&Method::Post, "/api/restart") => match crate::window::restart_self() {
            Ok(()) => {
                crate::window::log_to_file("Reinicio solicitado: nueva instancia lanzada");
                std::thread::spawn(|| {
                    std::thread::sleep(std::time::Duration::from_millis(400));
                    std::process::exit(0);
                });
                json_response(json!({"ok": true}))
            }
            Err(e) => {
                crate::window::log_to_file(&format!("Reinicio fallido: {e}"));
                json_response(json!({"ok": false, "error": format!("No se pudo reiniciar: {e}")}))
            }
        },
        (&Method::Get, p) if p.starts_with("/lang/") => lang_response(p),
        (&Method::Post, "/api/quit") => {
            std::thread::spawn(|| {
                std::thread::sleep(std::time::Duration::from_millis(250));
                std::process::exit(0);
            });
            json_response(json!({"ok": true}))
        }

        // --- Configuración y sistema --------------------------------------
        (&Method::Get, "/api/config") => json_response(api::handle_get_config(state)),
        (&Method::Post, "/api/config") => match parse::<reescaller_core::AppConfig>(body) {
            Ok(cfg) => json_response(api::handle_set_config(state, cfg)),
            Err(e) => json_response(e),
        },
        (&Method::Get, "/api/clipboard") => json_response(api::handle_clipboard_image()),
        (&Method::Get, "/api/compat") => json_response(api::handle_compat(state)),
        (&Method::Post, "/api/compat/test") => json_response(api::handle_compat_test(state)),
        (&Method::Post, "/api/rescan") => json_response(api::handle_rescan(state)),
        (&Method::Get, "/api/doctor") => json_response(api::handle_doctor(state)),
        (&Method::Get, "/api/hardware") => json_response(api::handle_hardware(
            state,
            get("refresh").as_deref() == Some("1"),
        )),
        (&Method::Get, "/api/logs") => {
            let since = get("since")
                .and_then(|v| v.parse::<u64>().ok())
                .unwrap_or(0);
            let (last, lines) = state.logs_since(since);
            json_response(json!({"ok": true, "result": {"last": last, "lines": lines}}))
        }

        // --- Archivos --------------------------------------------------------
        (&Method::Get, "/api/browse") => {
            let path = get("path").unwrap_or_default();
            json_response(api::handle_browse(state, &path))
        }
        (&Method::Get, "/api/pick") => {
            let kind = get("kind").unwrap_or_else(|| "image".into());
            let multi = get("multi").as_deref() == Some("1");
            json_response(api::handle_pick(&kind, multi))
        }
        (&Method::Get, "/api/reveal") => {
            let p = get("path").unwrap_or_default();
            let open = get("open").as_deref() == Some("1");
            json_response(api::handle_reveal(&p, open))
        }
        (&Method::Get, "/api/preview") => {
            let path = get("path").unwrap_or_default();
            let max = get("max")
                .and_then(|v| v.parse::<u32>().ok())
                .unwrap_or(720);
            json_response(api::handle_preview(&path, max))
        }
        (&Method::Post, "/api/upload") => match parse::<api::UploadRequest>(body) {
            Ok(req) => json_response(api::handle_upload(state, req)),
            Err(e) => json_response(e),
        },

        // --- Trabajos ------------------------------------------------------------
        (&Method::Post, "/api/remaster") => {
            // El análisis previo (dry_run) no es un trabajo pesado.
            post_job!(api::RemasterRequest, api::handle_remaster)
        }
        (&Method::Post, "/api/upscale") => post_job!(api::UpscaleRequest, api::handle_upscale),
        (&Method::Post, "/api/batch") => post_job!(api::BatchRequest, api::handle_batch),
        (&Method::Post, "/api/video") => post_job!(api::VideoRequest, api::handle_video),
        (&Method::Post, "/api/face") => post_job!(api::FaceRequest, api::handle_face),
        (&Method::Post, "/api/convert") => post_job!(api::ConvertRequest, api::handle_convert),
        (&Method::Post, "/api/trim") => post_job!(api::TrimRequest, api::handle_trim),

        _ => json_response(json!({"ok": false, "error": format!("Ruta no encontrada: {path}")})),
    }
}

fn json_response(value: serde_json::Value) -> Response<Cursor<Vec<u8>>> {
    let body = serde_json::to_vec(&value).unwrap_or_else(|_| b"{\"ok\":false}".to_vec());
    let header = Header::from_bytes(
        &b"Content-Type"[..],
        &b"application/json; charset=utf-8"[..],
    )
    .unwrap();
    Response::from_data(body)
        .with_header(header)
        .with_header(close_header())
}

/// Se cierra cada conexión para que el navegador no deje bloqueadas las siguientes.
fn close_header() -> Header {
    Header::from_bytes(&b"Connection"[..], &b"close"[..]).unwrap()
}

fn html_response(body: &str, token: &str) -> Response<Cursor<Vec<u8>>> {
    let body = body.replace("__REESCALLER_TOKEN__", token);
    let csp = Header::from_bytes(
        &b"Content-Security-Policy"[..],
        &b"default-src 'self'; img-src 'self' data: blob:; style-src 'self' 'unsafe-inline'; script-src 'self' 'unsafe-inline'; connect-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'"[..],
    )
    .unwrap();
    let nosniff = Header::from_bytes(&b"X-Content-Type-Options"[..], &b"nosniff"[..]).unwrap();
    let header =
        Header::from_bytes(&b"Content-Type"[..], &b"text/html; charset=utf-8"[..]).unwrap();
    let no_cache = Header::from_bytes(&b"Cache-Control"[..], &b"no-store"[..]).unwrap();
    Response::from_data(body.as_bytes().to_vec())
        .with_header(header)
        .with_header(no_cache)
        .with_header(csp)
        .with_header(nosniff)
        .with_header(close_header())
}

fn url_decode(input: &str) -> String {
    let bytes = input.replace('+', " ");
    let mut out = Vec::with_capacity(bytes.len());
    let raw = bytes.as_bytes();
    let mut i = 0;
    while i < raw.len() {
        if raw[i] == b'%' && i + 2 < raw.len() {
            let hex = std::str::from_utf8(&raw[i + 1..i + 3]).unwrap_or("");
            if let Ok(byte) = u8::from_str_radix(hex, 16) {
                out.push(byte);
                i += 3;
                continue;
            }
        }
        out.push(raw[i]);
        i += 1;
    }
    String::from_utf8_lossy(&out).to_string()
}

/// Diccionarios de la interfaz (el español es el idioma base y no necesita archivo).
fn lang_response(path: &str) -> Response<Cursor<Vec<u8>>> {
    let code = path.trim_start_matches("/lang/").trim_end_matches(".json");
    // Desarrollo: con REESCALLER_UI los diccionarios se leen de `lang/` junto al HTML, sin recompilar.
    let dev = std::env::var_os("REESCALLER_UI")
        .and_then(|p| {
            std::path::PathBuf::from(p)
                .parent()
                .map(|d| d.join("lang").join(format!("{code}.json")))
        })
        .and_then(|f| std::fs::read_to_string(f).ok());
    let raw: &str = match dev.as_deref() {
        Some(d) => d,
        None => match code {
            "en" => include_str!("../assets/lang/en.json"),
            "ru" => include_str!("../assets/lang/ru.json"),
            "zh" => include_str!("../assets/lang/zh.json"),
            "ja" => include_str!("../assets/lang/ja.json"),
            "pt" => include_str!("../assets/lang/pt.json"),
            "fr" => include_str!("../assets/lang/fr.json"),
            "de" => include_str!("../assets/lang/de.json"),
            "ko" => include_str!("../assets/lang/ko.json"),
            "it" => include_str!("../assets/lang/it.json"),
            _ => "{}",
        },
    };
    let header = Header::from_bytes(
        &b"Content-Type"[..],
        &b"application/json; charset=utf-8"[..],
    )
    .unwrap();
    let no_cache = Header::from_bytes(&b"Cache-Control"[..], &b"no-store"[..]).unwrap();
    Response::from_data(raw.as_bytes().to_vec())
        .with_header(header)
        .with_header(no_cache)
        .with_header(close_header())
}
