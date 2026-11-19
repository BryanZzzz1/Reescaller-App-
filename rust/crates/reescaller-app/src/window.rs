//! Ventana de la aplicación y utilidades de arranque.
//!
//! La interfaz es HTML servido por el propio ejecutable. Para que se sienta
//! como una aplicación y no como una pestaña, se abre en el modo `--app` de
//! Microsoft Edge (incluido en Windows 10/11) o de Google Chrome: ventana
//! propia, sin barra de direcciones ni pestañas, con su perfil aislado.

use std::io::{Read, Write};
use std::net::{SocketAddr, TcpStream};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Duration;

/// Busca la carpeta que contiene `models/` (o el motor directamente): junto al
/// ejecutable, en alguna carpeta superior (desarrollo) o en el directorio actual.
pub fn find_project_dir() -> PathBuf {
    let mut candidatos: Vec<PathBuf> = Vec::new();
    if let Ok(exe) = std::env::current_exe() {
        let mut dir = exe.parent().map(|p| p.to_path_buf());
        for _ in 0..6 {
            match dir {
                Some(d) => {
                    candidatos.push(d.clone());
                    dir = d.parent().map(|p| p.to_path_buf());
                }
                None => break,
            }
        }
    }
    if let Ok(cwd) = std::env::current_dir() {
        candidatos.push(cwd);
    }
    for c in &candidatos {
        let tiene = [
            "models/realesrgan-ncnn-vulkan.exe",
            "realesrgan-ncnn-vulkan.exe",
        ]
        .iter()
        .any(|rel| c.join(rel).is_file());
        if tiene {
            return c.clone();
        }
    }
    std::env::current_exe()
        .ok()
        .and_then(|e| e.parent().map(|p| p.to_path_buf()))
        .unwrap_or_else(|| PathBuf::from("."))
}

fn candidatos_navegador() -> Vec<PathBuf> {
    let var = |n: &str| std::env::var(n).ok().map(PathBuf::from);
    let mut v = Vec::new();
    for base in [
        var("ProgramFiles(x86)"),
        var("ProgramFiles"),
        var("LOCALAPPDATA"),
    ]
    .into_iter()
    .flatten()
    {
        v.push(base.join("Microsoft/Edge/Application/msedge.exe"));
    }
    for base in [
        var("ProgramFiles"),
        var("ProgramFiles(x86)"),
        var("LOCALAPPDATA"),
    ]
    .into_iter()
    .flatten()
    {
        v.push(base.join("Google/Chrome/Application/chrome.exe"));
    }
    v
}

/// Abre `url` como ventana de aplicación. Devuelve `false` si no hay Edge ni Chrome.
pub fn open_app_window(url: &str, profile_dir: &Path) -> bool {
    let Some(exe) = candidatos_navegador().into_iter().find(|p| p.is_file()) else {
        return false;
    };
    let _ = std::fs::create_dir_all(profile_dir);
    let mut cmd = Command::new(exe);
    cmd.arg(format!("--app={url}"))
        .arg("--window-size=840,660")
        .arg(format!("--user-data-dir={}", profile_dir.display()))
        .arg("--no-first-run")
        .arg("--no-default-browser-check")
        .arg("--disable-features=Translate,msEdgeSidebarV2")
        // Que el latido no se frene con la ventana minimizada.
        .arg("--disable-background-timer-throttling")
        .arg("--disable-renderer-backgrounding")
        .arg("--disable-backgrounding-occluded-windows");
    reescaller_core::process::sin_ventana(&mut cmd);
    cmd.spawn().is_ok()
}

/// Abre `url` en el navegador predeterminado.
pub fn open_in_default_browser(url: &str) {
    let mut cmd = Command::new("explorer.exe");
    cmd.arg(url);
    let _ = cmd.spawn();
}

/// Comprueba si ya hay una instancia de Reescaller escuchando en `port`.
pub fn is_reescaller_running(port: u16) -> bool {
    let addr: SocketAddr = ([127, 0, 0, 1], port).into();
    let Ok(mut s) = TcpStream::connect_timeout(&addr, Duration::from_millis(400)) else {
        return false;
    };
    let _ = s.set_read_timeout(Some(Duration::from_millis(800)));
    let peticion = format!(
        "GET /api/ping HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nX-Reescaller: 1\r\nConnection: close\r\n\r\n"
    );
    if s.write_all(peticion.as_bytes()).is_err() {
        return false;
    }
    let mut resp = String::new();
    let _ = s.read_to_string(&mut resp);
    resp.contains("\"app\":\"reescaller\"")
}

/// Pide a la instancia abierta que traiga su ventana al frente.
pub fn request_focus(port: u16) -> bool {
    let addr: SocketAddr = ([127, 0, 0, 1], port).into();
    let Ok(mut s) = TcpStream::connect_timeout(&addr, Duration::from_millis(400)) else {
        return false;
    };
    let _ = s.set_read_timeout(Some(Duration::from_millis(800)));
    let peticion = format!(
        "GET /api/focus HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nX-Reescaller: 1\r\nConnection: close\r\n\r\n"
    );
    if s.write_all(peticion.as_bytes()).is_err() {
        return false;
    }
    let mut resp = String::new();
    let _ = s.read_to_string(&mut resp);
    resp.contains("\"ok\":true")
}

/// Cuadro de mensaje nativo (la app no tiene consola donde informar de errores).
pub fn message_box(title: &str, text: &str) {
    #[cfg(windows)]
    {
        #[link(name = "user32")]
        extern "system" {
            fn MessageBoxW(hwnd: *mut u8, text: *const u16, caption: *const u16, kind: u32) -> i32;
        }
        let wide = |s: &str| -> Vec<u16> { s.encode_utf16().chain(std::iter::once(0)).collect() };
        let (t, c) = (wide(text), wide(title));
        // MB_OK | MB_ICONINFORMATION | MB_SETFOREGROUND
        unsafe {
            MessageBoxW(
                std::ptr::null_mut(),
                t.as_ptr(),
                c.as_ptr(),
                0x0000_0040 | 0x0001_0000,
            );
        }
    }
    #[cfg(not(windows))]
    {
        eprintln!("{title}: {text}");
    }
}

/// Escribe una línea en `%LOCALAPPDATA%\Reescaller\logs\app.log`.
pub fn log_to_file(line: &str) {
    let dir = crate::state::State::data_dir().join("logs");
    if std::fs::create_dir_all(&dir).is_ok() {
        if let Ok(mut f) = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(dir.join("app.log"))
        {
            let _ = writeln!(f, "[{}] {line}", crate::state::now_ms() / 1000);
        }
    }
}

/// Relanza la aplicación: abre otra instancia del mismo ejecutable (con los mismos argumentos más `--restarted`)
/// y devuelve `Ok` solo si se pudo lanzar. La nueva instancia espera a que esta libere el puerto.
pub fn restart_self() -> Result<(), String> {
    let exe = std::env::current_exe().map_err(|e| e.to_string())?;
    let mut args: Vec<String> = std::env::args()
        .skip(1)
        .filter(|a| a != "--restarted")
        .collect();
    args.push("--restarted".to_string());
    let lanzar = |flags: u32| -> std::io::Result<std::process::Child> {
        let mut cmd = Command::new(&exe);
        cmd.args(&args);
        if let Some(dir) = exe.parent() {
            cmd.current_dir(dir);
        }
        #[cfg(windows)]
        {
            use std::os::windows::process::CommandExt;
            cmd.creation_flags(flags);
        }
        #[cfg(not(windows))]
        let _ = flags;
        cmd.spawn()
    };
    // DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP, y si es posible fuera del «job» del proceso actual.
    const DETACHED: u32 = 0x0000_0008;
    const NEW_GROUP: u32 = 0x0000_0200;
    const BREAKAWAY: u32 = 0x0100_0000;
    lanzar(DETACHED | NEW_GROUP | BREAKAWAY)
        .or_else(|_| lanzar(DETACHED | NEW_GROUP))
        .map(|_| ())
        .map_err(|e| e.to_string())
}

/// Mete la aplicación en un «Job Object» de Windows con `KILL_ON_JOB_CLOSE`: al cerrarse (o terminar de golpe),
/// Windows termina también los procesos hijo (`ffmpeg`, `realesrgan`…) y no quedan huérfanos usando la GPU.
/// Se permite que un proceso se «escape» del job (`BREAKAWAY_OK`) para que el reinicio de la app siga funcionando.
#[cfg(windows)]
pub fn kill_children_on_exit() {
    use std::ffi::c_void;
    #[link(name = "kernel32")]
    extern "system" {
        fn CreateJobObjectW(attrs: *mut c_void, name: *const u16) -> isize;
        fn SetInformationJobObject(job: isize, class: i32, info: *mut c_void, len: u32) -> i32;
        fn AssignProcessToJobObject(job: isize, process: isize) -> i32;
        fn GetCurrentProcess() -> isize;
    }
    // JOBOBJECT_EXTENDED_LIMIT_INFORMATION (144 bytes en x64); LimitFlags está en el desplazamiento 16.
    let mut info = [0u8; 144];
    const KILL_ON_JOB_CLOSE: u32 = 0x2000;
    const BREAKAWAY_OK: u32 = 0x0800;
    info[16..20].copy_from_slice(&(KILL_ON_JOB_CLOSE | BREAKAWAY_OK).to_le_bytes());
    unsafe {
        let job = CreateJobObjectW(std::ptr::null_mut(), std::ptr::null());
        if job == 0 {
            return;
        }
        // 9 = JobObjectExtendedLimitInformation
        if SetInformationJobObject(job, 9, info.as_mut_ptr() as *mut c_void, info.len() as u32) == 0
        {
            return;
        }
        // El manejador se deja abierto a propósito: al terminar el proceso, Windows lo cierra y mata el job.
        let _ = AssignProcessToJobObject(job, GetCurrentProcess());
    }
}

#[cfg(not(windows))]
pub fn kill_children_on_exit() {}
