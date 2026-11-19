//! Instalador y desinstalador de Reescaller con ventana propia (estilo de la app).
//!
//! · El instalador es este mismo ejecutable con un ZIP añadido al final (el «payload»: la carpeta de la app, el desinstalador
//!   y el instalador de WebView2). Se lee a sí mismo; no hace falta descomprimir nada antes.
//! · Instala por usuario (sin administrador), crea accesos directos, registra la entrada de «Aplicaciones instaladas»
//!   y copia `Uninstall.exe` (este mismo programa sin payload) a la carpeta de la app.
//! · Modos silenciosos para scripts:  `/S`  `--dir=RUTA`  `--no-faces`  `--no-desktop`  y  `Uninstall.exe /S`.
#![cfg_attr(windows, windows_subsystem = "windows")]

use serde_json::{json, Value};
use std::fs::File;
use std::io::{BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Duration, Instant};
use tao::dpi::{LogicalSize, PhysicalPosition};
use tao::event::{Event, WindowEvent};
use tao::event_loop::{ControlFlow, EventLoopBuilder, EventLoopProxy};
use tao::window::{Icon, WindowBuilder};
use wry::{WebContext, WebViewBuilder};

const HTML: &str = include_str!("../assets/setup.html");
const ICO: &[u8] = include_bytes!("../../reescaller-app/assets/reescaller.ico");
const VERSION: &str = env!("CARGO_PKG_VERSION");
const WEBVIEW2_GUID: &str = "{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}";
const UNINSTALL_KEY: &str = r"HKCU\Software\Microsoft\Windows\CurrentVersion\Uninstall\Reescaller";

#[derive(Debug)]
enum Ev {
    Js(String),
    Show,
    Quit,
}

#[derive(Clone)]
struct Opts {
    dir: PathBuf,
    faces: bool,
    desktop: bool,
}

// ---------------------------------------------------------------------------------------------------------------- utilidades

fn sin_ventana(cmd: &mut Command) -> &mut Command {
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        cmd.creation_flags(0x0800_0000); // CREATE_NO_WINDOW
    }
    cmd
}

fn ejecutar(program: &str, args: &[&str]) -> Option<std::process::Output> {
    let mut c = Command::new(program);
    c.args(args);
    sin_ventana(&mut c);
    c.output().ok()
}

fn dir_por_defecto() -> PathBuf {
    let base = std::env::var_os("LOCALAPPDATA")
        .map(PathBuf::from)
        .unwrap_or_else(std::env::temp_dir);
    base.join("Programs").join("Reescaller")
}

fn escapar_ps(s: &str) -> String {
    s.replace('\'', "''")
}

fn carpeta_inicio() -> Option<PathBuf> {
    std::env::var_os("APPDATA").map(|a| {
        PathBuf::from(a)
            .join("Microsoft")
            .join("Windows")
            .join("Start Menu")
            .join("Programs")
    })
}

fn escritorio() -> Option<PathBuf> {
    std::env::var_os("USERPROFILE").map(|u| PathBuf::from(u).join("Desktop"))
}

fn webview2_instalado() -> bool {
    let claves = [
        format!(r"HKLM\SOFTWARE\WOW6432Node\Microsoft\EdgeUpdate\Clients\{WEBVIEW2_GUID}"),
        format!(r"HKLM\SOFTWARE\Microsoft\EdgeUpdate\Clients\{WEBVIEW2_GUID}"),
        format!(r"HKCU\Software\Microsoft\EdgeUpdate\Clients\{WEBVIEW2_GUID}"),
    ];
    for k in claves {
        if let Some(o) = ejecutar("reg.exe", &["query", &k, "/v", "pv"]) {
            let t = String::from_utf8_lossy(&o.stdout);
            if o.status.success() && t.contains("pv") && !t.contains("0.0.0.0") {
                return true;
            }
        }
    }
    false
}

fn vulkan_presente() -> bool {
    let root = std::env::var_os("SystemRoot")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(r"C:\Windows"));
    root.join("System32").join("vulkan-1.dll").is_file()
}

/// Abre el ZIP añadido al final de este ejecutable.
fn abrir_payload(exe: &Path) -> Result<zip::ZipArchive<BufReader<File>>, String> {
    let f = File::open(exe).map_err(|e| format!("No se pudo leer el instalador: {e}"))?;
    zip::ZipArchive::new(BufReader::new(f))
        .map_err(|e| format!("El instalador está dañado o incompleto ({e})."))
}

/// (bytes totales, bytes de los modelos de rostros) de los archivos de la app dentro del payload.
fn medir_payload(exe: &Path) -> (u64, u64) {
    let Ok(mut z) = abrir_payload(exe) else {
        return (0, 0);
    };
    let (mut total, mut rostros) = (0u64, 0u64);
    for i in 0..z.len() {
        if let Ok(e) = z.by_index_raw(i) {
            if e.is_dir() || e.name().starts_with("redist/") {
                continue;
            }
            total += e.size();
            if e.name().starts_with("models/onnx/") {
                rostros += e.size();
            }
        }
    }
    (total, rostros)
}

/// Si falta Microsoft Edge WebView2 Runtime, lo instala en silencio con el instalador oficial incluido en el payload.
fn asegurar_webview2(exe: &Path) {
    if webview2_instalado() {
        return;
    }
    let Ok(mut z) = abrir_payload(exe) else {
        return;
    };
    let Ok(mut e) = z.by_name("redist/MicrosoftEdgeWebview2Setup.exe") else {
        return;
    };
    let tmp = std::env::temp_dir().join("MicrosoftEdgeWebview2Setup.exe");
    if let Ok(mut out) = File::create(&tmp) {
        let _ = std::io::copy(&mut e, &mut out);
        drop(out);
        let mut c = Command::new(&tmp);
        c.args(["/silent", "/install"]);
        sin_ventana(&mut c);
        if let Ok(mut child) = c.spawn() {
            let t0 = Instant::now();
            while t0.elapsed() < Duration::from_secs(300) {
                if let Ok(Some(_)) = child.try_wait() {
                    break;
                }
                std::thread::sleep(Duration::from_millis(500));
            }
        }
        let _ = std::fs::remove_file(&tmp);
    }
}

fn crear_acceso(lnk: &Path, objetivo: &Path, dir: &Path) {
    if let Some(p) = lnk.parent() {
        let _ = std::fs::create_dir_all(p);
    }
    let script = format!(
        "$s=(New-Object -ComObject WScript.Shell).CreateShortcut('{}');$s.TargetPath='{}';$s.WorkingDirectory='{}';$s.IconLocation='{},0';$s.Description='Reescaller';$s.Save()",
        escapar_ps(&lnk.display().to_string()),
        escapar_ps(&objetivo.display().to_string()),
        escapar_ps(&dir.display().to_string()),
        escapar_ps(&objetivo.display().to_string())
    );
    let _ = ejecutar(
        "powershell.exe",
        &[
            "-NoProfile",
            "-NonInteractive",
            "-ExecutionPolicy",
            "Bypass",
            "-Command",
            &script,
        ],
    );
}

fn registrar_desinstalacion(dir: &Path, kb: u64) {
    let exe = dir.join("Reescaller.exe");
    let un = dir.join("Uninstall.exe");
    let add = |name: &str, tipo: &str, valor: &str| {
        let _ = ejecutar(
            "reg.exe",
            &[
                "add",
                UNINSTALL_KEY,
                "/v",
                name,
                "/t",
                tipo,
                "/d",
                valor,
                "/f",
            ],
        );
    };
    add("DisplayName", "REG_SZ", "Reescaller");
    add("DisplayVersion", "REG_SZ", VERSION);
    add("Publisher", "REG_SZ", "Reescaller");
    add("InstallLocation", "REG_SZ", &dir.display().to_string());
    add("DisplayIcon", "REG_SZ", &format!("{},0", exe.display()));
    add(
        "UninstallString",
        "REG_SZ",
        &format!("\"{}\"", un.display()),
    );
    add(
        "QuietUninstallString",
        "REG_SZ",
        &format!("\"{}\" /S", un.display()),
    );
    add("NoModify", "REG_DWORD", "1");
    add("NoRepair", "REG_DWORD", "1");
    add("EstimatedSize", "REG_DWORD", &kb.to_string());
}

// ---------------------------------------------------------------------------------------------------------------- instalar

fn instalar(opts: &Opts, exe: &Path, emit: &dyn Fn(Value)) -> Result<Vec<String>, String> {
    let mut zip = abrir_payload(exe)?;
    let mut plan: Vec<(usize, u64)> = Vec::new();
    let mut total = 0u64;
    for i in 0..zip.len() {
        let e = zip.by_index_raw(i).map_err(|e| e.to_string())?;
        let n = e.name().to_string();
        if e.is_dir()
            || n.starts_with("redist/")
            || n.starts_with("ui/")
            || (!opts.faces && n.starts_with("models/onnx/"))
        {
            continue;
        }
        plan.push((i, e.size()));
        total += e.size();
    }
    if plan.is_empty() {
        return Err("El instalador no contiene archivos.".into());
    }
    std::fs::create_dir_all(&opts.dir)
        .map_err(|e| format!("No se pudo crear la carpeta {}: {e}", opts.dir.display()))?;

    let t0 = Instant::now();
    let mut hecho = 0u64;
    let mut ultimo = Instant::now() - Duration::from_secs(1);
    let mut buf = vec![0u8; 1 << 20];
    for (i, _) in &plan {
        let mut e = zip.by_index(*i).map_err(|e| e.to_string())?;
        let rel = match e.enclosed_name() {
            Some(p) => p.to_owned(),
            None => continue,
        };
        let destino = opts.dir.join(&rel);
        if let Some(p) = destino.parent() {
            std::fs::create_dir_all(p)
                .map_err(|er| format!("No se pudo crear {}: {er}", p.display()))?;
        }
        let mut out = File::create(&destino).map_err(|er| {
            format!("No se pudo escribir {}: {er}. Si Reescaller está abierto, ciérralo y vuelve a intentarlo.", destino.display())
        })?;
        let nombre = rel.display().to_string().replace('\\', "/");
        loop {
            let n = e
                .read(&mut buf)
                .map_err(|er| format!("Error al leer {nombre}: {er}"))?;
            if n == 0 {
                break;
            }
            out.write_all(&buf[..n])
                .map_err(|er| format!("Error al escribir {nombre}: {er}"))?;
            hecho += n as u64;
            if ultimo.elapsed() >= Duration::from_millis(90) {
                ultimo = Instant::now();
                let seg = t0.elapsed().as_secs_f64().max(0.001);
                let vel = hecho as f64 / seg;
                let resto = if vel > 1.0 {
                    ((total - hecho) as f64 / vel) as u64
                } else {
                    0
                };
                emit(
                    json!({ "ev": "progress", "pct": hecho as f64 / total as f64 * 100.0, "file": nombre, "eta": resto, "done": hecho, "total": total }),
                );
            }
        }
    }
    emit(
        json!({ "ev": "progress", "pct": 99.0, "file": "", "eta": 0, "done": total, "total": total, "step": "shortcuts" }),
    );

    // Desinstalador = la interfaz del instalador (sin payload); necesita WebView2Loader.dll al lado (normalmente ya viene con la app).
    let mut copiar = |nombre: &str, destino: PathBuf| -> Result<(), String> {
        let mut e = zip
            .by_name(nombre)
            .map_err(|_| format!("El instalador está incompleto (falta {nombre})."))?;
        let mut out = File::create(&destino)
            .map_err(|er| format!("No se pudo escribir {}: {er}", destino.display()))?;
        std::io::copy(&mut e, &mut out).map_err(|er| er.to_string())?;
        Ok(())
    };
    copiar("ui/reescaller-setup.exe", opts.dir.join("Uninstall.exe"))?;
    if !opts.dir.join("WebView2Loader.dll").is_file() {
        copiar("ui/WebView2Loader.dll", opts.dir.join("WebView2Loader.dll"))?;
    }

    let app = opts.dir.join("Reescaller.exe");
    if let Some(menu) = carpeta_inicio() {
        crear_acceso(&menu.join("Reescaller.lnk"), &app, &opts.dir);
    }
    if opts.desktop {
        if let Some(esc) = escritorio() {
            crear_acceso(&esc.join("Reescaller.lnk"), &app, &opts.dir);
        }
    }
    registrar_desinstalacion(&opts.dir, total / 1024);

    let mut avisos = Vec::new();
    if !vulkan_presente() {
        avisos.push("novulkan".to_string());
    }
    emit(
        json!({ "ev": "progress", "pct": 100.0, "file": "", "eta": 0, "done": total, "total": total }),
    );
    Ok(avisos)
}

// ---------------------------------------------------------------------------------------------------------------- desinstalar

fn desinstalar(dir: &Path, borrar_datos: bool, emit: &dyn Fn(Value)) -> Result<(), String> {
    emit(json!({ "ev": "progress", "pct": 10.0, "file": "accesos directos" }));
    if let Some(m) = carpeta_inicio() {
        let _ = std::fs::remove_file(m.join("Reescaller.lnk"));
    }
    if let Some(e) = escritorio() {
        let _ = std::fs::remove_file(e.join("Reescaller.lnk"));
    }
    emit(json!({ "ev": "progress", "pct": 30.0, "file": "registro" }));
    let _ = ejecutar("reg.exe", &["delete", UNINSTALL_KEY, "/f"]);

    if borrar_datos {
        emit(json!({ "ev": "progress", "pct": 45.0, "file": "ajustes y caché" }));
        if let Some(a) = std::env::var_os("APPDATA") {
            let _ = std::fs::remove_dir_all(PathBuf::from(a).join("Reescaller"));
        }
        if let Some(l) = std::env::var_os("LOCALAPPDATA") {
            let _ = std::fs::remove_dir_all(PathBuf::from(l).join("Reescaller"));
        }
    } else if let Some(l) = std::env::var_os("LOCALAPPDATA") {
        let base = PathBuf::from(l).join("Reescaller");
        for n in ["webview", "webview2", "uploads"] {
            let _ = std::fs::remove_dir_all(base.join(n));
        }
    }

    emit(json!({ "ev": "progress", "pct": 60.0, "file": "archivos" }));
    let yo = std::env::current_exe().ok();
    let mut fallos = 0;
    if let Ok(rd) = std::fs::read_dir(dir) {
        let entradas: Vec<_> = rd.flatten().collect();
        let n = entradas.len().max(1);
        for (k, e) in entradas.iter().enumerate() {
            let p = e.path();
            if Some(&p) == yo.as_ref() {
                continue;
            }
            // El cargador de WebView2 está cargado por este mismo proceso: lo quita la orden final.
            let cargada = p
                .file_name()
                .map(|n| {
                    n.to_string_lossy()
                        .eq_ignore_ascii_case("WebView2Loader.dll")
                })
                .unwrap_or(false);
            let r = if p.is_dir() {
                std::fs::remove_dir_all(&p)
            } else {
                std::fs::remove_file(&p)
            };
            if r.is_err() && !cargada {
                fallos += 1;
            }
            emit(
                json!({ "ev": "progress", "pct": 60.0 + 38.0 * (k + 1) as f64 / n as f64, "file": p.file_name().map(|s| s.to_string_lossy().to_string()).unwrap_or_default() }),
            );
        }
    }
    // El desinstalador no puede borrarse a sí mismo mientras corre: una orden aparte quita la carpeta un par de segundos después.
    let carpeta = dir.display().to_string();
    let mut c = Command::new("cmd.exe");
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        // Reintenta hasta 2 minutos: en modo con ventana este programa sigue abierto hasta que el usuario lo cierra.
        c.raw_arg(format!(
            "/C for /l %i in (1,1,120) do (ping -n 2 127.0.0.1 >nul & rmdir /s /q \"{carpeta}\" 2>nul & if not exist \"{carpeta}\" exit)"
        ));
        // Solo CREATE_NO_WINDOW: con una consola oculta `ping` funciona como pausa (sin consola falla al instante).
        c.creation_flags(0x0800_0000);
    }
    let _ = c.spawn();
    emit(json!({ "ev": "progress", "pct": 100.0, "file": "" }));
    if fallos > 0 {
        Err("Algunos archivos estaban en uso (¿Reescaller está abierto?). Ciérralo y vuelve a desinstalar.".into())
    } else {
        Ok(())
    }
}

// ---------------------------------------------------------------------------------------------------------------- selector de carpeta

fn elegir_carpeta(inicial: &str) -> Option<String> {
    let script = format!(
        "Add-Type -AssemblyName System.Windows.Forms;$f=New-Object System.Windows.Forms.FolderBrowserDialog;$f.Description='Elige dónde instalar Reescaller';$f.SelectedPath='{}';$f.ShowNewFolderButton=$true;if($f.ShowDialog() -eq 'OK'){{[Console]::OutputEncoding=[Text.Encoding]::UTF8;Write-Output $f.SelectedPath}}",
        escapar_ps(inicial)
    );
    let o = ejecutar(
        "powershell.exe",
        &[
            "-NoProfile",
            "-STA",
            "-ExecutionPolicy",
            "Bypass",
            "-Command",
            &script,
        ],
    )?;
    let t = String::from_utf8_lossy(&o.stdout).trim().to_string();
    if t.is_empty() {
        None
    } else {
        Some(t)
    }
}

// ---------------------------------------------------------------------------------------------------------------- principal

fn icono() -> Option<Icon> {
    let png = ICO.get(22..)?;
    let img = image::load_from_memory(png).ok()?.to_rgba8();
    let (w, h) = img.dimensions();
    Icon::from_rgba(img.into_raw(), w, h).ok()
}

fn js(proxy: &EventLoopProxy<Ev>, v: Value) {
    let _ = proxy.send_event(Ev::Js(format!("window.__ev&&window.__ev({})", v)));
}

fn arg_valor(args: &[String], clave: &str) -> Option<String> {
    args.iter().find_map(|a| {
        a.strip_prefix(&format!("{clave}="))
            .map(|s| s.trim_matches('"').to_string())
    })
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let exe = std::env::current_exe().unwrap_or_default();
    // El lanzador pasa `--payload=<instalador original>`: ahí está el ZIP con la app (Uninstall.exe no lo necesita).
    let payload = arg_valor(&args, "--payload")
        .map(PathBuf::from)
        .unwrap_or_else(|| exe.clone());
    let nombre = exe
        .file_name()
        .map(|s| s.to_string_lossy().to_lowercase())
        .unwrap_or_default();
    let desinst = nombre.starts_with("uninstall")
        || args.iter().any(|a| a == "--uninstall" || a == "/uninstall");
    let silencioso = args.iter().any(|a| {
        matches!(
            a.to_lowercase().as_str(),
            "/s" | "/silent" | "/verysilent" | "--silent"
        )
    });
    let dir_arg = arg_valor(&args, "--dir").or_else(|| arg_valor(&args, "/DIR"));

    // -------- modo silencioso (sin ventana)
    if silencioso {
        let code = if desinst {
            let dir = exe.parent().map(|p| p.to_path_buf()).unwrap_or_default();
            match desinstalar(&dir, args.iter().any(|a| a == "--purge"), &|_| {}) {
                Ok(()) => 0,
                Err(_) => 1,
            }
        } else {
            asegurar_webview2(&payload);
            let opts = Opts {
                dir: dir_arg.map(PathBuf::from).unwrap_or_else(dir_por_defecto),
                faces: !args.iter().any(|a| a == "--no-faces"),
                desktop: !args.iter().any(|a| a == "--no-desktop"),
            };
            match instalar(&opts, &payload, &|_| {}) {
                Ok(_) => 0,
                Err(_) => 1,
            }
        };
        std::process::exit(code);
    }

    if !desinst {
        asegurar_webview2(&payload);
    }

    let event_loop = EventLoopBuilder::<Ev>::with_user_event().build();
    let proxy = event_loop.create_proxy();
    let proxy_show = event_loop.create_proxy();
    std::thread::spawn(move || {
        std::thread::sleep(Duration::from_secs(4));
        let _ = proxy_show.send_event(Ev::Show);
    });

    let window = WindowBuilder::new()
        .with_title(if desinst {
            "Desinstalar Reescaller"
        } else {
            "Instalar Reescaller"
        })
        .with_decorations(false)
        .with_resizable(false)
        .with_visible(false)
        .with_inner_size(LogicalSize::new(920.0, 600.0))
        .with_window_icon(icono())
        .build(&event_loop)
        .expect("ventana");
    if let Some(m) = window.primary_monitor() {
        let (ms, ws) = (m.size(), window.outer_size());
        window.set_outer_position(PhysicalPosition::new(
            m.position().x + (ms.width as i32 - ws.width as i32) / 2,
            m.position().y + (ms.height as i32 - ws.height as i32) / 2,
        ));
    }

    let datos = std::env::temp_dir().join("Reescaller-Setup-webview");
    let mut ctx = WebContext::new(Some(datos));
    let dir_instalado = exe.parent().map(|p| p.to_path_buf()).unwrap_or_default();
    let exe_ipc = payload.clone();
    let proxy_ipc = proxy.clone();
    let webview = WebViewBuilder::with_web_context(&mut ctx)
        .with_html(HTML)
        .with_background_color((10, 13, 20, 255))
        .with_ipc_handler(move |req| {
            let v: Value = serde_json::from_str(req.body()).unwrap_or(Value::Null);
            let cmd = v["cmd"].as_str().unwrap_or("").to_string();
            let p = proxy_ipc.clone();
            match cmd.as_str() {
                "drag" | "min" | "close" | "ready" => {
                    let _ = p.send_event(Ev::Js(format!("@{cmd}")));
                }
                "init" => {
                    let exe = exe_ipc.clone();
                    let dir_inst = dir_instalado.clone();
                    std::thread::spawn(move || {
                        let (total, rostros) = if desinst { (0, 0) } else { medir_payload(&exe) };
                        let defecto = if desinst { dir_inst.display().to_string() } else { dir_por_defecto().display().to_string() };
                        js(&p, json!({ "ev": "init", "mode": if desinst { "uninstall" } else { "install" }, "dir": defecto, "version": VERSION,
                                       "total": total, "faces": rostros, "vulkan": vulkan_presente(), "payload": total > 0 || desinst }));
                    });
                }
                "pickdir" => {
                    let ini = v["dir"].as_str().unwrap_or("").to_string();
                    std::thread::spawn(move || {
                        if let Some(d) = elegir_carpeta(&ini) {
                            js(&p, json!({ "ev": "dir", "dir": d }));
                        }
                    });
                }
                "install" => {
                    let opts = Opts {
                        dir: PathBuf::from(v["dir"].as_str().unwrap_or("")),
                        faces: v["faces"].as_bool().unwrap_or(true),
                        desktop: v["desktop"].as_bool().unwrap_or(true),
                    };
                    let exe = exe_ipc.clone();
                    std::thread::spawn(move || {
                        let emit = |x: Value| js(&p, x);
                        match instalar(&opts, &exe, &emit) {
                            Ok(avisos) => js(&p, json!({ "ev": "done", "warnings": avisos, "dir": opts.dir.display().to_string() })),
                            Err(e) => js(&p, json!({ "ev": "error", "message": e })),
                        }
                    });
                }
                "uninstall" => {
                    let dir = dir_instalado.clone();
                    let purga = v["purge"].as_bool().unwrap_or(false);
                    std::thread::spawn(move || {
                        let emit = |x: Value| js(&p, x);
                        match desinstalar(&dir, purga, &emit) {
                            Ok(()) => js(&p, json!({ "ev": "done", "warnings": [] })),
                            Err(e) => js(&p, json!({ "ev": "error", "message": e })),
                        }
                    });
                }
                "launch" => {
                    let dir = PathBuf::from(v["dir"].as_str().unwrap_or(""));
                    let app = dir.join("Reescaller.exe");
                    let mut c = Command::new(&app);
                    c.current_dir(&dir);
                    #[cfg(windows)]
                    {
                        use std::os::windows::process::CommandExt;
                        c.creation_flags(0x0000_0008 | 0x0000_0200);
                    }
                    let _ = c.spawn();
                    let _ = p.send_event(Ev::Quit);
                }
                _ => {}
            }
        })
        .build(&window)
        .expect("WebView2");

    event_loop.run(move |event, _, control_flow| {
        *control_flow = ControlFlow::Wait;
        match event {
            Event::UserEvent(Ev::Js(s)) => match s.as_str() {
                "@drag" => {
                    let _ = window.drag_window();
                }
                "@min" => window.set_minimized(true),
                "@close" => std::process::exit(0),
                "@ready" => {
                    window.set_visible(true);
                    window.set_minimized(false);
                    window.set_focus();
                }
                script => {
                    let _ = webview.evaluate_script(script);
                }
            },
            Event::UserEvent(Ev::Show) => {
                if !window.is_visible() {
                    window.set_visible(true);
                    window.set_minimized(false);
                    window.set_focus();
                }
            }
            Event::UserEvent(Ev::Quit) => std::process::exit(0),
            Event::WindowEvent {
                event: WindowEvent::CloseRequested,
                ..
            } => std::process::exit(0),
            Event::WindowEvent {
                event: WindowEvent::Focused(true),
                ..
            } => {
                let _ = webview.focus();
            }
            _ => {}
        }
    });
}
