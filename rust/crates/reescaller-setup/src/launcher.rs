//! Lanzador del instalador: es el `.exe` que descarga el usuario (lanzador + ZIP con todo al final).
//!
//! La interfaz del instalador (`reescaller-setup.exe`, con WebView2) necesita `WebView2Loader.dll` al lado, y un único `.exe` no
//! puede llevarla. Por eso este programa, que no depende de ninguna DLL, extrae la interfaz y su DLL a una carpeta temporal,
//! la ejecuta con los mismos argumentos (más `--payload=<este exe>` para que sepa dónde está la app) y espera a que termine.
#![cfg_attr(windows, windows_subsystem = "windows")]

use std::fs::File;
use std::io::BufReader;
use std::path::{Path, PathBuf};
use std::process::Command;

fn extraer(
    z: &mut zip::ZipArchive<BufReader<File>>,
    nombre: &str,
    destino: &Path,
) -> Result<(), String> {
    let mut e = z.by_name(nombre).map_err(|_| {
        format!("El instalador está incompleto (falta {nombre}). Descárgalo de nuevo.")
    })?;
    let mut out = File::create(destino)
        .map_err(|er| format!("No se pudo escribir {}: {er}", destino.display()))?;
    std::io::copy(&mut e, &mut out).map_err(|er| format!("No se pudo extraer {nombre}: {er}"))?;
    Ok(())
}

fn ejecutar(exe: &Path, args: &[String]) -> Result<i32, String> {
    let f = File::open(exe).map_err(|e| format!("No se pudo leer el instalador: {e}"))?;
    let mut z = zip::ZipArchive::new(BufReader::new(f))
        .map_err(|e| format!("El instalador está dañado o incompleto ({e})."))?;
    let tmp: PathBuf = std::env::temp_dir().join(format!("ReescallerSetup-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&tmp);
    std::fs::create_dir_all(&tmp)
        .map_err(|e| format!("No se pudo crear la carpeta temporal: {e}"))?;

    let res = (|| {
        let ui = tmp.join("reescaller-setup.exe");
        extraer(&mut z, "ui/reescaller-setup.exe", &ui)?;
        extraer(
            &mut z,
            "ui/WebView2Loader.dll",
            &tmp.join("WebView2Loader.dll"),
        )?;
        let mut c = Command::new(&ui);
        c.args(args)
            .arg(format!("--payload={}", exe.display()))
            .current_dir(&tmp);
        let mut hijo = c
            .spawn()
            .map_err(|e| format!("No se pudo iniciar el instalador: {e}"))?;
        let estado = hijo.wait().map_err(|e| e.to_string())?;
        Ok(estado.code().unwrap_or(1))
    })();
    let _ = std::fs::remove_dir_all(&tmp);
    res
}

fn aviso(texto: &str) {
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        let t = texto.replace('\'', "''");
        let _ = Command::new("powershell.exe")
            .args(["-NoProfile", "-NonInteractive", "-Command", &format!("Add-Type -AssemblyName PresentationFramework;[void][System.Windows.MessageBox]::Show('{t}','Reescaller')")])
            .creation_flags(0x0800_0000)
            .status();
    }
    #[cfg(not(windows))]
    eprintln!("{texto}");
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let silencioso = args.iter().any(|a| {
        matches!(
            a.to_lowercase().as_str(),
            "/s" | "/silent" | "/verysilent" | "--silent"
        )
    });
    let exe = std::env::current_exe().unwrap_or_default();
    match ejecutar(&exe, &args) {
        Ok(c) => std::process::exit(c),
        Err(e) => {
            if !silencioso {
                aviso(&e);
            }
            std::process::exit(1);
        }
    }
}
