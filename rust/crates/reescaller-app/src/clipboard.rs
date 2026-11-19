//! Lectura del portapapeles de Windows: imagen (Recortes, capturas, navegadores) o archivos copiados/cortados.
//!
//! La interfaz web no siempre recibe la imagen en el evento `paste` (depende de cómo la deje la app de origen),
//! así que la aplicación puede leerla directamente del sistema.

use std::path::PathBuf;

/// Lo que hay en el portapapeles.
pub enum Clip {
    /// Un archivo de imagen copiado o cortado en el Explorador.
    File(PathBuf),
    /// Imagen PNG (formato «PNG» registrado).
    Png(Vec<u8>),
    /// Mapa de bits (BMP completo con cabecera).
    Bmp(Vec<u8>),
}

fn es_imagen(p: &std::path::Path) -> bool {
    matches!(
        p.extension()
            .map(|e| e.to_string_lossy().to_lowercase())
            .as_deref(),
        Some("png" | "jpg" | "jpeg" | "webp" | "bmp" | "tif" | "tiff")
    )
}

#[cfg(windows)]
pub fn read() -> Result<Option<Clip>, String> {
    use clipboard_win::{formats, get_clipboard, is_format_avail, register_format, Clipboard};

    let _clip = Clipboard::new_attempts(10)
        .map_err(|e| format!("No se pudo abrir el portapapeles: {e}"))?;

    // 1) archivos copiados o cortados
    if is_format_avail(formats::CF_HDROP) {
        if let Ok(lista) = get_clipboard::<Vec<String>, _>(formats::FileList) {
            for f in lista {
                let p = PathBuf::from(f);
                if p.is_file() && es_imagen(&p) {
                    return Ok(Some(Clip::File(p)));
                }
            }
        }
    }
    // 2) PNG (Recortes de Windows y la mayoría de apps)
    if let Some(png) = register_format("PNG") {
        if is_format_avail(png.get()) {
            if let Ok(bytes) = get_clipboard::<Vec<u8>, _>(formats::RawData(png.get())) {
                if !bytes.is_empty() {
                    return Ok(Some(Clip::Png(bytes)));
                }
            }
        }
    }
    // 3) mapa de bits clásico
    if is_format_avail(formats::CF_BITMAP)
        || is_format_avail(formats::CF_DIB)
        || is_format_avail(formats::CF_DIBV5)
    {
        if let Ok(bytes) = get_clipboard::<Vec<u8>, _>(formats::Bitmap) {
            if !bytes.is_empty() {
                return Ok(Some(Clip::Bmp(bytes)));
            }
        }
    }
    Ok(None)
}

#[cfg(not(windows))]
pub fn read() -> Result<Option<Clip>, String> {
    Ok(None)
}
