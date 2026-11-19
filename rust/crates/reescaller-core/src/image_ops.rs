//! Operaciones de imagen: reescalado controlado, mezcla de mejora, nitidez y
//! codificación con "peso" (calidad / tamaño objetivo).
//!
//! Este módulo es el reemplazo de la parte de OpenCV/PIL del proyecto Python.

use crate::config::{OutputFormat, ResizeFilter, Weight};
use crate::error::{Error, Result};
use image::codecs::jpeg::JpegEncoder;
use image::codecs::png::{CompressionType, FilterType as PngFilter, PngEncoder};
use image::{DynamicImage, GenericImageView, ImageBuffer, ImageEncoder, Rgb, RgbImage};
use std::io::Cursor;
use std::path::Path;

/// Carga una imagen desde disco.
pub fn load(path: &Path) -> Result<DynamicImage> {
    if !path.is_file() {
        return Err(Error::FileNotFound(path.to_path_buf()));
    }
    let err = |e: &dyn std::fmt::Display| Error::Image(format!("{}: {e}", path.display()));
    // Aplica la orientación EXIF (las fotos de móvil vienen «giradas» en los metadatos).
    let decoder = image::ImageReader::open(path)
        .map_err(|e| err(&e))?
        .with_guessed_format()
        .map_err(|e| err(&e))?
        .into_decoder();
    match decoder {
        Ok(mut dec) => {
            let orientation = image::ImageDecoder::orientation(&mut dec)
                .unwrap_or(image::metadata::Orientation::NoTransforms);
            let mut img = DynamicImage::from_decoder(dec).map_err(|e| err(&e))?;
            img.apply_orientation(orientation);
            Ok(img)
        }
        Err(_) => image::open(path).map_err(|e| err(&e)),
    }
}

/// Tamaño (ancho, alto) aplicando una escala al tamaño original.
pub fn scaled_size(original: (u32, u32), scale: f32) -> (u32, u32) {
    let w = ((original.0 as f32) * scale).round().max(1.0) as u32;
    let h = ((original.1 as f32) * scale).round().max(1.0) as u32;
    (w, h)
}

/// Redimensiona una imagen con el filtro indicado.
pub fn resize(img: &DynamicImage, width: u32, height: u32, filter: ResizeFilter) -> DynamicImage {
    let filter = match filter {
        ResizeFilter::Lanczos3 => image::imageops::FilterType::Lanczos3,
        ResizeFilter::CatmullRom => image::imageops::FilterType::CatmullRom,
        ResizeFilter::Bilinear => image::imageops::FilterType::Triangle,
        ResizeFilter::Nearest => image::imageops::FilterType::Nearest,
    };
    let resized = image::imageops::resize(img, width, height, filter);
    DynamicImage::ImageRgba8(resized)
}

/// Mezcla `over` (imagen mejorada) sobre `base`.
///
/// `percent` es el porcentaje de la imagen mejorada que se conserva
/// (100 = solo la mejorada, 0 = solo la original redimensionada).
pub fn blend(base: &DynamicImage, over: &DynamicImage, percent: u8) -> DynamicImage {
    let pct = percent.min(100) as f32 / 100.0;
    if pct >= 1.0 {
        return over.clone();
    }
    if pct <= 0.0 {
        return base.clone();
    }

    let (w, h) = over.dimensions();
    let base_rgb = base.to_rgb8();
    let over_rgb = over.to_rgb8();

    let mut out = RgbImage::new(w, h);
    for (x, y, pixel) in out.enumerate_pixels_mut() {
        let b = base_rgb.get_pixel(x.min(base_rgb.width() - 1), y.min(base_rgb.height() - 1));
        let o = over_rgb.get_pixel(x, y);
        *pixel = Rgb([
            (b[0] as f32 + (o[0] as f32 - b[0] as f32) * pct)
                .round()
                .clamp(0.0, 255.0) as u8,
            (b[1] as f32 + (o[1] as f32 - b[1] as f32) * pct)
                .round()
                .clamp(0.0, 255.0) as u8,
            (b[2] as f32 + (o[2] as f32 - b[2] as f32) * pct)
                .round()
                .clamp(0.0, 255.0) as u8,
        ]);
    }
    DynamicImage::ImageRgb8(out)
}

/// Aplica nitidez (unsharp mask 3x3) con intensidad 0..=100.
pub fn sharpen(img: &DynamicImage, amount: u8) -> DynamicImage {
    let k = amount.min(100) as f32 / 100.0;
    if k <= 0.0 {
        return img.clone();
    }
    let src = img.to_rgb8();
    let (w, h) = src.dimensions();
    let mut out: RgbImage = ImageBuffer::new(w, h);

    for y in 0..h {
        for x in 0..w {
            let center = src.get_pixel(x, y);
            let mut acc = [0f32; 3];
            for (dx, dy) in [
                (-1i32, -1i32),
                (0, -1),
                (1, -1),
                (-1, 0),
                (1, 0),
                (-1, 1),
                (0, 1),
                (1, 1),
            ] {
                let nx = (x as i32 + dx).clamp(0, w as i32 - 1) as u32;
                let ny = (y as i32 + dy).clamp(0, h as i32 - 1) as u32;
                let p = src.get_pixel(nx, ny);
                for c in 0..3 {
                    acc[c] += p[c] as f32;
                }
            }
            let mut value = [0u8; 3];
            for c in 0..3 {
                let blur = acc[c] / 8.0;
                let sharpened = center[c] as f32 + (center[c] as f32 - blur) * k * 1.5;
                value[c] = sharpened.round().clamp(0.0, 255.0) as u8;
            }
            out.put_pixel(x, y, Rgb(value));
        }
    }
    DynamicImage::ImageRgb8(out)
}

/// Codifica la imagen aplicando el "peso" elegido.
///
/// - `Weight::Quality { quality }`: JPEG con esa calidad; en PNG se mapea a
///   nivel de compresión (calidad alta = archivo más pequeño).
/// - `Weight::TargetSize { target_kb }`: se busca por búsqueda binaria la calidad
///   JPEG que más se acerca al tamaño objetivo.
pub fn encode(img: &DynamicImage, format: OutputFormat, weight: Weight) -> Result<Vec<u8>> {
    match (format, weight) {
        (OutputFormat::Jpg, Weight::Quality { quality }) => encode_jpeg(img, quality.clamp(1, 100)),
        (OutputFormat::Jpg, Weight::TargetSize { target_kb }) => {
            let target = target_kb.max(1) as u64 * 1024;
            let mut best: Option<(u8, Vec<u8>)> = None;
            let mut best_diff = u64::MAX;

            let mut lo = 10u8;
            let mut hi = 100u8;
            while lo <= hi {
                let mid = (lo + hi) / 2;
                let data = encode_jpeg(img, mid)?;
                let size = data.len() as u64;
                let diff = size.abs_diff(target);
                if diff < best_diff {
                    best_diff = diff;
                    best = Some((mid, data));
                }
                if size > target {
                    if mid == lo {
                        break;
                    }
                    hi = mid - 1;
                } else {
                    if mid == hi {
                        break;
                    }
                    lo = mid + 1;
                }
            }
            match best {
                Some((q, data)) => {
                    log::info!(
                        "Peso ajustado: calidad {} -> {:.1} KB (objetivo {} KB)",
                        q,
                        data.len() as f64 / 1024.0,
                        target_kb
                    );
                    Ok(data)
                }
                None => encode_jpeg(img, 92),
            }
        }
        (OutputFormat::Png, Weight::Quality { quality }) => {
            let compression = png_compression_for(quality);
            let rgb = img.to_rgb8();
            let mut buf: Vec<u8> = Vec::new();
            {
                let cursor = Cursor::new(&mut buf);
                let encoder =
                    PngEncoder::new_with_quality(cursor, compression, PngFilter::Adaptive);
                encoder
                    .write_image(
                        rgb.as_raw(),
                        rgb.width(),
                        rgb.height(),
                        image::ExtendedColorType::Rgb8,
                    )
                    .map_err(|e| Error::Image(format!("escritura PNG: {e}")))?;
            }
            Ok(buf)
        }
        (OutputFormat::Png, Weight::TargetSize { target_kb }) => {
            // PNG es sin pérdida: el tamaño solo depende de la compresión.
            let rgb = img.to_rgb8();
            let mut buf: Vec<u8> = Vec::new();
            {
                let cursor = Cursor::new(&mut buf);
                let encoder = PngEncoder::new_with_quality(
                    cursor,
                    CompressionType::Best,
                    PngFilter::Adaptive,
                );
                encoder
                    .write_image(
                        rgb.as_raw(),
                        rgb.width(),
                        rgb.height(),
                        image::ExtendedColorType::Rgb8,
                    )
                    .map_err(|e| Error::Image(format!("escritura PNG: {e}")))?;
            }
            log::info!(
                "PNG es sin pérdida: {:.1} KB generados (objetivo {} KB no aplicable al compress)",
                buf.len() as f64 / 1024.0,
                target_kb
            );
            Ok(buf)
        }
    }
}

fn png_compression_for(quality: u8) -> CompressionType {
    // Calidad alta -> compresión alta (archivo más pequeño pero más lento).
    match quality {
        0..=20 => CompressionType::Fast,
        21..=50 => CompressionType::Default,
        51..=85 => CompressionType::Best,
        _ => CompressionType::Best,
    }
}

fn encode_jpeg(img: &DynamicImage, quality: u8) -> Result<Vec<u8>> {
    let rgb = img.to_rgb8();
    let mut buf: Vec<u8> = Vec::new();
    {
        let mut cursor = Cursor::new(&mut buf);
        let mut encoder = JpegEncoder::new_with_quality(&mut cursor, quality);
        encoder
            .encode_image(&rgb)
            .map_err(|e| Error::Image(format!("escritura JPEG: {e}")))?;
    }
    Ok(buf)
}

/// Guarda la imagen ya codificada.
pub fn save_bytes(data: &[u8], path: &Path) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(path, data)?;
    Ok(())
}

/// Decodifica bytes de imagen (útil para previsualizaciones HTTP).
pub fn load_bytes(data: &[u8]) -> Result<DynamicImage> {
    image::load_from_memory(data).map_err(|e| Error::Image(format!("decodificación: {e}")))
}

/// Crea una miniatura proporcional (máx. `max_side`).
pub fn thumbnail(img: &DynamicImage, max_side: u32) -> DynamicImage {
    let (w, h) = img.dimensions();
    let longest = w.max(h);
    if longest <= max_side {
        return img.clone();
    }
    let ratio = max_side as f32 / longest as f32;
    resize(
        img,
        ((w as f32 * ratio).round() as u32).max(1),
        ((h as f32 * ratio).round() as u32).max(1),
        ResizeFilter::Lanczos3,
    )
}

/// Luminosidad media (0..255) de una imagen.
pub fn mean_luma(img: &DynamicImage) -> f32 {
    let g = img.to_luma8();
    let n = (g.width() as u64 * g.height() as u64).max(1);
    g.pixels().map(|p| p.0[0] as u64).sum::<u64>() as f32 / n as f32
}

/// Aclara una foto oscura (gamma hacia ~105/255) y suaviza un poco el ruido de poca luz.
/// Con `solo_si_oscura` solo actúa si la luminosidad media es baja (< 75). Devuelve `None` si no hizo nada.
pub fn lift_dark(img: &DynamicImage, solo_si_oscura: bool) -> Option<DynamicImage> {
    let media = mean_luma(img).max(1.0);
    if solo_si_oscura && media >= 75.0 {
        return None;
    }
    let gamma = if media < 105.0 {
        ((105.0f32 / 255.0).ln() / (media / 255.0).ln()).clamp(0.5, 1.0)
    } else {
        1.0
    };
    let mut lut = [0u8; 256];
    for (i, v) in lut.iter_mut().enumerate() {
        *v = (255.0 * (i as f32 / 255.0).powf(gamma))
            .round()
            .clamp(0.0, 255.0) as u8;
    }
    let mut rgb = img.to_rgb8();
    for p in rgb.pixels_mut() {
        for c in p.0.iter_mut() {
            *c = lut[*c as usize];
        }
    }
    // el ruido de las fotos oscuras se acentúa al aclarar: un desenfoque muy ligero lo contiene y la IA devuelve el detalle
    let suave = image::imageops::blur(&rgb, 0.7);
    Some(DynamicImage::ImageRgb8(suave))
}
