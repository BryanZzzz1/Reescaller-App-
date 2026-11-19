//! Tests del núcleo: reescalado controlado, peso y configuración.

use reescaller_core::config::{AppConfig, OutputFormat, ResizeMode, Weight};
use reescaller_core::image_ops;
use reescaller_core::upscale::UpscalePlan;
use reescaller_core::video::format_fps;

fn imagen(w: u32, h: u32) -> image::DynamicImage {
    image::DynamicImage::ImageRgb8(image::RgbImage::from_pixel(w, h, image::Rgb([120, 90, 60])))
}

#[test]
fn escala_numerica_respeta_proporciones() {
    assert_eq!(image_ops::scaled_size((400, 300), 1.5), (600, 450));
    assert_eq!(image_ops::scaled_size((400, 300), 2.0), (800, 600));
    assert_eq!(image_ops::scaled_size((400, 300), 0.5), (200, 150));
    // Nunca debe colapsar a cero.
    assert_eq!(image_ops::scaled_size((10, 10), 0.01), (1, 1));
}

#[test]
fn modo_only_if_needed_no_reduce() {
    let mut plan = UpscalePlan {
        target_scale: 2.0,
        resize_mode: ResizeMode::OnlyIfNeeded,
        ..Default::default()
    };
    // Original menor que el objetivo -> escala.
    assert_eq!(plan.target_dimensions((400, 300)), (800, 600));
    // Original mayor o igual -> conserva.
    plan.target_scale = 0.5;
    assert_eq!(plan.target_dimensions((800, 600)), (800, 600));
}

#[test]
fn modo_improve_only_conserva_tamano() {
    let plan = UpscalePlan {
        target_scale: 4.0,
        resize_mode: ResizeMode::ImproveOnly,
        ..Default::default()
    };
    assert_eq!(plan.target_dimensions((400, 300)), (400, 300));
}

#[test]
fn escala_del_motor_depende_del_modelo() {
    let x4 = UpscalePlan {
        engine_scale: 2,
        model: "realesrgan-x4plus".into(),
        ..Default::default()
    };
    assert_eq!(x4.effective_engine_scale(), 4, "x4plus solo escala a x4");

    let anime = UpscalePlan {
        engine_scale: 2,
        model: "realesr-animevideov3".into(),
        ..Default::default()
    };
    assert_eq!(anime.effective_engine_scale(), 2, "animevideov3 admite x2");
}

#[test]
fn mezcla_respeta_el_porcentaje_de_mejora() {
    let base = imagen(10, 10);
    let mejorada = image::DynamicImage::ImageRgb8(image::RgbImage::from_pixel(
        10,
        10,
        image::Rgb([200, 200, 200]),
    ));

    let cien = image_ops::blend(&base, &mejorada, 100);
    assert_eq!(cien.to_rgb8().get_pixel(0, 0).0, [200, 200, 200]);

    let cero = image_ops::blend(&base, &mejorada, 0);
    assert_eq!(cero.to_rgb8().get_pixel(0, 0).0, [120, 90, 60]);

    let medio = image_ops::blend(&base, &mejorada, 50);
    let px = medio.to_rgb8().get_pixel(0, 0).0;
    assert!(
        (140..=180).contains(&px[0]),
        "valor intermedio esperado, obtenue {px:?}"
    );
}

#[test]
fn peso_objetivo_se_acerca_al_tamano_pedido() {
    // Imagen con mucho detalle para que el tamaño dependa de la calidad.
    let mut buffer = image::RgbImage::new(320, 320);
    for (x, y, pixel) in buffer.enumerate_pixels_mut() {
        *pixel = image::Rgb([(x % 256) as u8, (y % 256) as u8, ((x * y) % 256) as u8]);
    }
    let img = image::DynamicImage::ImageRgb8(buffer);

    let data = image_ops::encode(
        &img,
        OutputFormat::Jpg,
        Weight::TargetSize { target_kb: 40 },
    )
    .expect("la codificación debe funcionar");

    let kb = data.len() as f64 / 1024.0;
    assert!(
        (20.0..=90.0).contains(&kb),
        "el tamaño debe acercarse a 40 KB, obtuvo {kb:.1} KB"
    );
}

#[test]
fn calidad_fija_genera_archivo_menor_a_calidad_maxima() {
    let mut buffer = image::RgbImage::new(256, 256);
    for (x, y, pixel) in buffer.enumerate_pixels_mut() {
        *pixel = image::Rgb([(x % 256) as u8, (y % 256) as u8, ((x + y) % 256) as u8]);
    }
    let img = image::DynamicImage::ImageRgb8(buffer);

    let baja = image_ops::encode(&img, OutputFormat::Jpg, Weight::Quality { quality: 20 }).unwrap();
    let alta = image_ops::encode(&img, OutputFormat::Jpg, Weight::Quality { quality: 95 }).unwrap();
    assert!(
        baja.len() < alta.len(),
        "calidad 20 debe pesar menos que 95"
    );
}

#[test]
fn fps_se_formatea_sin_ceros_sobrantes() {
    assert_eq!(format_fps(25.0), "25");
    assert_eq!(format_fps(23.976), "23.976");
    assert_eq!(format_fps(29.97), "29.97");
}

#[test]
fn configuracion_va_y_vuelve_en_json() {
    let mut cfg = AppConfig::default();
    cfg.target_scale = 1.75;
    cfg.improve_percent = 80;
    cfg.weight = Weight::TargetSize { target_kb: 300 };

    let json = serde_json::to_string(&cfg).unwrap();
    let vuelta: AppConfig = serde_json::from_str(&json).unwrap();

    assert_eq!(vuelta.target_scale, 1.75);
    assert_eq!(vuelta.improve_percent, 80);
    assert_eq!(vuelta.weight, Weight::TargetSize { target_kb: 300 });
}

#[test]
fn validacion_ajusta_valores_fuera_de_rango() {
    let mut cfg = AppConfig {
        engine_scale: 9,
        target_scale: 99.0,
        improve_percent: 250,
        sharpen: 200,
        max_workers: 0,
        ..Default::default()
    };
    cfg.validar();
    assert_eq!(cfg.engine_scale, 4);
    assert_eq!(cfg.target_scale, 8.0);
    assert_eq!(cfg.improve_percent, 100);
    assert_eq!(cfg.sharpen, 100);
    assert_eq!(cfg.max_workers, 1);
}

#[test]
fn nitidez_cambia_la_imagen() {
    let img = imagen(16, 16);
    let nitida = image_ops::sharpen(&img, 50);
    assert_eq!(nitida.to_rgb8().dimensions(), (16, 16));
}
