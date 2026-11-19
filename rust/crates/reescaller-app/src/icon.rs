//! Icono de la aplicación generado en tiempo de ejecución con los colores del tema.
//!
//! Es el mismo símbolo del logo (un píxel sólido y dos esquinas que se expanden) dibujado sobre un degradado
//! diagonal entre los dos colores principales del tema. Se renderiza con suavizado por supermuestreo.

use tao::window::Icon;

/// Como `palette_colors`, pero entiende el tema personalizado (`custom-dark` / `custom-light`) con sus colores guardados.
pub fn palette_colors_cfg(theme: &str, custom_json: &str) -> ([u8; 3], [u8; 3]) {
    if theme.to_lowercase().contains("custom") {
        let modo = if theme.to_lowercase().contains("light") {
            "light"
        } else {
            "dark"
        };
        if let Ok(v) = serde_json::from_str::<serde_json::Value>(custom_json) {
            let hex = |k: &str| -> Option<[u8; 3]> {
                let h = v
                    .get(modo)?
                    .get(k)?
                    .as_str()?
                    .trim_start_matches('#')
                    .to_string();
                if h.len() != 6 {
                    return None;
                }
                Some([
                    u8::from_str_radix(&h[0..2], 16).ok()?,
                    u8::from_str_radix(&h[2..4], 16).ok()?,
                    u8::from_str_radix(&h[4..6], 16).ok()?,
                ])
            };
            if let (Some(a), Some(b)) = (hex("p"), hex("s")) {
                return (a, b);
            }
        }
    }
    palette_colors(theme)
}

/// Colores (principal, secundario) de cada paleta; se usan al arrancar, antes de que cargue la interfaz.
pub fn palette_colors(theme: &str) -> ([u8; 3], [u8; 3]) {
    let t = theme.to_lowercase();
    let claro = t.contains("light");
    // (oscuro, claro): mismos valores `--tint` / `--tint-2` que usa la interfaz.
    let (oscuro, luz) = if t.contains("miami") {
        (
            ([201, 52, 155], [255, 155, 85]),
            ([226, 38, 124], [0, 160, 172]),
        )
    } else if t.contains("carmesi") {
        (
            ([228, 28, 76], [255, 92, 112]),
            ([200, 18, 62], [240, 70, 92]),
        )
    } else if t.contains("eva") {
        (
            ([138, 64, 224], [112, 255, 72]),
            ([98, 40, 168], [64, 150, 10]),
        )
    } else if t.contains("esmeralda") {
        (
            ([24, 150, 28], [120, 200, 20]),
            ([14, 122, 14], [58, 128, 0]),
        )
    } else if t.contains("zafiro") {
        (
            ([0, 112, 209], [40, 170, 255]),
            ([0, 85, 190], [0, 120, 205]),
        )
    } else if t.contains("magma") {
        (
            ([214, 40, 20], [255, 140, 20]),
            ([196, 36, 12], [186, 78, 0]),
        )
    } else if t.contains("distrito") {
        (
            ([255, 34, 74], [28, 222, 238]),
            ([214, 0, 58], [0, 136, 158]),
        )
    } else {
        (
            ([139, 92, 246], [34, 211, 238]),
            ([108, 66, 228], [6, 134, 170]),
        )
    };
    if claro {
        luz
    } else {
        oscuro
    }
}

fn dist_segment(px: f32, py: f32, a: (f32, f32), b: (f32, f32)) -> f32 {
    let (vx, vy) = (b.0 - a.0, b.1 - a.1);
    let len2 = vx * vx + vy * vy;
    let t = if len2 == 0.0 {
        0.0
    } else {
        (((px - a.0) * vx + (py - a.1) * vy) / len2).clamp(0.0, 1.0)
    };
    let (cx, cy) = (a.0 + t * vx, a.1 + t * vy);
    ((px - cx).powi(2) + (py - cy).powi(2)).sqrt()
}

/// Distancia con signo a un rectángulo redondeado (<= 0 dentro).
fn sdf_round_rect(px: f32, py: f32, x0: f32, y0: f32, x1: f32, y1: f32, r: f32) -> f32 {
    let (cx, cy) = ((x0 + x1) / 2.0, (y0 + y1) / 2.0);
    let (hx, hy) = ((x1 - x0) / 2.0 - r, (y1 - y0) / 2.0 - r);
    let (dx, dy) = ((px - cx).abs() - hx, (py - cy).abs() - hy);
    (dx.max(0.0).powi(2) + dy.max(0.0).powi(2)).sqrt() + dx.max(dy).min(0.0) - r
}

/// RGBA (n×n) del icono con degradado `c1` → `c2`.
pub fn render(c1: [u8; 3], c2: [u8; 3], n: u32) -> Vec<u8> {
    let ss = 3u32;
    let scale = 32.0 / n as f32;
    let mut out = vec![0u8; (n * n * 4) as usize];
    for y in 0..n {
        for x in 0..n {
            let (mut r, mut g, mut b, mut a) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
            for sy in 0..ss {
                for sx in 0..ss {
                    let u = (x as f32 + (sx as f32 + 0.5) / ss as f32) * scale;
                    let v = (y as f32 + (sy as f32 + 0.5) / ss as f32) * scale;
                    if sdf_round_rect(u, v, 0.0, 0.0, 32.0, 32.0, 9.0) > 0.0 {
                        continue;
                    }
                    let t = ((u + v) / 64.0).clamp(0.0, 1.0);
                    let mut col = [0.0f32; 3];
                    for (i, c) in col.iter_mut().enumerate() {
                        *c = c1[i] as f32 * (1.0 - t) + c2[i] as f32 * t;
                    }
                    // píxel sólido
                    let mut white = 0.0f32;
                    if sdf_round_rect(u, v, 8.0, 18.0, 14.0, 24.0, 1.2) <= 0.0 {
                        white = 1.0;
                    }
                    // esquinas concéntricas (trazo de 2 con extremos redondos)
                    let d1 = dist_segment(u, v, (15.0, 12.0), (21.0, 12.0)).min(dist_segment(
                        u,
                        v,
                        (21.0, 12.0),
                        (21.0, 18.0),
                    ));
                    if d1 <= 1.0 {
                        white = white.max(1.0);
                    }
                    let d2 = dist_segment(u, v, (15.0, 6.0), (26.0, 6.0)).min(dist_segment(
                        u,
                        v,
                        (26.0, 6.0),
                        (26.0, 17.0),
                    ));
                    if d2 <= 1.0 {
                        white = white.max(0.6);
                    }
                    for c in col.iter_mut() {
                        *c = *c * (1.0 - white) + 255.0 * white;
                    }
                    r += col[0];
                    g += col[1];
                    b += col[2];
                    a += 1.0;
                }
            }
            let i = ((y * n + x) * 4) as usize;
            let total = (ss * ss) as f32;
            if a > 0.0 {
                out[i] = (r / a).round() as u8;
                out[i + 1] = (g / a).round() as u8;
                out[i + 2] = (b / a).round() as u8;
                out[i + 3] = (a / total * 255.0).round() as u8;
            }
        }
    }
    out
}

/// Icono listo para `tao`.
pub fn tao_icon(c1: [u8; 3], c2: [u8; 3]) -> Option<Icon> {
    Icon::from_rgba(render(c1, c2, 128), 128, 128).ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn el_icono_usa_los_colores_del_tema() {
        let img = render([255, 0, 0], [0, 0, 255], 64);
        // esquina superior izquierda (fuera del redondeo) transparente
        assert_eq!(img[3], 0);
        // zona roja arriba a la izquierda dentro del icono
        let i = ((10 * 64 + 10) * 4) as usize;
        assert!(
            img[i] > img[i + 2],
            "arriba-izquierda debe parecerse al color principal"
        );
        // abajo a la derecha se parece al secundario
        let j = ((58 * 64 + 50) * 4) as usize;
        assert!(
            img[j + 2] > img[j],
            "abajo-derecha debe parecerse al color secundario"
        );
    }

    #[test]
    fn tema_personalizado() {
        let j = r##"{"dark":{"p":"#112233","s":"#445566"},"light":{"p":"#aabbcc","s":"#ddeeff"}}"##;
        assert_eq!(
            palette_colors_cfg("custom-dark", j),
            ([0x11, 0x22, 0x33], [0x44, 0x55, 0x66])
        );
        assert_eq!(
            palette_colors_cfg("custom-light", j),
            ([0xaa, 0xbb, 0xcc], [0xdd, 0xee, 0xff])
        );
        assert_eq!(
            palette_colors_cfg("custom-dark", "no es json").0,
            [139, 92, 246]
        );
    }

    #[test]
    fn paleta_conocida_y_desconocida() {
        assert_eq!(palette_colors("miami-dark").0, [201, 52, 155]);
        assert_eq!(palette_colors("miami-light").1, [0, 160, 172]);
        assert_eq!(palette_colors("carmesi-dark").0, [228, 28, 76]);
        assert_eq!(palette_colors("eva-light").1, [64, 150, 10]);
        assert_eq!(palette_colors("cosa-rara").0, [139, 92, 246]);
    }
}
