//! Prueba de concepto: restauración de rostros con GFPGAN **sin Python ni torch**.
//!
//! Replica el pipeline de `facexlib` + `GFPGANer` (el que usa hoy el puente):
//!   1. RetinaFace (ONNX)  -> caras + 5 puntos
//!   2. alineación afín a plantilla 512 (similitud por mínimos cuadrados)
//!   3. GFPGAN v1.4 (ONNX) -> cara restaurada
//!   4. ParseNet (ONNX)    -> máscara de la cara, suavizada
//!   5. pegado sobre la imagen ampliada con la afín inversa
//!
//! Uso: onnx-faces --input foto.jpg --output salida.png [--cpu] [--upscale 2]
//!                 [--models ..\models] [--dll ..\bin\onnxruntime.dll]

use anyhow::{anyhow, Context, Result};
use image::{imageops::FilterType, RgbImage};
use ort::{
    execution_providers::{CPUExecutionProvider, DirectMLExecutionProvider},
    session::Session,
    value::Tensor,
};
use std::path::{Path, PathBuf};
use std::time::Instant;

// ------------------------------------------------------------------ imagen f32
/// Imagen RGB en f32 (0..255), canales intercalados.
#[derive(Clone)]
struct Img {
    w: usize,
    h: usize,
    d: Vec<f32>,
}

impl Img {
    fn new(w: usize, h: usize) -> Self {
        Img { w, h, d: vec![0.0; w * h * 3] }
    }
    fn from_rgb8(i: &RgbImage) -> Self {
        let (w, h) = (i.width() as usize, i.height() as usize);
        Img { w, h, d: i.as_raw().iter().map(|&v| v as f32).collect() }
    }
    fn to_rgb8(&self) -> RgbImage {
        let raw: Vec<u8> = self.d.iter().map(|&v| v.round().clamp(0.0, 255.0) as u8).collect();
        RgbImage::from_raw(self.w as u32, self.h as u32, raw).unwrap()
    }
    /// Muestreo bilinear; fuera de la imagen devuelve `border`.
    fn sample(&self, x: f32, y: f32, border: [f32; 3]) -> [f32; 3] {
        if x < -0.5 || y < -0.5 || x > self.w as f32 - 0.5 || y > self.h as f32 - 0.5 {
            return border;
        }
        let x = x.clamp(0.0, self.w as f32 - 1.0);
        let y = y.clamp(0.0, self.h as f32 - 1.0);
        let (x0, y0) = (x.floor() as usize, y.floor() as usize);
        let (x1, y1) = ((x0 + 1).min(self.w - 1), (y0 + 1).min(self.h - 1));
        let (fx, fy) = (x - x0 as f32, y - y0 as f32);
        let mut out = [0.0; 3];
        for c in 0..3 {
            let p = |xx: usize, yy: usize| self.d[(yy * self.w + xx) * 3 + c];
            let top = p(x0, y0) * (1.0 - fx) + p(x1, y0) * fx;
            let bot = p(x0, y1) * (1.0 - fx) + p(x1, y1) * fx;
            out[c] = top * (1.0 - fy) + bot * fy;
        }
        out
    }
}

fn resize(img: &Img, w: usize, h: usize, filter: FilterType) -> Img {
    let r = image::imageops::resize(&img.to_rgb8(), w as u32, h as u32, filter);
    Img::from_rgb8(&r)
}

// ---------------------------------------------------------------- geometría 2x3
type Affine = [f32; 6]; // [a b tx; c d ty]

fn apply(m: &Affine, x: f32, y: f32) -> (f32, f32) {
    (m[0] * x + m[1] * y + m[2], m[3] * x + m[4] * y + m[5])
}

fn invert(m: &Affine) -> Affine {
    let det = m[0] * m[4] - m[1] * m[3];
    let (a, b, c, d) = (m[4] / det, -m[1] / det, -m[3] / det, m[0] / det);
    [a, b, -(a * m[2] + b * m[5]), c, d, -(c * m[2] + d * m[5])]
}

/// Transformación de similitud (4 grados de libertad) de `src` a `dst` por mínimos cuadrados.
fn estimate_similarity(src: &[(f32, f32)], dst: &[(f32, f32)]) -> Affine {
    let n = src.len() as f32;
    let (mpx, mpy) = (src.iter().map(|p| p.0).sum::<f32>() / n, src.iter().map(|p| p.1).sum::<f32>() / n);
    let (mqx, mqy) = (dst.iter().map(|p| p.0).sum::<f32>() / n, dst.iter().map(|p| p.1).sum::<f32>() / n);
    // Con números complejos: s = Σ conj(p)·q / Σ |p|²
    let (mut re, mut im, mut den) = (0.0f32, 0.0f32, 0.0f32);
    for (p, q) in src.iter().zip(dst) {
        let (px, py, qx, qy) = (p.0 - mpx, p.1 - mpy, q.0 - mqx, q.1 - mqy);
        re += px * qx + py * qy;
        im += px * qy - py * qx;
        den += px * px + py * py;
    }
    let (a, b) = (re / den, im / den);
    [a, -b, mqx - (a * mpx - b * mpy), b, a, mqy - (b * mpx + a * mpy)]
}

/// `cv2.warpAffine` (interpolación bilinear): `m` lleva coordenadas de origen a destino.
fn warp_affine(src: &Img, m: &Affine, w: usize, h: usize, border: [f32; 3]) -> Img {
    let inv = invert(m);
    let mut out = Img::new(w, h);
    for y in 0..h {
        for x in 0..w {
            let (sx, sy) = apply(&inv, x as f32, y as f32);
            let p = src.sample(sx, sy, border);
            let o = (y * w + x) * 3;
            out.d[o..o + 3].copy_from_slice(&p);
        }
    }
    out
}

/// Igual que `warp_affine` pero para una máscara de un canal.
fn warp_mask(src: &[f32], sw: usize, sh: usize, m: &Affine, w: usize, h: usize) -> Vec<f32> {
    let inv = invert(m);
    let mut out = vec![0.0f32; w * h];
    for y in 0..h {
        for x in 0..w {
            let (fx, fy) = apply(&inv, x as f32, y as f32);
            if fx < -0.5 || fy < -0.5 || fx > sw as f32 - 0.5 || fy > sh as f32 - 0.5 {
                continue;
            }
            let fx = fx.clamp(0.0, sw as f32 - 1.0);
            let fy = fy.clamp(0.0, sh as f32 - 1.0);
            let (x0, y0) = (fx.floor() as usize, fy.floor() as usize);
            let (x1, y1) = ((x0 + 1).min(sw - 1), (y0 + 1).min(sh - 1));
            let (ax, ay) = (fx - x0 as f32, fy - y0 as f32);
            let g = |xx: usize, yy: usize| src[yy * sw + xx];
            out[y * w + x] = (g(x0, y0) * (1.0 - ax) + g(x1, y0) * ax) * (1.0 - ay)
                + (g(x0, y1) * (1.0 - ax) + g(x1, y1) * ax) * ay;
        }
    }
    out
}

/// `cv2.GaussianBlur(mask, (k, k), sigma)` separable, borde reflejado (reflect101).
fn gaussian_blur(src: &[f32], w: usize, h: usize, k: usize, sigma: f32) -> Vec<f32> {
    let r = (k / 2) as i32;
    let mut ker: Vec<f32> = (-r..=r).map(|i| (-(i * i) as f32 / (2.0 * sigma * sigma)).exp()).collect();
    let s: f32 = ker.iter().sum();
    ker.iter_mut().for_each(|v| *v /= s);
    let refl = |i: i32, n: i32| -> usize {
        let mut i = i;
        if n == 1 {
            return 0;
        }
        while i < 0 || i >= n {
            if i < 0 {
                i = -i;
            }
            if i >= n {
                i = 2 * (n - 1) - i;
            }
        }
        i as usize
    };
    let mut tmp = vec![0.0f32; w * h];
    for y in 0..h {
        for x in 0..w {
            let mut acc = 0.0;
            for (j, kv) in ker.iter().enumerate() {
                acc += kv * src[y * w + refl(x as i32 + j as i32 - r, w as i32)];
            }
            tmp[y * w + x] = acc;
        }
    }
    let mut out = vec![0.0f32; w * h];
    for y in 0..h {
        for x in 0..w {
            let mut acc = 0.0;
            for (j, kv) in ker.iter().enumerate() {
                acc += kv * tmp[refl(y as i32 + j as i32 - r, h as i32) * w + x];
            }
            out[y * w + x] = acc;
        }
    }
    out
}

// ------------------------------------------------------------------- ONNX utils
struct Net {
    s: Session,
    on_gpu: bool,
}

fn load(path: &Path, gpu: bool) -> Result<Net> {
    let mut b = Session::builder()?.with_memory_pattern(false)?.with_parallel_execution(false)?;
    if gpu {
        b = b.with_execution_providers([DirectMLExecutionProvider::default().build().error_on_failure()])?;
    } else {
        b = b.with_execution_providers([CPUExecutionProvider::default().build()])?;
    }
    let s = b.commit_from_file(path).with_context(|| format!("cargando {}", path.display()))?;
    Ok(Net { s, on_gpu: gpu })
}

/// Ejecuta una red de una entrada NCHW y devuelve todas sus salidas como (forma, datos).
fn run(net: &mut Net, input: Vec<f32>, shape: [usize; 4]) -> Result<Vec<(Vec<i64>, Vec<f32>)>> {
    let t = Tensor::from_array((shape, input))?;
    let outs = net.s.run(ort::inputs![t])?;
    let mut res = Vec::new();
    for (_, v) in outs.iter() {
        let (sh, d) = v.try_extract_tensor::<f32>()?;
        res.push((sh.to_vec(), d.to_vec()));
    }
    Ok(res)
}

// ------------------------------------------------------------------ detección
struct Face {
    lm: [(f32, f32); 5],
}

fn detect(net: &mut Net, img: &Img, conf_thr: f32) -> Result<Vec<Face>> {
    let (w, h) = (img.w, img.h);
    // BGR menos la media (104, 117, 123), formato CHW
    let mut x = vec![0.0f32; 3 * w * h];
    for i in 0..w * h {
        let (r, g, b) = (img.d[i * 3], img.d[i * 3 + 1], img.d[i * 3 + 2]);
        x[i] = b - 104.0;
        x[w * h + i] = g - 117.0;
        x[2 * w * h + i] = r - 123.0;
    }
    let out = run(net, x, [1, 3, h, w])?;
    let (loc, conf, ldm) = (&out[0].1, &out[1].1, &out[2].1);

    // anclas (PriorBox de RetinaFace-ResNet50)
    let steps = [8.0f32, 16.0, 32.0];
    let min_sizes: [&[f32]; 3] = [&[16.0, 32.0], &[64.0, 128.0], &[256.0, 512.0]];
    let mut priors: Vec<[f32; 4]> = Vec::new();
    for (k, &step) in steps.iter().enumerate() {
        let (fh, fw) = ((h as f32 / step).ceil() as usize, (w as f32 / step).ceil() as usize);
        for i in 0..fh {
            for j in 0..fw {
                for &ms in min_sizes[k] {
                    priors.push([
                        (j as f32 + 0.5) * step / w as f32,
                        (i as f32 + 0.5) * step / h as f32,
                        ms / w as f32,
                        ms / h as f32,
                    ]);
                }
            }
        }
    }
    if priors.len() * 4 != loc.len() {
        return Err(anyhow!("anclas ({}) y salida del detector ({}) no coinciden", priors.len(), loc.len() / 4));
    }

    struct Det {
        b: [f32; 4],
        score: f32,
        lm: [(f32, f32); 5],
    }
    let mut dets: Vec<Det> = Vec::new();
    for (n, p) in priors.iter().enumerate() {
        // conf ya viene con softmax: columna 1 = cara
        let score = conf[n * 2 + 1];
        if score <= conf_thr {
            continue;
        }
        let l = &loc[n * 4..n * 4 + 4];
        let (cx, cy) = (p[0] + l[0] * 0.1 * p[2], p[1] + l[1] * 0.1 * p[3]);
        let (bw, bh) = (p[2] * (l[2] * 0.2).exp(), p[3] * (l[3] * 0.2).exp());
        let b = [(cx - bw / 2.0) * w as f32, (cy - bh / 2.0) * h as f32, (cx + bw / 2.0) * w as f32, (cy + bh / 2.0) * h as f32];
        let mut lm = [(0.0, 0.0); 5];
        for (i, pt) in lm.iter_mut().enumerate() {
            let o = n * 10 + i * 2;
            *pt = ((p[0] + ldm[o] * 0.1 * p[2]) * w as f32, (p[1] + ldm[o + 1] * 0.1 * p[3]) * h as f32);
        }
        dets.push(Det { b, score, lm });
    }
    dets.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());

    // NMS (py_cpu_nms de facexlib: IoU con +1 píxel, umbral 0.4)
    let mut keep: Vec<Det> = Vec::new();
    for d in dets {
        let area = |b: &[f32; 4]| (b[2] - b[0] + 1.0) * (b[3] - b[1] + 1.0);
        let overlaps = keep.iter().any(|k| {
            let (x1, y1, x2, y2) = (d.b[0].max(k.b[0]), d.b[1].max(k.b[1]), d.b[2].min(k.b[2]), d.b[3].min(k.b[3]));
            let inter = (x2 - x1 + 1.0).max(0.0) * (y2 - y1 + 1.0).max(0.0);
            inter / (area(&d.b) + area(&k.b) - inter) > 0.4
        });
        if !overlaps {
            keep.push(d);
        }
    }
    // descarta caras diminutas (distancia entre ojos <= 5 px), igual que GFPGANer
    Ok(keep
        .into_iter()
        .filter(|d| ((d.lm[0].0 - d.lm[1].0).powi(2) + (d.lm[0].1 - d.lm[1].1).powi(2)).sqrt() > 5.0)
        .map(|d| Face { lm: d.lm })
        .collect())
}

// ------------------------------------------------------------------- pipeline
const TEMPLATE: [(f32, f32); 5] = [
    (192.98138, 239.94708),
    (318.90277, 240.1936),
    (256.63416, 314.01935),
    (201.26117, 371.41043),
    (313.08905, 371.15118),
];

fn to_nchw_norm(img: &Img) -> Vec<f32> {
    let n = img.w * img.h;
    let mut x = vec![0.0f32; 3 * n];
    for i in 0..n {
        for c in 0..3 {
            x[c * n + i] = (img.d[i * 3 + c] / 255.0 - 0.5) / 0.5;
        }
    }
    x
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let get = |k: &str| args.iter().position(|a| a == k).and_then(|i| args.get(i + 1)).cloned();
    let input = PathBuf::from(get("--input").ok_or_else(|| anyhow!("falta --input"))?);
    let output = PathBuf::from(get("--output").ok_or_else(|| anyhow!("falta --output"))?);
    let models = PathBuf::from(get("--models").unwrap_or_else(|| "../models".into()));
    let dll = get("--dll").unwrap_or_else(|| "../bin/onnxruntime.dll".into());
    let upscale: usize = get("--upscale").and_then(|v| v.parse().ok()).unwrap_or(2);
    let gpu = !args.iter().any(|a| a == "--cpu");

    let t_all = Instant::now();
    ort::init_from(&dll).commit()?;

    let t = Instant::now();
    let mut det = load(&models.join("retinaface_r50.onnx"), gpu)?;
    let model_name = get("--model").unwrap_or_else(|| "gfpgan".into());
    let model_file = match model_name.as_str() {
        "gfpgan" => "gfpgan_v14.onnx",
        "restoreformer" => "restoreformer_pp.onnx",
        other => return Err(anyhow!("modelo desconocido: {other} (gfpgan | restoreformer)")),
    };
    let mut gfp = load(&models.join(model_file), gpu)?;
    let mut par = load(&models.join("parsenet.onnx"), gpu)?;
    println!("modelos cargados ({}) en {:.2}s", if det.on_gpu { "DirectML/GPU" } else { "CPU" }, t.elapsed().as_secs_f32());

    let base = Img::from_rgb8(&image::open(&input)?.to_rgb8());
    println!("imagen: {}x{}", base.w, base.h);

    let t = Instant::now();
    let faces = detect(&mut det, &base, 0.97)?;
    println!("caras detectadas: {} ({:.2}s)", faces.len(), t.elapsed().as_secs_f32());
    if faces.is_empty() {
        return Err(anyhow!("No se encontraron rostros."));
    }

    let (wu, hu) = (base.w * upscale, base.h * upscale);
    let mut canvas = resize(&base, wu, hu, FilterType::Lanczos3);

    for (idx, f) in faces.iter().enumerate() {
        let t = Instant::now();
        let aff = estimate_similarity(&f.lm, &TEMPLATE);
        let crop = warp_affine(&base, &aff, 512, 512, [132.0, 133.0, 135.0]);

        let out = run(&mut gfp, to_nchw_norm(&crop), [1, 3, 512, 512])?;
        let o = &out[0].1;
        let mut restored = Img::new(512, 512);
        for i in 0..512 * 512 {
            for c in 0..3 {
                let v = o[c * 512 * 512 + i].clamp(-1.0, 1.0);
                restored.d[i * 3 + c] = ((v + 1.0) / 2.0 * 255.0).round();
            }
        }

        // máscara del rostro con ParseNet (facexlib: use_parse=True)
        let lab = run(&mut par, to_nchw_norm(&restored), [1, 3, 512, 512])?;
        let (cls, data) = (lab[0].0[1] as usize, &lab[0].1);
        let keep_label = [0u8, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 0, 255, 0, 0, 0];
        let mut mask = vec![0.0f32; 512 * 512];
        for i in 0..512 * 512 {
            let (mut best, mut bi) = (f32::MIN, 0usize);
            for c in 0..cls {
                let v = data[c * 512 * 512 + i];
                if v > best {
                    best = v;
                    bi = c;
                }
            }
            mask[i] = keep_label[bi.min(18)] as f32;
        }
        let mut mask = gaussian_blur(&mask, 512, 512, 101, 11.0);
        let mask2 = gaussian_blur(&mask, 512, 512, 101, 11.0);
        mask.copy_from_slice(&mask2);
        let th = 10;
        for y in 0..512 {
            for x in 0..512 {
                if y < th || y >= 512 - th || x < th || x >= 512 - th {
                    mask[y * 512 + x] = 0.0;
                }
                mask[y * 512 + x] /= 255.0;
            }
        }

        // pegado: afín inversa escalada al tamaño ampliado
        let mut inv = invert(&aff);
        inv.iter_mut().for_each(|v| *v *= upscale as f32);
        let inv_restored = warp_affine(&restored, &inv, wu, hu, [0.0, 0.0, 0.0]);
        let inv_mask = warp_mask(&mask, 512, 512, &inv, wu, hu);
        for i in 0..wu * hu {
            let a = inv_mask[i];
            for c in 0..3 {
                canvas.d[i * 3 + c] = a * inv_restored.d[i * 3 + c] + (1.0 - a) * canvas.d[i * 3 + c];
            }
        }
        println!("  cara {}: restaurada y pegada en {:.2}s", idx + 1, t.elapsed().as_secs_f32());
    }

    canvas.to_rgb8().save(&output)?;
    println!("guardado: {} ({}x{}) · total {:.2}s", output.display(), wu, hu, t_all.elapsed().as_secs_f32());
    Ok(())
}
