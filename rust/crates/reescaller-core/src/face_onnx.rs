//! Restauración de rostros **nativa**: ONNX Runtime (DirectML o CPU), sin Python ni torch.
//!
//! Réplica del pipeline de `facexlib` + `GFPGANer` / `RestoreFormer`:
//!   1. RetinaFace (ONNX)  → caras + 5 puntos
//!   2. alineación afín a la plantilla de 512 px
//!   3. GFPGAN v1.4 / RestoreFormer++ (ONNX)
//!   4. ParseNet (ONNX)    → máscara de la cara
//!   5. pegado sobre la imagen ampliada con la afín inversa
//!
//! Requisitos en disco (carpeta `models/onnx`): `onnxruntime.dll`, `DirectML.dll`, `retinaface_r50.onnx`,
//! `parsenet.onnx` y al menos uno de `gfpgan_v14.onnx` / `restoreformer_pp.onnx`.
//!
//! Robustez:
//! - GPU → CPU: si DirectML no está disponible o falla en ejecución, se continúa en CPU.
//! - Fotos grandes: la detección se hace sobre una copia reducida (≤ 1280 px de lado largo) y el pegado solo toca
//!   la zona de cada cara; la ampliación se limita para que la salida no supere 120 MP.
//! - Se respeta la orientación EXIF y se aceptan escala de grises, alfa y 16 bits.

use crate::error::{Error, Result};
use crate::face::FaceModel;
use image::{imageops::FilterType, DynamicImage, ImageDecoder, ImageReader, RgbImage};
use ort::{
    execution_providers::{CPUExecutionProvider, DirectMLExecutionProvider},
    session::Session,
    value::Tensor,
};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

const DET_FILE: &str = "retinaface_r50.onnx";
const PARSE_FILE: &str = "parsenet.onnx";
const DLL_FILE: &str = "onnxruntime.dll";
/// Lado largo máximo de la copia usada para detectar caras.
const DET_MAX_SIDE: usize = 1280;
/// Píxeles máximos de la imagen resultante.
const MAX_OUTPUT_PIXELS: u64 = 120_000_000;

type Log<'a> = &'a (dyn Fn(&str) + Send + Sync);

fn msg(e: impl std::fmt::Display) -> Error {
    Error::Message(e.to_string())
}

// ------------------------------------------------------------ localización
fn is_valid_dir(dir: &Path) -> bool {
    dir.join(DLL_FILE).is_file()
        && dir.join(DET_FILE).is_file()
        && dir.join(PARSE_FILE).is_file()
        && (dir.join(FaceModel::Gfpgan.onnx_file()).is_file()
            || dir.join(FaceModel::RestoreFormer.onnx_file()).is_file())
}

/// Busca la carpeta con el runtime y los modelos ONNX (junto a la app, carpetas superiores o datos del usuario).
pub fn find_onnx_dir(base_dir: &Path) -> Option<PathBuf> {
    let mut bases: Vec<PathBuf> = vec![base_dir.to_path_buf()];
    let mut anc = base_dir.parent();
    for _ in 0..3 {
        if let Some(a) = anc {
            bases.push(a.to_path_buf());
            anc = a.parent();
        }
    }
    for b in &bases {
        for rel in ["models/onnx", "onnx"] {
            let d = b.join(rel);
            if is_valid_dir(&d) {
                return Some(d);
            }
        }
    }
    if let Some(d) = dirs::data_local_dir() {
        for rel in ["Reescaller/components/onnx", "Reescaller/models/onnx"] {
            let c = d.join(rel);
            if is_valid_dir(&c) {
                return Some(c);
            }
        }
    }
    None
}

/// ¿Esa carpeta sirve para el modelo pedido?
pub fn supports(dir: &Path, model: FaceModel) -> bool {
    is_valid_dir(dir) && dir.join(model.onnx_file()).is_file()
}

/// Modelos de rostros disponibles en la carpeta.
pub fn available_models(dir: &Path) -> Vec<FaceModel> {
    [FaceModel::Gfpgan, FaceModel::RestoreFormer]
        .into_iter()
        .filter(|m| supports(dir, *m))
        .collect()
}

// ----------------------------------------------------------------- imagen f32
#[derive(Clone)]
struct Img {
    w: usize,
    h: usize,
    d: Vec<f32>,
}

impl Img {
    fn new(w: usize, h: usize) -> Self {
        Img {
            w,
            h,
            d: vec![0.0; w * h * 3],
        }
    }
    fn from_rgb8(i: &RgbImage) -> Self {
        Img {
            w: i.width() as usize,
            h: i.height() as usize,
            d: i.as_raw().iter().map(|&v| v as f32).collect(),
        }
    }
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
        for (c, o) in out.iter_mut().enumerate() {
            let p = |xx: usize, yy: usize| self.d[(yy * self.w + xx) * 3 + c];
            let top = p(x0, y0) * (1.0 - fx) + p(x1, y0) * fx;
            let bot = p(x0, y1) * (1.0 - fx) + p(x1, y1) * fx;
            *o = top * (1.0 - fy) + bot * fy;
        }
        out
    }
}

fn sample_rgb(img: &RgbImage, x: f32, y: f32, border: [f32; 3]) -> [f32; 3] {
    let (w, h) = (img.width() as usize, img.height() as usize);
    if x < -0.5 || y < -0.5 || x > w as f32 - 0.5 || y > h as f32 - 0.5 {
        return border;
    }
    let x = x.clamp(0.0, w as f32 - 1.0);
    let y = y.clamp(0.0, h as f32 - 1.0);
    let (x0, y0) = (x.floor() as usize, y.floor() as usize);
    let (x1, y1) = ((x0 + 1).min(w - 1), (y0 + 1).min(h - 1));
    let (fx, fy) = (x - x0 as f32, y - y0 as f32);
    let raw = img.as_raw();
    let mut out = [0.0; 3];
    for (c, o) in out.iter_mut().enumerate() {
        let p = |xx: usize, yy: usize| raw[(yy * w + xx) * 3 + c] as f32;
        let top = p(x0, y0) * (1.0 - fx) + p(x1, y0) * fx;
        let bot = p(x0, y1) * (1.0 - fx) + p(x1, y1) * fx;
        *o = top * (1.0 - fy) + bot * fy;
    }
    out
}

// ------------------------------------------------------------ geometría 2×3
type Affine = [f32; 6];

fn apply(m: &Affine, x: f32, y: f32) -> (f32, f32) {
    (m[0] * x + m[1] * y + m[2], m[3] * x + m[4] * y + m[5])
}

fn invert(m: &Affine) -> Affine {
    let det = m[0] * m[4] - m[1] * m[3];
    let (a, b, c, d) = (m[4] / det, -m[1] / det, -m[3] / det, m[0] / det);
    [a, b, -(a * m[2] + b * m[5]), c, d, -(c * m[2] + d * m[5])]
}

/// Similitud (4 grados de libertad) de `src` a `dst` por mínimos cuadrados.
fn estimate_similarity(src: &[(f32, f32)], dst: &[(f32, f32)]) -> Affine {
    let n = src.len() as f32;
    let (mpx, mpy) = (
        src.iter().map(|p| p.0).sum::<f32>() / n,
        src.iter().map(|p| p.1).sum::<f32>() / n,
    );
    let (mqx, mqy) = (
        dst.iter().map(|p| p.0).sum::<f32>() / n,
        dst.iter().map(|p| p.1).sum::<f32>() / n,
    );
    let (mut re, mut im, mut den) = (0.0f32, 0.0f32, 0.0f32);
    for (p, q) in src.iter().zip(dst) {
        let (px, py, qx, qy) = (p.0 - mpx, p.1 - mpy, q.0 - mqx, q.1 - mqy);
        re += px * qx + py * qy;
        im += px * qy - py * qx;
        den += px * px + py * py;
    }
    let den = den.max(1e-6);
    let (a, b) = (re / den, im / den);
    [
        a,
        -b,
        mqx - (a * mpx - b * mpy),
        b,
        a,
        mqy - (b * mpx + a * mpy),
    ]
}

/// Recorte alineado de 512×512 muestreando directamente la imagen original (sin copiarla a f32).
fn warp_to_crop(base: &RgbImage, m: &Affine, border: [f32; 3]) -> Img {
    let inv = invert(m);
    let mut out = Img::new(512, 512);
    for y in 0..512 {
        for x in 0..512 {
            let (sx, sy) = apply(&inv, x as f32, y as f32);
            let p = sample_rgb(base, sx, sy, border);
            let o = (y * 512 + x) * 3;
            out.d[o..o + 3].copy_from_slice(&p);
        }
    }
    out
}

fn gaussian_blur(src: &[f32], w: usize, h: usize, k: usize, sigma: f32) -> Vec<f32> {
    let r = (k / 2) as i32;
    let mut ker: Vec<f32> = (-r..=r)
        .map(|i| (-(i * i) as f32 / (2.0 * sigma * sigma)).exp())
        .collect();
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

// ------------------------------------------------------------------ ONNX
struct Net {
    s: Session,
}

struct Engine {
    dir: PathBuf,
    want_gpu: bool,
    gpu: bool,
    det: Net,
    parse: Net,
    restorers: HashMap<&'static str, Net>,
}

static ENGINE: Mutex<Option<Engine>> = Mutex::new(None);
static RUNTIME: OnceLock<std::result::Result<(), String>> = OnceLock::new();
/// Tras un fallo de ejecución en GPU no se vuelve a intentar durante la sesión.
static GPU_BROKEN: AtomicBool = AtomicBool::new(false);
/// Fuerza de la restauración (1.0 = rostro totalmente restaurado, 0.0 = rostro original), como bits de `f32`.
static STRENGTH: AtomicU32 = AtomicU32::new(0x3f80_0000);

/// Fija la fuerza de la restauración de rostros (0.0..=1.0) para las siguientes llamadas.
pub fn set_strength(v: f32) {
    STRENGTH.store(v.clamp(0.0, 1.0).to_bits(), Ordering::SeqCst);
}

#[cfg(windows)]
fn set_dll_directory(dir: &Path) {
    use std::os::windows::ffi::OsStrExt;
    #[link(name = "kernel32")]
    extern "system" {
        fn SetDllDirectoryW(path: *const u16) -> i32;
    }
    let wide: Vec<u16> = dir
        .as_os_str()
        .encode_wide()
        .chain(std::iter::once(0))
        .collect();
    unsafe {
        SetDllDirectoryW(wide.as_ptr());
    }
}
#[cfg(not(windows))]
fn set_dll_directory(_dir: &Path) {}

fn init_runtime(dir: &Path) -> Result<()> {
    RUNTIME
        .get_or_init(|| {
            // DirectML.dll debe poder encontrarse junto a onnxruntime.dll
            set_dll_directory(dir);
            ort::init_from(dir.join(DLL_FILE).display().to_string())
                .commit()
                .map(|_| ())
                .map_err(|e| e.to_string())
        })
        .clone()
        .map_err(|e| Error::Message(format!("No se pudo iniciar ONNX Runtime: {e}")))
}

/// Adaptador DirectML (`REESCALLER_DML_DEVICE`, 0 por defecto; útil en portátiles con dos GPU).
fn dml_device() -> i32 {
    std::env::var("REESCALLER_DML_DEVICE")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0)
}

/// Carga una red; con `gpu` intenta DirectML y, si no puede, cae a CPU. Devuelve si quedó en GPU.
fn load_net(path: &Path, gpu: bool, log: Log) -> Result<(Net, bool)> {
    if gpu {
        let r = (|| -> ort::Result<Session> {
            Session::builder()?
                .with_memory_pattern(false)?
                .with_parallel_execution(false)?
                .with_execution_providers([DirectMLExecutionProvider::default()
                    .with_device_id(dml_device())
                    .build()
                    .error_on_failure()])?
                .commit_from_file(path)
        })();
        match r {
            Ok(s) => return Ok((Net { s }, true)),
            Err(e) => log(&format!("DirectML no disponible ({e}); se usa la CPU.")),
        }
    }
    let s = Session::builder()
        .and_then(|b| b.with_execution_providers([CPUExecutionProvider::default().build()]))
        .and_then(|b| b.commit_from_file(path))
        .map_err(|e| Error::Message(format!("No se pudo cargar {}: {e}", path.display())))?;
    Ok((Net { s }, false))
}

impl Engine {
    fn new(dir: &Path, want_gpu: bool, log: Log) -> Result<Engine> {
        init_runtime(dir)?;
        let t = Instant::now();
        let gpu_try = want_gpu && !GPU_BROKEN.load(Ordering::SeqCst);
        let (det, g1) = load_net(&dir.join(DET_FILE), gpu_try, log)?;
        // si el primero cayó a CPU, el resto también (no se mezclan dispositivos)
        let (parse, g2) = load_net(&dir.join(PARSE_FILE), g1, log)?;
        log(&format!(
            "Motor de rostros: ONNX Runtime ({}) · modelos base cargados en {:.1}s",
            if g1 && g2 { "DirectML/GPU" } else { "CPU" },
            t.elapsed().as_secs_f32()
        ));
        Ok(Engine {
            dir: dir.to_path_buf(),
            want_gpu,
            gpu: g1 && g2,
            det,
            parse,
            restorers: HashMap::new(),
        })
    }

    fn restorer(&mut self, model: FaceModel, log: Log) -> Result<&mut Net> {
        let key = model.onnx_file();
        if !self.restorers.contains_key(key) {
            let t = Instant::now();
            let (n, g) = load_net(&self.dir.join(key), self.gpu, log)?;
            if g != self.gpu {
                log("El modelo de restauración no pudo usar la GPU; se continúa en CPU.");
                self.gpu = false;
            }
            log(&format!(
                "Modelo {} cargado en {:.1}s",
                model.as_str(),
                t.elapsed().as_secs_f32()
            ));
            self.restorers.insert(key, n);
        }
        Ok(self.restorers.get_mut(key).unwrap())
    }

    fn device(&self) -> String {
        if self.gpu {
            "DirectML (GPU)".into()
        } else {
            "CPU".into()
        }
    }
}

fn run(net: &mut Net, input: Vec<f32>, shape: [usize; 4]) -> Result<Vec<(Vec<i64>, Vec<f32>)>> {
    let t = Tensor::from_array((shape, input)).map_err(msg)?;
    let outs = net.s.run(ort::inputs![t]).map_err(msg)?;
    let mut res = Vec::new();
    for (_, v) in outs.iter() {
        let (sh, d) = v.try_extract_tensor::<f32>().map_err(msg)?;
        res.push((sh.to_vec(), d.to_vec()));
    }
    Ok(res)
}

// ------------------------------------------------------------------ detección
struct Face {
    lm: [(f32, f32); 5],
}

/// Detecta caras en `det_img` (ya reducida si hacía falta); los puntos se devuelven en esa escala.
fn detect(net: &mut Net, det_img: &Img, conf_thr: f32) -> Result<Vec<Face>> {
    let (w, h) = (det_img.w, det_img.h);
    let mut x = vec![0.0f32; 3 * w * h];
    for i in 0..w * h {
        let (r, g, b) = (det_img.d[i * 3], det_img.d[i * 3 + 1], det_img.d[i * 3 + 2]);
        x[i] = b - 104.0;
        x[w * h + i] = g - 117.0;
        x[2 * w * h + i] = r - 123.0;
    }
    let out = run(net, x, [1, 3, h, w])?;
    if out.len() < 3 {
        return Err(Error::Message(
            "El detector de caras devolvió una salida inesperada.".into(),
        ));
    }
    let (loc, conf, ldm) = (&out[0].1, &out[1].1, &out[2].1);

    let steps = [8.0f32, 16.0, 32.0];
    let min_sizes: [&[f32]; 3] = [&[16.0, 32.0], &[64.0, 128.0], &[256.0, 512.0]];
    let mut priors: Vec<[f32; 4]> = Vec::new();
    for (k, &step) in steps.iter().enumerate() {
        let (fh, fw) = (
            (h as f32 / step).ceil() as usize,
            (w as f32 / step).ceil() as usize,
        );
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
        return Err(Error::Message(format!(
            "Anclas ({}) y salida del detector ({}) no coinciden.",
            priors.len(),
            loc.len() / 4
        )));
    }

    struct Det {
        b: [f32; 4],
        score: f32,
        lm: [(f32, f32); 5],
    }
    let mut dets: Vec<Det> = Vec::new();
    for (n, p) in priors.iter().enumerate() {
        let score = conf[n * 2 + 1];
        if score <= conf_thr {
            continue;
        }
        let l = &loc[n * 4..n * 4 + 4];
        let (cx, cy) = (p[0] + l[0] * 0.1 * p[2], p[1] + l[1] * 0.1 * p[3]);
        let (bw, bh) = (p[2] * (l[2] * 0.2).exp(), p[3] * (l[3] * 0.2).exp());
        let b = [
            (cx - bw / 2.0) * w as f32,
            (cy - bh / 2.0) * h as f32,
            (cx + bw / 2.0) * w as f32,
            (cy + bh / 2.0) * h as f32,
        ];
        let mut lm = [(0.0, 0.0); 5];
        for (i, pt) in lm.iter_mut().enumerate() {
            let o = n * 10 + i * 2;
            *pt = (
                (p[0] + ldm[o] * 0.1 * p[2]) * w as f32,
                (p[1] + ldm[o + 1] * 0.1 * p[3]) * h as f32,
            );
        }
        dets.push(Det { b, score, lm });
    }
    dets.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut keep: Vec<Det> = Vec::new();
    for d in dets {
        let area = |b: &[f32; 4]| (b[2] - b[0] + 1.0) * (b[3] - b[1] + 1.0);
        let overlaps = keep.iter().any(|k| {
            let (x1, y1, x2, y2) = (
                d.b[0].max(k.b[0]),
                d.b[1].max(k.b[1]),
                d.b[2].min(k.b[2]),
                d.b[3].min(k.b[3]),
            );
            let inter = (x2 - x1 + 1.0).max(0.0) * (y2 - y1 + 1.0).max(0.0);
            inter / (area(&d.b) + area(&k.b) - inter) > 0.4
        });
        if !overlaps {
            keep.push(d);
        }
    }
    Ok(keep.into_iter().map(|d| Face { lm: d.lm }).collect())
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

fn bilinear_1c(src: &[f32], sw: usize, sh: usize, x: f32, y: f32) -> f32 {
    let x = x.clamp(0.0, sw as f32 - 1.0);
    let y = y.clamp(0.0, sh as f32 - 1.0);
    let (x0, y0) = (x.floor() as usize, y.floor() as usize);
    let (x1, y1) = ((x0 + 1).min(sw - 1), (y0 + 1).min(sh - 1));
    let (ax, ay) = (x - x0 as f32, y - y0 as f32);
    let g = |xx: usize, yy: usize| src[yy * sw + xx];
    (g(x0, y0) * (1.0 - ax) + g(x1, y0) * ax) * (1.0 - ay)
        + (g(x0, y1) * (1.0 - ax) + g(x1, y1) * ax) * ay
}

/// Mezcla la cara restaurada en `canvas` recorriendo solo el rectángulo que ocupa.
fn paste_face(canvas: &mut RgbImage, restored: &Img, mask: &[f32], aff: &Affine, up: f32) {
    let (cw, ch) = (canvas.width() as i32, canvas.height() as i32);
    // lienzo -> cara: la afín original sobre coordenadas del lienzo / ampliación
    let to_face: Affine = [
        aff[0] / up,
        aff[1] / up,
        aff[2],
        aff[3] / up,
        aff[4] / up,
        aff[5],
    ];
    // rectángulo del lienzo que cubre el cuadrado de 512 px de la cara
    let inv = invert(aff);
    let (mut x0, mut y0, mut x1, mut y1) = (f32::MAX, f32::MAX, f32::MIN, f32::MIN);
    for (fx, fy) in [(0.0, 0.0), (511.0, 0.0), (0.0, 511.0), (511.0, 511.0)] {
        let (x, y) = apply(&inv, fx, fy);
        let (x, y) = (x * up, y * up);
        x0 = x0.min(x);
        y0 = y0.min(y);
        x1 = x1.max(x);
        y1 = y1.max(y);
    }
    let (x0, y0) = (
        (x0.floor() as i32 - 2).max(0),
        (y0.floor() as i32 - 2).max(0),
    );
    let (x1, y1) = (
        (x1.ceil() as i32 + 2).min(cw - 1),
        (y1.ceil() as i32 + 2).min(ch - 1),
    );
    for y in y0..=y1 {
        for x in x0..=x1 {
            let (fx, fy) = apply(&to_face, x as f32, y as f32);
            if fx < -0.5 || fy < -0.5 || fx > 511.5 || fy > 511.5 {
                continue;
            }
            let a = bilinear_1c(mask, 512, 512, fx, fy);
            if a <= 0.0 {
                continue;
            }
            let p = restored.sample(fx, fy, [0.0; 3]);
            let px = canvas.get_pixel_mut(x as u32, y as u32);
            for c in 0..3 {
                px[c] = (a * p[c] + (1.0 - a) * px[c] as f32)
                    .round()
                    .clamp(0.0, 255.0) as u8;
            }
        }
    }
}

fn load_oriented(path: &Path) -> Result<RgbImage> {
    let reader = ImageReader::open(path)
        .map_err(|e| Error::Message(format!("{}: {e}", path.display())))?
        .with_guessed_format()
        .map_err(msg)?;
    let mut decoder = reader
        .into_decoder()
        .map_err(|e| Error::Message(format!("{}: {e}", path.display())))?;
    let orientation = decoder
        .orientation()
        .unwrap_or(image::metadata::Orientation::NoTransforms);
    let mut img = DynamicImage::from_decoder(decoder)
        .map_err(|e| Error::Message(format!("{}: {e}", path.display())))?;
    img.apply_orientation(orientation);
    Ok(img.to_rgb8())
}

fn process(
    engine: &mut Engine,
    model: FaceModel,
    base: &RgbImage,
    upscale: u32,
    log: Log,
) -> Result<(RgbImage, usize)> {
    let (w, h) = (base.width() as usize, base.height() as usize);

    // ampliación efectiva (limitada para no agotar la memoria)
    let mut up = upscale.max(1) as u64;
    while up > 1 && (w as u64 * h as u64 * up * up) > MAX_OUTPUT_PIXELS {
        up -= 1;
    }
    if up != upscale.max(1) as u64 {
        log(&format!(
            "Imagen muy grande: la ampliación se reduce a ×{up} (máximo {} MP de salida).",
            MAX_OUTPUT_PIXELS / 1_000_000
        ));
    }

    // detección sobre una copia reducida si hace falta
    let t = Instant::now();
    let long = w.max(h);
    let det_scale = if long > DET_MAX_SIDE {
        DET_MAX_SIDE as f32 / long as f32
    } else {
        1.0
    };
    let det_img = if det_scale < 1.0 {
        let (dw, dh) = (
            ((w as f32 * det_scale).round() as u32).max(1),
            ((h as f32 * det_scale).round() as u32).max(1),
        );
        Img::from_rgb8(&image::imageops::resize(base, dw, dh, FilterType::Triangle))
    } else {
        Img::from_rgb8(base)
    };
    let mut faces = detect(&mut engine.det, &det_img, 0.97)?;
    for f in faces.iter_mut() {
        for p in f.lm.iter_mut() {
            *p = (p.0 / det_scale, p.1 / det_scale);
        }
    }
    // descarta caras diminutas (distancia entre ojos ≤ 5 px), igual que GFPGANer
    faces.retain(|f| {
        ((f.lm[0].0 - f.lm[1].0).powi(2) + (f.lm[0].1 - f.lm[1].1).powi(2)).sqrt() > 5.0
    });
    log(&format!(
        "Caras detectadas: {} ({:.2}s)",
        faces.len(),
        t.elapsed().as_secs_f32()
    ));
    if faces.is_empty() {
        return Err(Error::Message("No se encontraron rostros.".into()));
    }

    let mut canvas = if up > 1 {
        image::imageops::resize(
            base,
            (w as u64 * up) as u32,
            (h as u64 * up) as u32,
            FilterType::Lanczos3,
        )
    } else {
        base.clone()
    };

    let total = faces.len();
    for (idx, f) in faces.iter().enumerate() {
        let t = Instant::now();
        let aff = estimate_similarity(&f.lm, &TEMPLATE);
        let crop = warp_to_crop(base, &aff, [132.0, 133.0, 135.0]);

        let out = run(
            engine.restorer(model, log)?,
            to_nchw_norm(&crop),
            [1, 3, 512, 512],
        )?;
        let o = &out[0].1;
        let mut restored = Img::new(512, 512);
        for i in 0..512 * 512 {
            for c in 0..3 {
                let v = o[c * 512 * 512 + i].clamp(-1.0, 1.0);
                restored.d[i * 3 + c] = ((v + 1.0) / 2.0 * 255.0).round();
            }
        }

        let lab = run(&mut engine.parse, to_nchw_norm(&restored), [1, 3, 512, 512])?;
        let (cls, data) = (lab[0].0[1] as usize, &lab[0].1);
        let keep_label = [
            0u8, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 0, 255, 0, 0, 0,
        ];
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
        let mask = gaussian_blur(&mask, 512, 512, 101, 11.0);
        let mut mask = gaussian_blur(&mask, 512, 512, 101, 11.0);
        let th = 10;
        for y in 0..512 {
            for x in 0..512 {
                if y < th || y >= 512 - th || x < th || x >= 512 - th {
                    mask[y * 512 + x] = 0.0;
                }
                mask[y * 512 + x] /= 255.0;
            }
        }

        // Fuerza < 1: se mezcla el rostro restaurado con el recorte original (más fiel a la foto, menos «reconstruido»).
        let k = f32::from_bits(STRENGTH.load(Ordering::SeqCst));
        if k < 0.999 && crop.d.len() == restored.d.len() {
            for i in 0..restored.d.len() {
                restored.d[i] = crop.d[i] * (1.0 - k) + restored.d[i] * k;
            }
        }
        paste_face(&mut canvas, &restored, &mask, &aff, up as f32);
        log(&format!(
            "Cara {}/{} restaurada ({:.2}s)",
            idx + 1,
            total,
            t.elapsed().as_secs_f32()
        ));
    }
    Ok((canvas, total))
}

/// Resultado de una restauración nativa.
#[derive(Debug, Clone)]
pub struct OnnxReport {
    pub faces: usize,
    pub device: String,
    pub width: u32,
    pub height: u32,
}

/// Restaura los rostros de `input` y guarda el resultado (PNG) en `output`.
pub fn restore_faces_onnx(
    dir: &Path,
    model: FaceModel,
    input: &Path,
    output: &Path,
    prefer_gpu: bool,
    upscale: u32,
    log: Log,
) -> Result<OnnxReport> {
    if !supports(dir, model) {
        return Err(Error::Message(format!(
            "Faltan los modelos ONNX de {} en {}",
            model.as_str(),
            dir.display()
        )));
    }
    let base = load_oriented(input)?;

    let mut guard = ENGINE
        .lock()
        .map_err(|_| Error::Message("Motor de rostros bloqueado.".into()))?;
    let reuse = matches!(guard.as_ref(), Some(e) if e.dir == dir && e.want_gpu == prefer_gpu);
    if !reuse {
        *guard = Some(Engine::new(dir, prefer_gpu, log)?);
    }

    let first = process(guard.as_mut().unwrap(), model, &base, upscale, log);
    let result = match first {
        Err(e)
            if guard.as_ref().map(|g| g.gpu).unwrap_or(false)
                && !e.to_string().contains("No se encontraron rostros") =>
        {
            // fallo en la GPU (memoria, driver...): se reintenta en CPU y se recuerda
            log(&format!("Fallo en la GPU ({e}); se reintenta en CPU."));
            GPU_BROKEN.store(true, Ordering::SeqCst);
            *guard = Some(Engine::new(dir, false, log)?);
            process(guard.as_mut().unwrap(), model, &base, upscale, log)
        }
        other => other,
    };
    let (canvas, faces) = result?;
    let device = guard
        .as_ref()
        .map(|g| g.device())
        .unwrap_or_else(|| "CPU".into());
    if let Some(parent) = output.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    canvas
        .save(output)
        .map_err(|e| Error::Message(format!("No se pudo guardar {}: {e}", output.display())))?;
    Ok(OnnxReport {
        faces,
        device,
        width: canvas.width(),
        height: canvas.height(),
    })
}
