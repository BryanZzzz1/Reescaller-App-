//! Estado compartido del servidor de la aplicación.

use reescaller_core::config::AppConfig;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};

/// Estado global de la aplicación.
pub struct State {
    pub config: Mutex<AppConfig>,
    /// Carpeta que contiene `models/` (ffmpeg, motor, modelos...).
    pub project_dir: PathBuf,
    pub logs: Mutex<Vec<(u64, String)>>,
    pub log_counter: Mutex<u64>,
    /// Trabajos pesados en curso (la app no se cierra mientras haya alguno).
    pub active_jobs: AtomicUsize,
    /// Último latido de la interfaz (ms desde epoch). 0 = ninguno todavía.
    pub last_ping_ms: AtomicU64,
    /// Otra ejecución pidió traer la ventana al frente.
    pub focus_request: Arc<AtomicBool>,
    /// Token aleatorio de esta ejecución: la interfaz lo recibe en el HTML y debe enviarlo en cada llamada a la API.
    pub token: String,
}

/// Genera 128 bits aleatorios (claves aleatorias del sistema vía `RandomState`).
fn token_aleatorio() -> String {
    use std::collections::hash_map::RandomState;
    use std::hash::{BuildHasher, Hasher};
    let mut out = String::new();
    for i in 0..2u64 {
        let mut h = RandomState::new().build_hasher();
        h.write_u64(now_ms() ^ (i << 32) ^ std::process::id() as u64);
        out.push_str(&format!("{:016x}", h.finish()));
    }
    out
}

/// Marca un trabajo como activo mientras viva el guard.
pub struct JobGuard<'a>(&'a State);

impl Drop for JobGuard<'_> {
    fn drop(&mut self) {
        self.0.active_jobs.fetch_sub(1, Ordering::SeqCst);
    }
}

pub fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

impl State {
    pub fn new(project_dir: PathBuf) -> Self {
        let mut config = AppConfig::load();
        config.autodetectar(&project_dir);
        // La sonda de GPU usa el mismo motor que se va a ejecutar.
        if let Some(motor) = config.realesrgan_exe.clone() {
            reescaller_core::hardware::set_engine_probe(motor);
        }
        // Borra las imágenes subidas o pegadas hace más de 24 h (contienen fotos privadas del usuario).
        std::thread::spawn(|| {
            let limite = std::time::Duration::from_secs(24 * 3600);
            if let Ok(entradas) = std::fs::read_dir(State::uploads_dir()) {
                for e in entradas.flatten() {
                    let antigua = e
                        .metadata()
                        .and_then(|m| m.modified())
                        .ok()
                        .and_then(|t| t.elapsed().ok())
                        .map(|d| d > limite)
                        .unwrap_or(false);
                    if antigua && e.path().is_file() {
                        let _ = std::fs::remove_file(e.path());
                    }
                }
            }
        });
        // Quita el aviso «Abrir archivo - Advertencia de seguridad» de Windows en el motor y FFmpeg descargados.
        {
            let dir = project_dir.clone();
            std::thread::spawn(move || reescaller_core::process::unblock_downloaded(&dir));
        }
        Self {
            config: Mutex::new(config),
            project_dir,
            logs: Mutex::new(Vec::new()),
            log_counter: Mutex::new(0),
            active_jobs: AtomicUsize::new(0),
            last_ping_ms: AtomicU64::new(0),
            focus_request: Arc::new(AtomicBool::new(false)),
            token: token_aleatorio(),
        }
    }

    pub fn begin_job(&self) -> JobGuard<'_> {
        self.active_jobs.fetch_add(1, Ordering::SeqCst);
        JobGuard(self)
    }

    pub fn busy(&self) -> bool {
        self.active_jobs.load(Ordering::SeqCst) > 0
    }

    pub fn ping(&self) {
        self.last_ping_ms.store(now_ms(), Ordering::SeqCst);
    }

    /// Carpeta de datos del usuario (escribible aunque la app esté en Program Files).
    pub fn data_dir() -> PathBuf {
        dirs::data_local_dir()
            .unwrap_or_else(std::env::temp_dir)
            .join("Reescaller")
    }

    /// Carpeta para archivos subidos desde la interfaz.
    pub fn uploads_dir() -> PathBuf {
        Self::data_dir().join("uploads")
    }

    /// Carpeta de resultados por defecto cuando el original no tiene ubicación propia.
    pub fn default_outputs_dir() -> PathBuf {
        dirs::picture_dir()
            .unwrap_or_else(|| dirs::home_dir().unwrap_or_else(std::env::temp_dir))
            .join("Reescaller")
    }

    /// Añade una línea al registro global.
    pub fn push_log(&self, message: &str) {
        let mut counter = self.log_counter.lock().unwrap();
        *counter += 1;
        let id = *counter;
        drop(counter);
        let mut logs = self.logs.lock().unwrap();
        logs.push((id, message.to_string()));
        if logs.len() > 2_000 {
            let overflow = logs.len() - 2_000;
            logs.drain(0..overflow);
        }
    }

    /// Devuelve `(último_id, líneas)` posteriores a `since`.
    pub fn logs_since(&self, since: u64) -> (u64, Vec<String>) {
        let logs = self.logs.lock().unwrap();
        let last = logs.last().map(|(id, _)| *id).unwrap_or(since);
        let lines = logs
            .iter()
            .filter(|(id, _)| *id > since)
            .map(|(_, line)| line.clone())
            .collect();
        (last, lines)
    }
}
