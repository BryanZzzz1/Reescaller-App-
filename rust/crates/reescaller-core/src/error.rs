//! Errores del núcleo.

use std::path::PathBuf;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("No se encontró el ejecutable: {0}")]
    ExecutableNotFound(PathBuf),

    #[error("No se encontró el archivo: {0}")]
    FileNotFound(PathBuf),

    #[error("Fallo al ejecutar {program} (código {code}): {stderr}")]
    CommandFailed {
        program: String,
        code: i32,
        stderr: String,
    },

    #[error("Tiempo de espera agotado al ejecutar: {0}")]
    Timeout(String),

    #[error("Error de imagen: {0}")]
    Image(String),

    #[error("{0}")]
    Message(String),

    #[error("Operación cancelada")]
    Cancelled,

    #[error("Configuración inválida: {0}")]
    InvalidConfig(String),

    #[error(transparent)]
    Io(#[from] std::io::Error),

    #[error(transparent)]
    Json(#[from] serde_json::Error),
}

pub type Result<T> = std::result::Result<T, Error>;
