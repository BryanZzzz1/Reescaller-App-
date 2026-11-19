//! Autodetección de hardware: CPU, RAM, GPU (NVIDIA / AMD / Intel) y Vulkan.
//!
//! El motor de mejora es `realesrgan-ncnn-vulkan`, que funciona sobre **cualquier**
//! GPU con Vulkan 1.x, es decir:
//!
//! - **NVIDIA**: driver oficial (GeForce / RTX / GTX / Quadro) → Vulkan oficial.
//! - **AMD**: driver Adrenalin (RDNA 1-4) o Mesa/RADV → Vulkan oficial.
//! - **Intel**: driver Arc (Xe) o el driver de video de las iGPU recientes →
//!   Vulkan oficial; las iGPU Intel muy antiguas no lo traen y se avisa.
//!
//! Fuentes de información (de más a menos fiables):
//!
//! 1. `vulkaninfo --summary`: enumera los dispositivos Vulkan reales. Es la
//!    fuente de verdad: define el índice que usa `-g` del motor.
//! 2. `nvidia-smi`: nombre, driver y VRAM de las tarjetas NVIDIA.
//! 3. `Win32_VideoController` (PowerShell/CIM o `wmic`): todas las tarjetas.
//!
//! Cuando no hay Vulkan el motor IA no puede arrancar: la aplicación cae
//! automáticamente al modo **solo CPU** (reescalado de alta calidad sin IA).

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::{Mutex, OnceLock};

/// Fabricante de la GPU.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum GpuVendor {
    Nvidia,
    Amd,
    Intel,
    Other,
    Unknown,
}

impl GpuVendor {
    /// Identificador corto (`nvidia`, `amd`, `intel`).
    pub fn as_str(&self) -> &'static str {
        match self {
            GpuVendor::Nvidia => "nvidia",
            GpuVendor::Amd => "amd",
            GpuVendor::Intel => "intel",
            GpuVendor::Other => "otro",
            GpuVendor::Unknown => "desconocido",
        }
    }

    /// Nombre legible para la interfaz.
    pub fn label(&self) -> &'static str {
        match self {
            GpuVendor::Nvidia => "NVIDIA",
            GpuVendor::Amd => "AMD",
            GpuVendor::Intel => "Intel",
            GpuVendor::Other => "Otro",
            GpuVendor::Unknown => "Desconocida",
        }
    }

    /// Deduce el fabricante a partir del nombre y del adaptador.
    pub fn from_text(nombre: &str, adaptador: &str) -> GpuVendor {
        let texto = format!("{nombre} {adaptador}").to_lowercase();
        if texto.contains("nvidia")
            || texto.contains("geforce")
            || texto.contains("quadro")
            || texto.contains("rtx")
            || texto.contains("gtx")
        {
            GpuVendor::Nvidia
        } else if texto.contains("amd")
            || texto.contains("radeon")
            || texto.contains("ryzen")
            || texto.contains("advanced micro")
        {
            GpuVendor::Amd
        } else if texto.contains("intel")
            || texto.contains("arc ")
            || texto.contains("iris")
            || texto.contains("uhd graphics")
            || texto.contains("hd graphics")
        {
            GpuVendor::Intel
        } else if texto.contains("microsoft") {
            GpuVendor::Other
        } else {
            GpuVendor::Unknown
        }
    }
}
/// Información de una GPU detectada.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GpuInfo {
    /// Índice que usa `realesrgan-ncnn-vulkan` (`-g`). Corresponde al orden de
    /// dispositivos Vulkan cuando este está disponible.
    pub index: usize,
    pub vendor: GpuVendor,
    pub name: String,
    /// Versión del driver (WMI) o nombre del driver Vulkan.
    pub driver: String,
    /// VRAM en MB. `None` si el sistema no lo reporta (las iGPU no tienen VRAM propia).
    pub vram_mb: Option<u32>,
    /// `true` cuando la VRAM es el mínimo informado: Windows limita `AdapterRAM`
    /// a 4 GB, así que una tarjeta de 8 GB o más se reporta como "4 GB o más".
    pub vram_is_minimum: bool,
    /// `true` si es una GPU dedicada.
    pub dedicated: bool,
    /// De dónde salió la información (`vulkaninfo`, `nvidia-smi`, `WMI`, ...).
    pub source: String,
}

impl GpuInfo {
    /// Texto corto para la interfaz: "NVIDIA GeForce RTX 3060 (12 GB VRAM)".
    pub fn describe(&self) -> String {
        let mut out = if self
            .name
            .to_lowercase()
            .starts_with(&self.vendor.label().to_lowercase())
        {
            self.name.clone()
        } else {
            format!("{} {}", self.vendor.label(), self.name)
        };
        match self.vram_mb {
            Some(v) if v > 0 => {
                if self.vram_is_minimum {
                    out.push_str(&format!(" ({} GB VRAM o más)", (v + 512) / 1024));
                } else {
                    out.push_str(&format!(" ({} GB VRAM)", (v + 512) / 1024));
                }
            }
            _ => out.push_str(" (GPU integrada)"),
        }
        out
    }
}

/// Dispositivo Vulkan enumerado por `vulkaninfo`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct VulkanDevice {
    /// Índice del dispositivo Vulkan (el que entiende `-g` del motor).
    pub index: usize,
    pub name: String,
    pub driver: String,
    pub api_version: String,
    /// `true` si es una GPU dedicada (`PHYSICAL_DEVICE_TYPE_DISCRETE_GPU`).
    pub discrete: bool,
}

/// Nivel de soporte del motor de IA en este equipo.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum EngineSupport {
    /// Vulkan disponible: motor IA completo (x2 / x4).
    Full,
    /// Sin Vulkan: solo reescalado por CPU con filtros de alta calidad.
    CpuOnly,
}

impl EngineSupport {
    pub fn as_str(&self) -> &'static str {
        match self {
            EngineSupport::Full => "completo",
            EngineSupport::CpuOnly => "solo_cpu",
        }
    }
}
/// Informe completo de hardware.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct HardwareReport {
    pub os: String,
    pub cpu: String,
    pub logical_cores: usize,
    pub physical_cores: Option<usize>,
    pub ram_total_mb: u64,
    pub ram_available_mb: u64,
    pub gpus: Vec<GpuInfo>,
    /// Dispositivos Vulkan detectados (mismo orden que `gpus`).
    pub vulkan_devices: Vec<VulkanDevice>,
    pub vulkan_available: bool,
    pub vulkan_detail: String,
    pub vulkan_version: String,
    /// ICD de Vulkan encontrados (`nvidia_icd.json`, `amd_icd.json`, ...).
    pub vulkan_icds: Vec<String>,
    pub engine_support: EngineSupport,
    /// Número de trabajos simultáneos recomendado en modo lote.
    pub recommended_workers: usize,
    /// Avisos y recomendaciones legibles.
    pub notes: Vec<String>,
}

impl HardwareReport {
    /// GPU principal: la dedicada con más VRAM; si no hay, la primera.
    pub fn primary_gpu(&self) -> Option<&GpuInfo> {
        self.gpus
            .iter()
            .filter(|g| g.dedicated)
            .max_by_key(|g| g.vram_mb.unwrap_or(0))
            .or_else(|| self.gpus.first())
    }

    /// Índice de la primera GPU del fabricante pedido (para `-g` del motor).
    pub fn index_of_vendor(&self, vendor: GpuVendor) -> Option<usize> {
        self.gpus
            .iter()
            .find(|g| g.vendor == vendor)
            .map(|g| g.index)
    }

    /// Resumen de una línea para la consola.
    pub fn summary(&self) -> String {
        let gpu = self
            .primary_gpu()
            .map(|g| g.describe())
            .unwrap_or_else(|| "sin GPU detectada".to_string());
        format!(
            "{} · {} núcleos lógicos · {:.1} GB RAM · {} · Vulkan: {}",
            self.cpu,
            self.logical_cores,
            self.ram_total_mb as f64 / 1024.0,
            gpu,
            if self.vulkan_available { "sí" } else { "no" }
        )
    }
}
static CACHE: Mutex<Option<&'static HardwareReport>> = Mutex::new(None);
static CACHE_TIME: Mutex<Option<std::time::Instant>> = Mutex::new(None);

fn guardar_cache(r: &'static HardwareReport) {
    *CACHE.lock().unwrap_or_else(|e| e.into_inner()) = Some(r);
    *CACHE_TIME.lock().unwrap_or_else(|e| e.into_inner()) = Some(std::time::Instant::now());
}

/// Detecta el hardware (con caché: la detección tarda ~1 s). Un resultado «sin Vulkan» se vuelve a comprobar pasados
/// 30 s: la sonda puede fallar una vez (driver lento, GPU dormida en un portátil híbrido) y no debe apagar la IA toda la sesión.
pub fn detect_cached() -> &'static HardwareReport {
    {
        let g = CACHE.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(r) = *g {
            let viejo = CACHE_TIME
                .lock()
                .ok()
                .and_then(|t| *t)
                .map(|t| t.elapsed().as_secs() > 30)
                .unwrap_or(false);
            if r.vulkan_available || !viejo {
                return r;
            }
        }
    } // el bloqueo se suelta antes de detectar (puede tardar varios segundos)
    let r: &'static HardwareReport = Box::leak(Box::new(detect()));
    guardar_cache(r);
    r
}

/// Vuelve a detectar el hardware (GPU nuevas, drivers actualizados...) y reemplaza la caché.
pub fn refresh() -> &'static HardwareReport {
    let r: &'static HardwareReport = Box::leak(Box::new(detect()));
    guardar_cache(r);
    r
}

/// Detecta el hardware del equipo.
pub fn detect() -> HardwareReport {
    let logical_cores = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);

    let (cpu, physical_cores) = detectar_cpu();
    let (ram_total_mb, ram_available_mb) = memoria();

    let (gpus, vulkan_dispositivos) = detectar_gpus();
    let (vulkan_available, vulkan_detail, vulkan_version, vulkan_icds) =
        estado_vulkan(&vulkan_dispositivos);

    let mut notes: Vec<String> = Vec::new();

    if gpus.is_empty() {
        notes.push(
            "No se detectó ninguna GPU. Reescaller funcionará solo con CPU (reescalado de \
             alta calidad, sin mejora por IA)."
                .to_string(),
        );
    }

    if !vulkan_available && !gpus.is_empty() {
        let aviso = match gpus.first().map(|g| g.vendor) {
            Some(GpuVendor::Intel) => {
                "Intel sin Vulkan detectado: actualiza el driver de \
                gráficos (Intel Arc o el driver de video más reciente) para habilitar la mejora \
                por IA. Mientras tanto se usa el modo CPU."
            }
            Some(GpuVendor::Amd) => {
                "AMD sin Vulkan detectado: instala el driver Adrenalin o \
                Mesa para habilitar la mejora por IA."
            }
            Some(GpuVendor::Nvidia) => {
                "NVIDIA sin Vulkan detectado: reinstala el driver para \
                habilitar la mejora por IA."
            }
            _ => "Sin Vulkan: la mejora por IA no está disponible.",
        };
        notes.push(aviso.to_string());
    }

    if vulkan_available {
        if let Some(g) = gpus.iter().find(|g| g.vendor == GpuVendor::Intel) {
            notes.push(format!(
                "Intel detectada ({}): el motor Vulkan funciona, aunque las iGPU integradas \
                 son más lentas que una GPU dedicada.",
                g.name
            ));
        }
        if let Some(g) = gpus.iter().find(|g| g.vendor == GpuVendor::Amd) {
            notes.push(format!(
                "AMD detectada ({}): listo para Vulkan (requiere Vulkan 1.2 o superior).",
                g.name
            ));
        }
        if let Some(g) = gpus.iter().find(|g| g.vendor == GpuVendor::Nvidia) {
            notes.push(format!(
                "NVIDIA detectada ({}): el motor Vulkan funciona a máxima velocidad.",
                g.name
            ));
        }
    }

    let engine_support = if vulkan_available {
        EngineSupport::Full
    } else {
        EngineSupport::CpuOnly
    };

    // El motor ya usa todos los recursos de la GPU, así que en modo lote con
    // GPU dedicada conviene un solo proceso; sin GPU se paraleliza por CPU.
    let recommended_workers = if gpus.iter().any(|g| g.dedicated) {
        1
    } else {
        (logical_cores / 2).clamp(1, 8)
    };

    HardwareReport {
        os: os_desc(),
        cpu,
        logical_cores,
        physical_cores,
        ram_total_mb,
        ram_available_mb,
        gpus,
        vulkan_devices: vulkan_dispositivos,
        vulkan_available,
        vulkan_detail,
        vulkan_version,
        vulkan_icds,
        engine_support,
        recommended_workers,
        notes,
    }
}
fn os_desc() -> String {
    if cfg!(windows) {
        match nuevo_cmd("cmd").args(["/C", "ver"]).output() {
            Ok(o) => String::from_utf8_lossy(&o.stdout).trim().to_string(),
            Err(_) => "Windows".to_string(),
        }
    } else {
        std::env::consts::OS.to_string()
    }
}

/// Detecta el nombre de la CPU y los núcleos físicos.
fn detectar_cpu() -> (String, Option<usize>) {
    if cfg!(windows) {
        let script = "$c=Get-CimInstance Win32_Processor|Select-Object -First 1;\
                      [PSCustomObject]@{cpu=$c.Name;cores=$c.NumberOfCores}|ConvertTo-Json -Compress";
        if let Some(json) = ejecutar_powershell(script) {
            if let Ok(v) = serde_json::from_str::<serde_json::Value>(&json) {
                let cpu = v
                    .get("cpu")
                    .and_then(|x| x.as_str())
                    .unwrap_or("CPU desconocido")
                    .trim()
                    .to_string();
                let cores = v.get("cores").and_then(|x| x.as_u64()).map(|n| n as usize);
                return (cpu, cores);
            }
        }
    }
    let cpu = std::env::var("PROCESSOR_IDENTIFIER").unwrap_or_else(|_| "CPU".to_string());
    (cpu, None)
}

/// RAM del sistema mediante `GlobalMemoryStatusEx` (sin dependencias externas).
fn memoria() -> (u64, u64) {
    #[cfg(windows)]
    {
        #[repr(C)]
        #[derive(Default)]
        struct MemoryStatusEx {
            length: u32,
            memory_load: u32,
            total_phys: u64,
            avail_phys: u64,
            total_page_file: u64,
            avail_page_file: u64,
            total_virtual: u64,
            avail_virtual: u64,
            avail_extended_virtual: u64,
        }

        #[link(name = "kernel32")]
        extern "system" {
            fn GlobalMemoryStatusEx(buf: *mut MemoryStatusEx) -> i32;
        }

        let mut buf = MemoryStatusEx {
            length: std::mem::size_of::<MemoryStatusEx>() as u32,
            ..Default::default()
        };
        unsafe {
            if GlobalMemoryStatusEx(&mut buf) != 0 {
                return (
                    buf.total_phys / (1024 * 1024),
                    buf.avail_phys / (1024 * 1024),
                );
            }
        }
        (0, 0)
    }
    #[cfg(not(windows))]
    {
        (0, 0)
    }
}

/// Crea un comando que no abre ventana de consola (la app es gráfica).
fn nuevo_cmd<S: AsRef<std::ffi::OsStr>>(programa: S) -> Command {
    let mut c = Command::new(programa);
    crate::process::sin_ventana(&mut c);
    c
}

/// Ejecuta un script de PowerShell y devuelve su salida estándar.
fn ejecutar_powershell(script: &str) -> Option<String> {
    if crate::process::which("powershell").is_none()
        && crate::process::which("powershell.exe").is_none()
        && !Path::new("powershell.exe").is_file()
    {
        return None;
    }
    let salida = nuevo_cmd("powershell.exe")
        .args([
            "-NoProfile",
            "-NonInteractive",
            "-ExecutionPolicy",
            "Bypass",
            "-Command",
            script,
        ])
        .stdin(std::process::Stdio::null())
        .output()
        .ok()?;
    if !salida.status.success() {
        return None;
    }
    let texto = String::from_utf8_lossy(&salida.stdout).trim().to_string();
    if texto.is_empty() {
        None
    } else {
        Some(texto)
    }
}
/// Detecta las GPUs combinando Vulkan, nvidia-smi y WMI.
///
/// El índice final es el índice del dispositivo Vulkan (el que entiende `-g`
/// del motor); si no hay Vulkan se usa el orden de WMI.
fn detectar_gpus() -> (Vec<GpuInfo>, Vec<VulkanDevice>) {
    let vulkan = combinar_vulkan(detectar_motor(), detectar_vulkaninfo());
    let nvidia = detectar_nvidia();
    let wmi = detectar_wmi();

    // Base: lo que reporta WMI (nombre, VRAM, versión de driver).
    let mut base: BTreeMap<String, GpuInfo> = BTreeMap::new();
    for gpu in wmi {
        base.insert(gpu.name.to_lowercase(), gpu);
    }
    // nvidia-smi tiene datos más precisos de VRAM:補全会.
    for gpu in nvidia {
        let clave = gpu.name.to_lowercase();
        match base.get_mut(&clave) {
            Some(existente) => {
                if !gpu.driver.is_empty() {
                    existente.driver = gpu.driver;
                }
                if gpu.vram_mb.is_some() {
                    existente.vram_mb = gpu.vram_mb;
                    existente.vram_is_minimum = false;
                }
                existente.dedicated = true;
                existente.source = "nvidia-smi".to_string();
            }
            None => {
                base.insert(clave, gpu);
            }
        }
    }

    // Windows limita AdapterRAM a 4 GB; el registro guarda la VRAM real.
    let registro = vram_registro();
    for gpu in base.values_mut() {
        if gpu.vram_is_minimum || gpu.vram_mb.is_none() {
            if let Some((_, bytes)) = registro
                .iter()
                .find(|(n, _)| nombres_coinciden(n, &gpu.name))
            {
                let mb = (*bytes / (1024 * 1024)) as u32;
                if mb > 0 {
                    gpu.vram_mb = Some(mb);
                    gpu.vram_is_minimum = false;
                }
            }
        }
    }

    let mut lista: Vec<GpuInfo> = Vec::new();

    if vulkan.is_empty() {
        // Sin enumeración Vulkan: se respeta el orden de WMI.
        let mut ordenadas: Vec<GpuInfo> = base.into_values().collect();
        ordenadas.sort_by_key(|a| a.name.to_lowercase());
        for (i, gpu) in ordenadas.into_iter().enumerate() {
            lista.push(GpuInfo { index: i, ..gpu });
        }
        return (lista, Vec::new());
    }

    // Con Vulkan: el índice es el del dispositivo Vulkan.
    for dev in &vulkan {
        let info = match base
            .keys()
            .find(|k| nombres_coinciden(k, &dev.name))
            .cloned()
        {
            Some(clave) => base.remove(&clave).unwrap(),
            None => GpuInfo {
                index: dev.index,
                vendor: GpuVendor::from_text(&dev.name, ""),
                name: dev.name.clone(),
                driver: dev.driver.clone(),
                vram_mb: None,
                vram_is_minimum: false,
                dedicated: dev.discrete,
                source: "vulkan".to_string(),
            },
        };
        lista.push(GpuInfo {
            index: dev.index,
            // El tipo de dispositivo Vulkan manda; WMI no distingue una iGPU AMD
            // con memoria compartida de una tarjeta dedicada.
            dedicated: dev.discrete || info.source == "nvidia-smi",
            driver: if info.driver.is_empty() {
                dev.driver.clone()
            } else {
                info.driver
            },
            source: if info.source == "WMI" {
                "vulkan+WMI".to_string()
            } else {
                info.source.clone()
            },
            ..info
        });
    }

    // GPUs que solo aparecen en WMI (sin Vulkan: p.ej. NVIDIA con el driver
    // antiguo o una iGPU sin Vulkan) se añaden al final.
    for (_, gpu) in base.into_iter() {
        let indice = lista.len();
        lista.push(GpuInfo {
            index: indice,
            ..gpu
        });
    }

    (lista, vulkan)
}

/// VRAM real de cada adaptador (`HardwareInformation.qwMemorySize` del registro).
fn vram_registro() -> Vec<(String, u64)> {
    if !cfg!(windows) {
        return Vec::new();
    }
    let script = r#"Get-ItemProperty 'HKLM:\SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}\0*' -ErrorAction SilentlyContinue | Where-Object { $_.'HardwareInformation.qwMemorySize' } | ForEach-Object { $m=$_.'HardwareInformation.qwMemorySize'; if($m -is [byte[]]){ $m=[BitConverter]::ToUInt64($m,0) }; [PSCustomObject]@{n=[string]$_.DriverDesc;m=[uint64]$m} } | ConvertTo-Json -Compress"#;
    let Some(json) = ejecutar_powershell(script) else {
        return Vec::new();
    };
    let Ok(valor) = serde_json::from_str::<serde_json::Value>(&json) else {
        return Vec::new();
    };
    let elementos: Vec<serde_json::Value> = match valor {
        serde_json::Value::Array(a) => a,
        v @ serde_json::Value::Object(_) => vec![v],
        _ => return Vec::new(),
    };
    elementos
        .iter()
        .filter_map(|e| Some((e.get("n")?.as_str()?.to_string(), e.get("m")?.as_u64()?)))
        .collect()
}

/// GPUs NVIDIA mediante `nvidia-smi`.
fn detectar_nvidia() -> Vec<GpuInfo> {
    let salida = nuevo_cmd("nvidia-smi")
        .args([
            "--query-gpu=index,name,driver_version,memory.total",
            "--format=csv,noheader,nounits",
        ])
        .stdin(std::process::Stdio::null())
        .output();
    let salida = match salida {
        Ok(s) if s.status.success() => s,
        _ => return Vec::new(),
    };
    let texto = String::from_utf8_lossy(&salida.stdout);
    let mut lista = Vec::new();
    for (i, linea) in texto.lines().enumerate() {
        let campos: Vec<&str> = linea.split(',').map(|c| c.trim()).collect();
        if campos.len() < 4 {
            continue;
        }
        let vram = campos[3]
            .parse::<u32>()
            .ok()
            .map(|mb| mb / 1024)
            .filter(|gb| *gb > 0);
        lista.push(GpuInfo {
            index: campos[0].parse().unwrap_or(i),
            vendor: GpuVendor::Nvidia,
            name: campos[1].to_string(),
            driver: campos[2].to_string(),
            vram_mb: vram,
            vram_is_minimum: false,
            dedicated: true,
            source: "nvidia-smi".to_string(),
        });
    }
    lista
}
/// GPUs detectadas con PowerShell (CIM) o `wmic` como alternativa.
fn detectar_wmi() -> Vec<GpuInfo> {
    if cfg!(windows) {
        let script = "Get-CimInstance Win32_VideoController|Select-Object Name,DriverVersion,\
                      AdapterCompatibility,AdapterRAM|ConvertTo-Json -Compress";
        if let Some(json) = ejecutar_powershell(script) {
            let lista = parse_gpus_json(&json);
            if !lista.is_empty() {
                return lista;
            }
        }
        if let Some(salida) = nuevo_cmd("wmic")
            .args([
                "path",
                "win32_VideoController",
                "get",
                "Name,DriverVersion,AdapterRAM,AdapterCompatibility",
                "/format:csv",
            ])
            .stdin(std::process::Stdio::null())
            .output()
            .ok()
            .filter(|s| s.status.success())
        {
            let texto = String::from_utf8_lossy(&salida.stdout);
            let lista = parse_gpus_wmic(&texto);
            if !lista.is_empty() {
                return lista;
            }
        }
    }
    Vec::new()
}

/// Convierte la salida JSON de `Win32_VideoController` en una lista de GPUs.
fn parse_gpus_json(json: &str) -> Vec<GpuInfo> {
    let valor: serde_json::Value = match serde_json::from_str(json) {
        Ok(v) => v,
        Err(_) => return Vec::new(),
    };
    let elementos: Vec<&serde_json::Value> = match &valor {
        serde_json::Value::Array(a) => a.iter().collect(),
        serde_json::Value::Object(_) => vec![&valor],
        _ => return Vec::new(),
    };

    let mut lista = Vec::new();
    for e in elementos {
        let nombre = e
            .get("Name")
            .and_then(|x| x.as_str())
            .unwrap_or("")
            .trim()
            .to_string();
        if nombre.is_empty() {
            continue;
        }
        let adaptador = e
            .get("AdapterCompatibility")
            .and_then(|x| x.as_str())
            .unwrap_or("");
        let driver = e
            .get("DriverVersion")
            .and_then(|x| x.as_str())
            .unwrap_or("")
            .trim()
            .to_string();
        let (vram_mb, vram_is_minimum) = vram_de_wmi(e.get("AdapterRAM").and_then(|x| x.as_u64()));
        let vendor = GpuVendor::from_text(&nombre, adaptador);
        let dedicada = vendor != GpuVendor::Intel
            && vram_mb.map(|m| m >= 1024).unwrap_or(false)
            && !nombre.to_lowercase().contains("software");
        lista.push(GpuInfo {
            index: 0,
            vendor,
            name: nombre,
            driver,
            vram_mb,
            vram_is_minimum,
            dedicated: dedicada,
            source: "WMI".to_string(),
        });
    }
    lista
}

/// Windows limita `AdapterRAM` a 4 GB: si el valor llega al tope se marca como
/// "4 GB o más" en lugar de dar una cifra falsa.
fn vram_de_wmi(bytes: Option<u64>) -> (Option<u32>, bool) {
    match bytes {
        Some(bytes) if bytes >= 3_500_000_000 => (Some(4096), true),
        Some(bytes) => {
            let mb = (bytes / (1024 * 1024)) as u32;
            if mb == 0 {
                (None, false)
            } else {
                (Some(mb), false)
            }
        }
        None => (None, false),
    }
}
/// Alternativa con `wmic` (salida CSV).
fn parse_gpus_wmic(texto: &str) -> Vec<GpuInfo> {
    let mut lista = Vec::new();
    for linea in texto.lines().skip(1) {
        let campos: Vec<&str> = linea.split(',').map(|c| c.trim()).collect();
        if campos.len() < 5 {
            continue;
        }
        let nombre = campos[3];
        if nombre.is_empty() || nombre.eq_ignore_ascii_case("Name") {
            continue;
        }
        let adaptador = campos[4];
        let driver = campos.get(2).copied().unwrap_or("").trim().to_string();
        let (vram_mb, vram_is_minimum) =
            vram_de_wmi(campos.get(1).and_then(|v| v.parse::<u64>().ok()));
        let vendor = GpuVendor::from_text(nombre, adaptador);
        let dedicada = vendor != GpuVendor::Intel
            && vram_mb.map(|m| m >= 1024).unwrap_or(false)
            && !nombre.to_lowercase().contains("software");
        lista.push(GpuInfo {
            index: 0,
            vendor,
            name: nombre.to_string(),
            driver,
            vram_mb,
            vram_is_minimum,
            dedicated: dedicada,
            source: "wmic".to_string(),
        });
    }
    lista
}
/// Busca el ejecutable `vulkaninfo` y enumera los dispositivos Vulkan.
fn detectar_vulkaninfo() -> Vec<VulkanDevice> {
    let exe = match crate::process::which("vulkaninfo") {
        Some(p) => p,
        None => {
            let system32 = std::env::var("SystemRoot")
                .map(PathBuf::from)
                .unwrap_or_else(|_| PathBuf::from("C:\\Windows"))
                .join("System32");
            let mut encontrado = None;
            for nombre in ["vulkaninfo.exe", "vulkaninfo"] {
                let cand = system32.join(nombre);
                if cand.is_file() {
                    encontrado = Some(cand);
                    break;
                }
            }
            match encontrado {
                Some(p) => p,
                None => return Vec::new(),
            }
        }
    };

    let salida = nuevo_cmd(&exe)
        .args(["--summary"])
        .stdin(std::process::Stdio::null())
        .output();
    let salida = match salida {
        Ok(s) => s,
        Err(_) => return Vec::new(),
    };
    let texto = format!(
        "{}\n{}",
        String::from_utf8_lossy(&salida.stdout),
        String::from_utf8_lossy(&salida.stderr)
    );
    parse_vulkaninfo(&texto)
}

/// Parsea la salida de `vulkaninfo --summary`.
fn parse_vulkaninfo(texto: &str) -> Vec<VulkanDevice> {
    let mut dispositivos: Vec<VulkanDevice> = Vec::new();
    let mut api_global = String::new();
    let mut indice_actual: Option<usize> = None;
    let mut nombre = String::new();
    let mut driver = String::new();
    let mut api = String::new();
    let mut discreta = false;

    for linea in texto.lines() {
        let limpio = linea.trim();

        if limpio.starts_with("Vulkan Instance Version:") {
            api_global = limpio
                .split(':')
                .nth(1)
                .unwrap_or("")
                .trim()
                .trim_end_matches('.')
                .to_string();
            continue;
        }

        // Bloque "GPU0:"
        if limpio.starts_with("GPU") && limpio.ends_with(':') {
            let numero = limpio
                .trim_start_matches("GPU")
                .trim_end_matches(':')
                .trim();
            if let Ok(idx) = numero.parse::<usize>() {
                if let Some(prev) = indice_actual {
                    if !nombre.is_empty() {
                        dispositivos.push(VulkanDevice {
                            index: prev,
                            name: nombre.clone(),
                            driver: if driver.is_empty() {
                                api_global.clone()
                            } else {
                                driver.clone()
                            },
                            api_version: if api.is_empty() {
                                api_global.clone()
                            } else {
                                api.clone()
                            },
                            discrete: discreta,
                        });
                    }
                }
                indice_actual = Some(idx);
                nombre.clear();
                driver.clear();
                api.clear();
                discreta = false;
                continue;
            }
        }

        // Propiedades "clave = valor"
        if let Some((clave, valor)) = limpio.split_once('=') {
            let clave = clave.trim();
            let valor = valor.trim();
            match clave {
                "deviceName" => nombre = valor.to_string(),
                "driverName" => driver = valor.to_string(),
                "apiVersion" => api = valor.to_string(),
                "deviceType" => discreta = valor.contains("DISCRETE_GPU"),
                _ => {}
            }
        }
    }

    // Último dispositivo del volcado.
    if let Some(prev) = indice_actual {
        if !nombre.is_empty() {
            dispositivos.push(VulkanDevice {
                index: prev,
                name: nombre,
                driver: if driver.is_empty() {
                    api_global.clone()
                } else {
                    driver
                },
                api_version: if api.is_empty() { api_global } else { api },
                discrete: discreta,
            });
        }
    }

    dispositivos
}
/// Estado de Vulkan: si hay dispositivos enumerables, versión e ICD instalados.
fn estado_vulkan(dispositivos: &[VulkanDevice]) -> (bool, String, String, Vec<String>) {
    let mut icds: Vec<String> = Vec::new();

    let system32 = std::env::var("SystemRoot")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("C:\\Windows"))
        .join("System32");

    let loader = system32.join("vulkan-1.dll").is_file();

    for (fichero, etiqueta) in [
        ("nvidia_icd.json", "NVIDIA"),
        ("amd_icd.json", "AMD"),
        ("amd_icd64.json", "AMD"),
        ("intel_icd.json", "Intel"),
        ("intel_icd64.json", "Intel"),
    ] {
        if system32.join(fichero).is_file() && !icds.iter().any(|i| i.starts_with(etiqueta)) {
            icds.push(format!("{etiqueta} ({fichero})"));
        }
    }

    if !dispositivos.is_empty() {
        let version = dispositivos
            .first()
            .map(|d| d.api_version.clone())
            .unwrap_or_default();
        let lista: Vec<String> = dispositivos
            .iter()
            .map(|d| format!("GPU {} · {}", d.index, d.name))
            .collect();
        let detalle = format!("Vulkan {version} · {}", lista.join(", "));
        return (true, detalle, version, icds);
    }

    // Si el motor pudo sondear y no encontró ninguna GPU, no hay IA posible aunque
    // exista el cargador (vulkan-1.dll viene con Windows).
    if loader && ruta_motor_sonda().is_none() {
        return (
            true,
            "Cargador Vulkan presente (vulkan-1.dll); no se pudo comprobar con el motor"
                .to_string(),
            String::new(),
            icds,
        );
    }

    (
        false,
        "No se encontró Vulkan en el sistema".to_string(),
        String::new(),
        icds,
    )
}

// ---------------------------------------------------------------------------
// Sonda del motor: la fuente de verdad para `-g`
// ---------------------------------------------------------------------------

static MOTOR_SONDA: OnceLock<PathBuf> = OnceLock::new();

/// Indica qué `realesrgan-ncnn-vulkan` usar para sondear las GPU. Debe llamarse
/// antes de la primera detección; si no se llama se busca junto al ejecutable.
pub fn set_engine_probe(ruta: PathBuf) {
    let _ = MOTOR_SONDA.set(ruta);
}

fn ruta_motor_sonda() -> Option<PathBuf> {
    if let Some(p) = MOTOR_SONDA.get() {
        if p.is_file() {
            return Some(p.clone());
        }
    }
    let mut bases: Vec<PathBuf> = Vec::new();
    if let Ok(exe) = std::env::current_exe() {
        let mut dir = exe.parent().map(|p| p.to_path_buf());
        for _ in 0..6 {
            match dir {
                Some(d) => {
                    bases.push(d.clone());
                    dir = d.parent().map(|p| p.to_path_buf());
                }
                None => break,
            }
        }
    }
    if let Ok(cwd) = std::env::current_dir() {
        bases.push(cwd);
    }
    for base in bases {
        for rel in [
            "models/realesrgan-ncnn-vulkan.exe",
            "realesrgan-ncnn-vulkan.exe",
        ] {
            let cand = base.join(rel);
            if cand.is_file() {
                return Some(cand);
            }
        }
    }
    None
}

/// Ejecuta el propio motor con una entrada inexistente: ncnn enumera las GPU
/// Vulkan con el mismo índice que acepta `-g`. Funciona en cualquier PC con el
/// driver gráfico instalado, sin `vulkaninfo` ni el SDK de Vulkan.
fn detectar_motor() -> Vec<VulkanDevice> {
    let exe = match ruta_motor_sonda() {
        Some(p) => p,
        None => return Vec::new(),
    };
    let tmp = std::env::temp_dir().join("reescaller_probe_out.png");
    let mut hijo = match nuevo_cmd(&exe)
        .args(["-i", "reescaller_no_existe.png", "-o"])
        .arg(&tmp)
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
    {
        Ok(h) => h,
        Err(_) => return Vec::new(),
    };

    let mut stdout = hijo.stdout.take();
    let mut stderr = hijo.stderr.take();
    let lector_err = std::thread::spawn(move || {
        use std::io::Read;
        let mut buf = Vec::new();
        if let Some(e) = stderr.as_mut() {
            let _ = e.read_to_end(&mut buf);
        }
        buf
    });
    let lector_out = std::thread::spawn(move || {
        use std::io::Read;
        let mut buf = Vec::new();
        if let Some(o) = stdout.as_mut() {
            let _ = o.read_to_end(&mut buf);
        }
        buf
    });

    // Un driver roto puede colgar la inicialización: máximo 20 s.
    let inicio = std::time::Instant::now();
    loop {
        match hijo.try_wait() {
            Ok(Some(_)) => break,
            Ok(None) if inicio.elapsed().as_secs() >= 20 => {
                let _ = hijo.kill();
                let _ = hijo.wait();
                break;
            }
            Ok(None) => std::thread::sleep(std::time::Duration::from_millis(50)),
            Err(_) => break,
        }
    }
    let mut texto = String::from_utf8_lossy(&lector_err.join().unwrap_or_default()).to_string();
    texto.push('\n');
    texto.push_str(&String::from_utf8_lossy(
        &lector_out.join().unwrap_or_default(),
    ));
    parse_sonda_motor(&texto)
}

/// Parsea líneas del estilo `[0 AMD Radeon RX 6600]  queueC=1[4] ...`.
fn parse_sonda_motor(texto: &str) -> Vec<VulkanDevice> {
    let mut lista: Vec<VulkanDevice> = Vec::new();
    for linea in texto.lines() {
        let l = linea.trim();
        if !l.starts_with('[') || !l.contains("queueC=") {
            continue;
        }
        let Some(fin) = l.find(']') else { continue };
        let interior = &l[1..fin];
        let Some((idx, nombre)) = interior.split_once(' ') else {
            continue;
        };
        let Ok(index) = idx.parse::<usize>() else {
            continue;
        };
        if lista.iter().any(|d| d.index == index) {
            continue;
        }
        let nombre = nombre.trim().to_string();
        lista.push(VulkanDevice {
            index,
            discrete: !es_integrada(&nombre) && !es_software(&nombre),
            name: nombre,
            driver: String::new(),
            api_version: String::new(),
        });
    }
    lista
}

/// Combina la lista del motor (índices fiables) con la de `vulkaninfo`
/// (tipo de dispositivo y versión de la API).
fn combinar_vulkan(motor: Vec<VulkanDevice>, info: Vec<VulkanDevice>) -> Vec<VulkanDevice> {
    let mut base = if motor.is_empty() {
        info.clone()
    } else {
        motor
    };
    for dev in base.iter_mut() {
        if let Some(i) = info.iter().find(|i| nombres_coinciden(&i.name, &dev.name)) {
            dev.discrete = i.discrete;
            if dev.api_version.is_empty() {
                dev.api_version = i.api_version.clone();
            }
            if dev.driver.is_empty() {
                dev.driver = i.driver.clone();
            }
        }
    }
    // Dispositivos por software (llvmpipe, WARP...): inútiles para la IA.
    base.into_iter().filter(|d| !es_software(&d.name)).collect()
}

fn normalizar(nombre: &str) -> String {
    nombre
        .to_lowercase()
        .replace("(r)", "")
        .replace("(tm)", "")
        .chars()
        .filter(|c| c.is_alphanumeric())
        .collect()
}

/// Dos nombres de GPU (WMI vs Vulkan) coinciden si uno contiene al otro.
fn nombres_coinciden(a: &str, b: &str) -> bool {
    let (a, b) = (normalizar(a), normalizar(b));
    !a.is_empty() && !b.is_empty() && (a == b || a.contains(&b) || b.contains(&a))
}

fn es_software(nombre: &str) -> bool {
    let n = nombre.to_lowercase();
    [
        "llvmpipe",
        "swiftshader",
        "basic render",
        "software",
        "warp",
    ]
    .iter()
    .any(|p| n.contains(p))
}

/// Heurística para GPU integradas cuando no hay tipo de dispositivo Vulkan.
fn es_integrada(nombre: &str) -> bool {
    let n = nombre.to_lowercase();
    if n.contains("uhd graphics")
        || n.contains("hd graphics")
        || n.contains("iris")
        || n.contains("radeon(tm) graphics")
        || n.contains("radeon graphics")
        || n.contains("vega 3")
        || n.contains("vega 6")
        || n.contains("vega 7")
        || n.contains("vega 8")
        || n.contains("vega 10")
        || n.contains("vega 11")
    {
        return true;
    }
    // AMD Radeon 610M / 680M / 780M / 890M...
    n.split_whitespace()
        .last()
        .map(|t| {
            t.len() >= 4 && t.ends_with('m') && t[..t.len() - 1].chars().all(|c| c.is_ascii_digit())
        })
        .unwrap_or(false)
}

/// Argumentos `-g` del motor. Solo se pasan cuando el índice procede de una
/// enumeración Vulkan real (el orden de WMI no coincide con el de Vulkan).
pub fn engine_gpu_args(informe: &HardwareReport, indice: Option<usize>) -> Vec<String> {
    match indice {
        Some(i) if informe.vulkan_devices.iter().any(|d| d.index == i) => {
            vec!["-g".to_string(), i.to_string()]
        }
        _ => Vec::new(),
    }
}

/// Resuelve la preferencia de dispositivo contra el hardware real.
///
/// Devuelve el índice de GPU que se debe pasar al motor (`-g`) y la lista de
/// avisos (por ejemplo, "no hay ninguna AMD; se usa la automática").
pub fn resolve_preference(
    preferencia: crate::config::DevicePreference,
    informe: &HardwareReport,
    indice_manual: Option<usize>,
) -> (Option<usize>, Vec<String>) {
    use crate::config::DevicePreference;
    let mut avisos = Vec::new();

    if preferencia == DevicePreference::Cpu {
        return (None, avisos);
    }

    // El índice manual manda sobre todo lo demás.
    if let Some(idx) = indice_manual {
        if informe.gpus.iter().any(|g| g.index == idx) {
            return (Some(idx), avisos);
        }
        avisos.push(format!(
            "No existe la GPU {idx}; se usa la selección automática."
        ));
    }

    if preferencia == DevicePreference::Auto {
        if let Some(g) = informe.primary_gpu() {
            return (Some(g.index), avisos);
        }
        return (None, avisos);
    }

    let objetivo = match preferencia {
        DevicePreference::Nvidia => GpuVendor::Nvidia,
        DevicePreference::Amd => GpuVendor::Amd,
        DevicePreference::Intel => GpuVendor::Intel,
        _ => return (None, avisos),
    };

    match informe.index_of_vendor(objetivo) {
        Some(idx) => (Some(idx), avisos),
        None => {
            let disponibles: Vec<&str> = informe.gpus.iter().map(|g| g.vendor.label()).collect();
            avisos.push(format!(
                "No se detectó ninguna GPU {}. Detectadas: {}. Se usa la automática.",
                objetivo.label(),
                if disponibles.is_empty() {
                    "ninguna".to_string()
                } else {
                    disponibles.join(", ")
                }
            ));
            (informe.primary_gpu().map(|g| g.index), avisos)
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detecta_fabricante() {
        assert_eq!(
            GpuVendor::from_text("NVIDIA GeForce RTX 3060", "NVIDIA Corporation"),
            GpuVendor::Nvidia
        );
        assert_eq!(
            GpuVendor::from_text("AMD Radeon RX 6700 XT", "Advanced Micro Devices"),
            GpuVendor::Amd
        );
        assert_eq!(
            GpuVendor::from_text("Intel(R) UHD Graphics 770", "Intel Corporation"),
            GpuVendor::Intel
        );
        assert_eq!(
            GpuVendor::from_text("Intel Arc A770", "Intel Corporation"),
            GpuVendor::Intel
        );
    }

    #[test]
    fn parsea_json_de_gpus() {
        let json = r#"[{"Name":"AMD Radeon RX 6600","DriverVersion":"32.0.21045.16010","AdapterCompatibility":"Advanced Micro Devices, Inc.","AdapterRAM":4293918720},
                        {"Name":"Intel(R) UHD Graphics 770","DriverVersion":"31.0.101.5186","AdapterCompatibility":"Intel Corporation","AdapterRAM":1073741824}]"#;
        let gpus = parse_gpus_json(json);
        assert_eq!(gpus.len(), 2);
        assert_eq!(gpus[0].vendor, GpuVendor::Amd);
        assert_eq!(gpus[1].vendor, GpuVendor::Intel);
        assert!(!gpus[1].dedicated, "las iGPU no son dedicadas");
        assert!(gpus[0].vram_is_minimum, "4 GB se marca como mínimo");
    }

    #[test]
    fn parsea_vulkaninfo() {
        let salida = "Vulkan Instance Version: 1.4.309\n\nDevices:\n========\nGPU0:\n\tapiVersion         = 1.4.315\n\tdriverName         = AMD proprietary driver\n\tdeviceName         = AMD Radeon RX 6600\n\tdeviceType         = PHYSICAL_DEVICE_TYPE_DISCRETE_GPU\nGPU1:\n\tapiVersion         = 1.3.290\n\tdriverName         = Intel driver\n\tdeviceName         = Intel Arc A770\n\tdeviceType         = PHYSICAL_DEVICE_TYPE_DISCRETE_GPU\n";
        let devs = parse_vulkaninfo(salida);
        assert_eq!(devs.len(), 2);
        assert_eq!(devs[0].name, "AMD Radeon RX 6600");
        assert_eq!(devs[0].index, 0);
        assert!(devs[0].discrete);
        assert_eq!(devs[1].name, "Intel Arc A770");
        assert_eq!(devs[1].index, 1);
    }

    #[test]
    fn parsea_sonda_del_motor() {
        let salida = "[0 AMD Radeon RX 6600]  queueC=1[4]  queueG=0[8]  queueT=2[2]
[0 AMD Radeon RX 6600]  bugsbn1=0
[1 Intel(R) UHD Graphics 770]  queueC=0[1]  queueG=0[1]
decode image x failed
";
        let d = parse_sonda_motor(salida);
        assert_eq!(d.len(), 2);
        assert_eq!(d[0].name, "AMD Radeon RX 6600");
        assert!(d[0].discrete);
        assert_eq!(d[1].index, 1);
        assert!(!d[1].discrete, "UHD Graphics es integrada");
    }

    #[test]
    fn clasifica_integradas_y_software() {
        assert!(es_integrada("AMD Radeon(TM) Graphics"));
        assert!(es_integrada("AMD Radeon 780M"));
        assert!(es_integrada("Intel(R) Iris(R) Xe Graphics"));
        assert!(!es_integrada("AMD Radeon RX 7800 XT"));
        assert!(!es_integrada("NVIDIA GeForce RTX 4060"));
        assert!(!es_integrada("Intel(R) Arc(TM) A770 Graphics"));
        assert!(es_software("llvmpipe (LLVM 15.0.7, 256 bits)"));
        assert!(es_software("Microsoft Basic Render Driver"));
    }

    #[test]
    fn nombres_wmi_y_vulkan_coinciden() {
        assert!(nombres_coinciden(
            "AMD Radeon RX 6600",
            "AMD Radeon(TM) RX 6600"
        ));
        assert!(nombres_coinciden(
            "NVIDIA GeForce RTX 3060",
            "NVIDIA GeForce RTX 3060 Laptop GPU"
        ));
        assert!(!nombres_coinciden(
            "AMD Radeon RX 6600",
            "Intel(R) UHD Graphics 770"
        ));
    }

    #[test]
    fn combina_motor_y_vulkaninfo_sin_software() {
        let motor = parse_sonda_motor(
            "[0 llvmpipe (LLVM)]  queueC=0[1]
[1 NVIDIA GeForce RTX 3060]  queueC=2[8]
",
        );
        let r = combinar_vulkan(motor, Vec::new());
        assert_eq!(r.len(), 1);
        assert_eq!(r[0].index, 1);
    }

    #[test]
    fn argumento_g_solo_con_indice_vulkan_real() {
        let mut inf = detect();
        inf.vulkan_devices = parse_sonda_motor(
            "[0 Intel(R) UHD Graphics 770]  queueC=0[1]
[1 NVIDIA GeForce RTX 3060]  queueC=2[8]
",
        );
        assert_eq!(
            engine_gpu_args(&inf, Some(1)),
            vec!["-g".to_string(), "1".to_string()]
        );
        assert_eq!(
            engine_gpu_args(&inf, Some(0)),
            vec!["-g".to_string(), "0".to_string()]
        );
        assert!(engine_gpu_args(&inf, Some(7)).is_empty());
        assert!(engine_gpu_args(&inf, None).is_empty());
        inf.vulkan_devices.clear();
        assert!(
            engine_gpu_args(&inf, Some(1)).is_empty(),
            "sin enumeración Vulkan no se fuerza -g"
        );
    }

    #[test]
    fn informe_de_hardware_es_consistente() {
        let informe = detect();
        assert!(informe.logical_cores >= 1);
        assert!(informe.recommended_workers >= 1);
        if !informe.vulkan_available {
            assert_eq!(informe.engine_support, EngineSupport::CpuOnly);
        }
    }
}
