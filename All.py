"""
Herramienta de Mejora Visual
=============================
Aplicación de escritorio (CustomTkinter) para mejorar imágenes y videos con IA.

Motores utilizados:
  * Real-ESRGAN (realesrgan-ncnn-vulkan.exe)  -> mejora/escalado de imagen y video
  * GFPGAN models/GFPGANv1.4.pth               -> restauración de rostros
  * RestoreFormer++ models/RestoreFormer++.ckpt -> restauración de rostros
  * FFmpeg models/ffmpeg.exe                  -> extracción, fusión, conversión y recorte

Notas de la revisión (remodelación, F1–F12):
  F1  subprocess con shell=False
  F2  factor de escala completo ("4x" -> "4")
  F3  sin "except: pass" silencioso
  F4  messagebox/UI siempre en el hilo principal de Tk
  F5  el video final hereda el FPS real del origen
  F6  autodetección de ejecutables y modelos
  F7  configuración persistente en JSON
  F8  modelos de IA cargados una sola vez (cache)
  F9  límite de trabajos simultáneos (por defecto 2)
  F10 models/All.py (duplicado) eliminado
  F11 instalación de dependencias sin importlib.reload
  F12 requirements.txt + .gitignore
"""

import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
import subprocess
import os
import sys
import json
import shutil
import threading
import time
import re
import importlib
from concurrent.futures import ThreadPoolExecutor

# --- Dependencias imprescindibles -------------------------------------------
try:
    import customtkinter as ctk
except ImportError:
    raise SystemExit(
        "Falta la dependencia 'customtkinter'.\n"
        "Instala las dependencias con:  pip install -r requirements.txt"
    )

try:
    from PIL import Image, ImageTk
except ImportError:
    raise SystemExit(
        "Falta la dependencia 'Pillow'.\n"
        "Instala las dependencias con:  pip install -r requirements.txt"
    )

# --- Dependencias opcionales (se verifican en el uso) ------------------------
try:
    import cv2
except ImportError:
    cv2 = None

try:
    import numpy as np
except ImportError:
    np = None

# --- Rutas base de la aplicación --------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

# El paquete RestoreFormer vive en models/ y se importa como 'RestoreFormer.RestoreFormer'.
if os.path.isdir(MODELS_DIR) and MODELS_DIR not in sys.path:
    sys.path.insert(0, MODELS_DIR)


def _carpeta_config() -> str:
    """Carpeta donde se guarda la configuración persistente."""
    if os.name == "nt":
        base = os.environ.get("APPDATA") or os.path.expanduser("~")
    else:
        base = os.environ.get("XDG_CONFIG_HOME") or os.path.join(os.path.expanduser("~"), ".config")
    return os.path.join(base, "Reescaller")


RUTA_CONFIG = os.path.join(_carpeta_config(), "config.json")

# --- Rutas de ejecutables (se inicializan al inicio) -----------------------
RUTA_EJECUTABLE = ""
RUTA_FFMPEG = ""
RUTA_GFPGAN_PTH = ""
RUTA_RESTOREFORMER_PTH = ""

# --- Variables Globales de la Aplicación ------------------------------------
rutas_seleccionadas = {
    "imagen_unica": "",
    "carpeta_entrada": "",
    "carpeta_salida_img": "",
    "video": "",
    "video_convertir": "",
    "video_recortar": ""
}

# --- Configuración Personalizable ------------------------------------------
class Config:
    """Configuración de la aplicación. Se persiste en JSON (F7)."""

    factor_escala = "4x"
    modelo = "realesrgan-x4plus"
    formato_salida = "png"
    eliminar_temporales = True
    tema = "Dark"
    max_workers = 2  # F9: no saturar la GPU con nuevos procesos Vulkan

    @classmethod
    def to_dict(cls):
        return {
            "factor_escala": cls.factor_escala,
            "modelo": cls.modelo,
            "formato_salida": cls.formato_salida,
            "eliminar_temporales": cls.eliminar_temporales,
            "tema": cls.tema,
            "max_workers": cls.max_workers,
            "rutas": {
                "realesrgan": RUTA_EJECUTABLE,
                "ffmpeg": RUTA_FFMPEG,
                "gfpgan": RUTA_GFPGAN_PTH,
                "restoreformer": RUTA_RESTOREFORMER_PTH,
            },
        }

    @classmethod
    def save(cls, ruta=None):
        """Guarda la configuración en disco. Nunca lanza excepciones."""
        destino = ruta or RUTA_CONFIG
        try:
            os.makedirs(os.path.dirname(destino), exist_ok=True)
            with open(destino, "w", encoding="utf-8") as f:
                json.dump(cls.to_dict(), f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            print(f"[config] No se pudo guardar la configuración: {e}")
            return False

    @classmethod
    def load(cls, ruta=None):
        """Carga la configuración desde disco. Si no existe, la deja intacta."""
        global RUTA_EJECUTABLE, RUTA_FFMPEG, RUTA_GFPGAN_PTH, RUTA_RESTOREFORMER_PTH
        origen = ruta or RUTA_CONFIG
        if not os.path.isfile(origen):
            return False
        try:
            with open(origen, "r", encoding="utf-8") as f:
                datos = json.load(f)
        except Exception as e:
            print(f"[config] No se pudo leer la configuración: {e}")
            return False

        cls.factor_escala = datos.get("factor_escala", cls.factor_escala)
        cls.modelo = datos.get("modelo", cls.modelo)
        cls.formato_salida = datos.get("formato_salida", cls.formato_salida)
        cls.eliminar_temporales = bool(datos.get("eliminar_temporales", cls.eliminar_temporales))
        cls.tema = datos.get("tema", cls.tema)
        try:
            cls.max_workers = max(1, int(datos.get("max_workers", cls.max_workers)))
        except (TypeError, ValueError):
            pass

        rutas = datos.get("rutas", {}) or {}
        RUTA_EJECUTABLE = rutas.get("realesrgan", RUTA_EJECUTABLE) or ""
        RUTA_FFMPEG = rutas.get("ffmpeg", RUTA_FFMPEG) or ""
        RUTA_GFPGAN_PTH = rutas.get("gfpgan", RUTA_GFPGAN_PTH) or ""
        RUTA_RESTOREFORMER_PTH = rutas.get("restoreformer", RUTA_RESTOREFORMER_PTH) or ""
        return True


# --- Utilidades de hilo / UI ------------------------------------------------
def _en_hilo_principal(func, *args, **kwargs):
    """Ejecuta func en el hilo principal de Tkinter de forma segura (F4)."""
    if "ventana" in globals():
        try:
            ventana.after(0, lambda: func(*args, **kwargs))
            return True
        except tk.TclError:
            pass
    return False


def mostrar_error(titulo, texto):
    """messagebox de error siempre en el hilo principal."""
    _en_hilo_principal(messagebox.showerror, titulo, texto)


def mostrar_info(titulo, texto):
    """messagebox informativo siempre en el hilo principal."""
    _en_hilo_principal(messagebox.showinfo, titulo, texto)


# --- Funciones de la Interfaz de Usuario ---
def log_message(message, color="white"):
    """Inserta un mensaje en el área de registro (seguro desde hilos secundarios)."""

    def _insertar():
        try:
            log_text.configure(state="normal")
            log_text.insert(tk.END, message + "\n", color)
            log_text.configure(state="disabled")
            log_text.see(tk.END)
        except tk.TclError:
            pass

    if not _en_hilo_principal(_insertar):
        _insertar()


def set_progress(value):
    """Actualiza la barra de progreso (seguro desde hilos secundarios)."""
    try:
        valor = max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return

    def _actualizar():
        try:
            barra_progreso.set(valor)
        except tk.TclError:
            pass

    if not _en_hilo_principal(_actualizar):
        _actualizar()


def toggle_buttons_state(state):
    """Habilita o deshabilita todos los botones principales."""
    buttons = [
        boton_seleccionar_unico, boton_mejorar_unico,
        boton_seleccionar_gfpgan,
        boton_seleccionar_entrada, boton_seleccionar_salida, boton_mejorar_carpeta,
        boton_seleccionar_video, boton_iniciar_video,
        boton_seleccionar_video_convertir, boton_convertir_video_iniciar,
        boton_seleccionar_video_recortar, boton_recortar_video_iniciar
    ]

    def _aplicar():
        for button in buttons:
            try:
                button.configure(state=state)
            except tk.TclError:
                pass

    if not _en_hilo_principal(_aplicar):
        _aplicar()


def proceso_ejecutar(comando, mensaje_progreso, paso_actual=0, total_pasos=1, timeout=None):
    """Función genérica para ejecutar comandos con manejo de errores y progreso."""
    try:
        log_message(f"Iniciando: {mensaje_progreso}", "blue")
        set_progress((paso_actual / total_pasos) * 100)

        # F1: shell=False (lista de argumentos, sin intérprete de comandos).
        proceso = subprocess.run(
            comando,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            shell=False,
            timeout=timeout
        )

        log_message("Comando ejecutado con éxito.", "green")
        if proceso.stdout and proceso.stdout.strip():
            log_message(f"Salida:\n{proceso.stdout.strip()}", "green")
        set_progress(((paso_actual + 1) / total_pasos) * 100)
        return proceso

    except subprocess.TimeoutExpired:
        log_message(f"Tiempo de espera agotado (> {timeout}s): {mensaje_progreso}", "red")
        set_progress(0)
        raise
    except subprocess.CalledProcessError as e:
        detalle = (e.stderr or e.stdout or "").strip()
        log_message(f"Error en la ejecución:\n{detalle}", "red")
        set_progress(0)
        raise
    except FileNotFoundError as e:
        log_message(f"Error de archivo: '{e.filename}' no se encontró.", "red")
        set_progress(0)
        raise
    except Exception as e:
        log_message(f"Ocurrió un error inesperado:\n{e}", "red")
        set_progress(0)
        raise


def _escala_numerica():
    """F2: convierte '4x' en '4' (no trunca por el primer carácter)."""
    return str(Config.factor_escala).strip().rstrip("xX") or "4"


# --- Lógica de las pestañas ---
def seleccionar_archivo(tipo):
    ruta = filedialog.askopenfilename(
        title=f"Selecciona un archivo de {tipo}",
        filetypes=(("Archivos de imagen", "*.jpg *.png *.jpeg"), ("Archivos de video", "*.mp4 *.avi *.mov *.mkv *.webm"), ("Todos los archivos", "*.*"))
    )
    if ruta:
        rutas_seleccionadas[tipo] = ruta
        if tipo == "imagen_unica":
            etiqueta_ruta_unico.configure(text=f"Archivo: {os.path.basename(ruta)}")
        elif tipo == "video":
            etiqueta_ruta_video.configure(text=f"Video seleccionado: {os.path.basename(ruta)}")
        elif tipo == "video_convertir":
            etiqueta_ruta_convertir.configure(text=f"Video a convertir: {os.path.basename(ruta)}")
        elif tipo == "video_recortar":
            etiqueta_ruta_recortar.configure(text=f"Video a recortar: {os.path.basename(ruta)}")
        log_message(f"Archivo seleccionado: {ruta}")

def seleccionar_carpeta(tipo):
    ruta = filedialog.askdirectory(title=f"Selecciona la carpeta de {tipo}")
    if ruta:
        rutas_seleccionadas[tipo] = ruta
        if tipo == "carpeta_entrada":
            etiqueta_entrada_carpeta.configure(text=f"Carpeta de entrada: {ruta}")
        elif tipo == "carpeta_salida_img":
            etiqueta_salida_carpeta.configure(text=f"Carpeta de salida: {ruta}")
        log_message(f"Carpeta seleccionada: {ruta}")

def ejecutar_realesrgan_unico():
    if not RUTA_EJECUTABLE:
        mostrar_error("Error", "Por favor, configura las rutas de los ejecutables en la pestaña de Configuración.")
        return
    if not rutas_seleccionadas["imagen_unica"]:
        log_message("Error: Por favor, selecciona una imagen primero.", "red")
        return

    def worker():
        toggle_buttons_state("disabled")
        try:
            nombre_base, _ = os.path.splitext(os.path.basename(rutas_seleccionadas["imagen_unica"]))
            ruta_salida = os.path.join(os.path.dirname(rutas_seleccionadas["imagen_unica"]), f"{nombre_base}_mejorado.{Config.formato_salida}")
            comando = [
                RUTA_EJECUTABLE,
                "-i", rutas_seleccionadas["imagen_unica"],
                "-o", ruta_salida,
                "-s", _escala_numerica(),
                "-n", Config.modelo
            ]
            proceso_ejecutar(comando, "Mejorando imagen...")
            log_message(f"\n¡Éxito! Imagen mejorada y guardada en:\n{ruta_salida}", "green")
            mostrar_info("Proceso Completo", "¡La mejora de la imagen se completó con éxito!")
        except Exception as e:
            # F3: no silenciar el error, se informa al usuario.
            log_message(f"Ocurrió un error inesperado al mejorar la imagen:\n{e}", "red")
            mostrar_error("Error", f"No se pudo mejorar la imagen:\n{e}")
        finally:
            toggle_buttons_state("normal")

    threading.Thread(target=worker, daemon=True).start()

# --- Lógica de la restauración facial con selector de modelo (GFPGAN y RestoreFormer++) ---
# F8: cache de modelos para no recargar los pesos en cada llamada.
_MODELOS_CARGADOS = {}


def _parche_torchvision():
    """Aplica el parche de compatibilidad habitual para basicsr/gfpgan.

    torchvision >= 0.17 eliminó el módulo `torchvision.transforms.functional_tensor`,
    lo que rompe la importación de basicsr/gfpgan. Se crea un módulo sustituto
    que reenvía a `torchvision.transforms.functional`.
    """
    try:
        import types
        import torch  # noqa: F401
        import torchvision
        import torchvision.transforms.functional as _F
    except Exception:
        return False

    try:
        if not hasattr(torchvision.transforms, "functional_tensor"):
            shim = types.ModuleType("torchvision.transforms.functional_tensor")
            for nombre in dir(_F):
                if not nombre.startswith("_"):
                    try:
                        setattr(shim, nombre, getattr(_F, nombre))
                    except Exception:
                        pass
            if not hasattr(shim, "rgb_to_grayscale") and hasattr(shim, "to_grayscale"):
                shim.rgb_to_grayscale = shim.to_grayscale
            if not hasattr(shim, "tensor_to_image"):
                def tensor_to_image(tensor, rgb2bgr=False, out_type=None):
                    import numpy as _np
                    if out_type is None:
                        out_type = _np.uint8
                    imagen = tensor.detach().cpu().float().numpy()
                    if imagen.ndim == 3 and imagen.shape[0] in (1, 3):
                        imagen = imagen.transpose(1, 2, 0)
                    if imagen.shape[-1] == 1:
                        imagen = imagen[:, :, 0]
                    if rgb2bgr and imagen.ndim == 3 and imagen.shape[-1] == 3:
                        imagen = imagen[:, :, ::-1]
                    return (imagen * 255.0).round().astype(out_type)
                shim.tensor_to_image = tensor_to_image
            sys.modules["torchvision.transforms.functional_tensor"] = shim
            torchvision.transforms.functional_tensor = shim
        return True
    except Exception:
        return False


def _importar_dependencias_ia():
    """Importa (o reimporta) torch/gfpgan/basicsr tras una instalación."""
    _parche_torchvision()
    torch = importlib.import_module("torch")
    gfpgan = importlib.import_module("gfpgan")
    return torch, gfpgan


# Se aplica al importar el módulo para que cualquier import de basicsr/gfpgan/facexlib
# funcione, incluso si alguien los importa antes que esta aplicación.
_parche_torchvision()


def restaurar_cara():
    ruta_imagen = filedialog.askopenfilename(
        title="Selecciona una imagen para restaurar rostros",
        filetypes=(("Archivos de imagen", "*.jpg *.png *.jpeg"), ("Todos los archivos", "*.*"))
    )
    if not ruta_imagen:
        return

    modelo_seleccionado = modelo_restauracion_var.get()

    def worker():
        toggle_buttons_state("disabled")
        try:
            if modelo_seleccionado == "GFPGAN":
                ejecutar_gfpgan(ruta_imagen)
            elif modelo_seleccionado == "RestoreFormer++":
                ejecutar_restoreformer(ruta_imagen)
            else:
                log_message(f"Modelo de restauración desconocido: {modelo_seleccionado}", "red")
        except Exception as e:
            log_message(f"Ocurrió un error inesperado al usar el modelo:\n{e}", "red")
            mostrar_error("Error", f"No se pudo restaurar la imagen:\n{e}")
        finally:
            toggle_buttons_state("normal")

    threading.Thread(target=worker, daemon=True).start()

def ejecutar_gfpgan(ruta_imagen):
    """Lógica de GFPGAN (con modelo cacheado)."""
    try:
        if not RUTA_GFPGAN_PTH or not os.path.exists(RUTA_GFPGAN_PTH):
            log_message("Advertencia: No se encontró el modelo GFPGAN. Por favor, configúralo en la pestaña de Configuración.", "red")
            mostrar_error("Error", "No se encontró el modelo GFPGAN (GFPGANv1.4.pth). Configúralo en la pestaña de Configuración.")
            return

        log_message("Iniciando restauración de rostros con GFPGAN...", "blue")
        set_progress(25)

        if "gfpgan" not in _MODELOS_CARGADOS:
            try:
                torch, gfpgan = _importar_dependencias_ia()
            except ImportError:
                log_message("PyTorch y GFPGAN no están instalados. Instalando dependencias...", "yellow")
                if not instalar_dependencias_pytorch():
                    return
                try:
                    torch, gfpgan = _importar_dependencias_ia()
                except ImportError as e:
                    log_message(f"La instalación de PyTorch/GFPGAN falló: {e}", "red")
                    mostrar_error("Error", "No se pudieron instalar PyTorch y GFPGAN. Revisa tu conexión a internet.")
                    return

            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            from gfpgan import GFPGANer
            restaurador = GFPGANer(model_path=RUTA_GFPGAN_PTH, upscale=2, bg_upsampler=None, device=device)
            _MODELOS_CARGADOS["gfpgan"] = restaurador
            log_message(f"Modelo GFPGAN cargado ({device}).", "green")
        else:
            restaurador = _MODELOS_CARGADOS["gfpgan"]
            log_message("Modelo GFPGAN reutilizado desde caché.", "green")

        set_progress(50)

        if cv2 is None:
            raise RuntimeError("OpenCV no está instalado. Ejecuta: pip install opencv-python")

        imagen_input = cv2.imread(ruta_imagen, cv2.IMREAD_COLOR)
        if imagen_input is None:
            raise RuntimeError(f"No se pudo leer la imagen: {ruta_imagen}")

        _, _, imagen_salida = restaurador.enhance(imagen_input, has_aligned=False, only_center_face=False)

        nombre_base, _ = os.path.splitext(os.path.basename(ruta_imagen))
        ruta_salida_gfpgan = os.path.join(os.path.dirname(ruta_imagen), f"{nombre_base}_gfpgan.png")
        cv2.imwrite(ruta_salida_gfpgan, imagen_salida)

        log_message(f"\n¡Éxito! Imagen con rostros restaurados (GFPGAN) guardada en:\n{ruta_salida_gfpgan}", "green")
        set_progress(100)
        mostrar_info("Proceso Completo", "¡La restauración de rostros se completó con éxito!")

    except Exception as e:
        log_message(f"Ocurrió un error inesperado al usar GFPGAN:\n{e}", "red")
        set_progress(0)
        mostrar_error("Error", f"Error en GFPGAN:\n{e}")

def ejecutar_restoreformer(ruta_imagen):
    """Lógica para RestoreFormer++ integrada (con modelo cacheado)."""
    try:
        if not RUTA_RESTOREFORMER_PTH or not os.path.exists(RUTA_RESTOREFORMER_PTH):
            log_message("Advertencia: No se encontró el modelo RestoreFormer++. Por favor, configúralo en la pestaña de Configuración.", "red")
            mostrar_error("Error", "No se encontró el modelo RestoreFormer++ (RestoreFormer++.ckpt). Configúralo en la pestaña de Configuración.")
            return

        log_message("Iniciando restauración de rostros con RestoreFormer++...", "blue")
        set_progress(25)

        if "restoreformer" not in _MODELOS_CARGADOS:
            try:
                _parche_torchvision()
                rf_modulo = importlib.import_module("RestoreFormer.RestoreFormer")
                torch = importlib.import_module("torch")
                importlib.import_module("basicsr.utils")
                if cv2 is None:
                    raise ImportError("opencv-python")
            except ImportError as e:
                log_message(f"Error: La librería de RestoreFormer o sus dependencias no están instaladas: {getattr(e, 'name', e)}", "red")
                log_message("Por favor, instala las dependencias necesarias con 'pip install basicsr'", "yellow")
                mostrar_error("Error", "Faltan dependencias de RestoreFormer++.\nEjecuta: pip install basicsr opencv-python")
                return

            restaurador = rf_modulo.RestoreFormer(
                model_path=RUTA_RESTOREFORMER_PTH,
                upscale=2,
                arch='RestoreFormer++',
                bg_upsampler=None
            )
            _MODELOS_CARGADOS["restoreformer"] = restaurador
            log_message("Modelo RestoreFormer++ cargado.", "green")
        else:
            restaurador = _MODELOS_CARGADOS["restoreformer"]
            log_message("Modelo RestoreFormer++ reutilizado desde caché.", "green")

        set_progress(50)

        input_img = cv2.imread(ruta_imagen, cv2.IMREAD_COLOR)
        if input_img is None:
            raise RuntimeError(f"No se pudo leer la imagen: {ruta_imagen}")

        cropped_faces, restored_faces, restored_img = restaurador.enhance(
            input_img,
            has_aligned=False,
            only_center_face=False,
            paste_back=True
        )

        if restored_img is not None:
            from basicsr.utils import imwrite
            nombre_base, _ = os.path.splitext(os.path.basename(ruta_imagen))
            ruta_salida_rf = os.path.join(os.path.dirname(ruta_imagen), f"{nombre_base}_rf++.png")
            imwrite(restored_img, ruta_salida_rf)
            log_message(f"\n¡Éxito! Imagen con rostros restaurados (RestoreFormer++) guardada en:\n{ruta_salida_rf}", "green")
            set_progress(100)
            mostrar_info("Proceso Completo", "¡La restauración de rostros se completó con éxito!")
        else:
            log_message("No se encontraron rostros en la imagen o la restauración falló.", "yellow")
            set_progress(0)

    except Exception as e:
        log_message(f"Ocurrió un error inesperado al usar RestoreFormer++:\n{e}", "red")
        set_progress(0)
        mostrar_error("Error", f"Error en RestoreFormer++:\n{e}")

def ejecutar_realesrgan_carpeta():
    if not RUTA_EJECUTABLE:
        mostrar_error("Error", "Por favor, configura las rutas de los ejecutables en la pestaña de Configuración.")
        return
    if not rutas_seleccionadas["carpeta_entrada"] or not rutas_seleccionadas["carpeta_salida_img"]:
        log_message("Error: Por favor, selecciona ambas carpetas.", "red")
        return

    def worker():
        toggle_buttons_state("disabled")
        try:
            input_folder = rutas_seleccionadas["carpeta_entrada"]
            output_folder = rutas_seleccionadas["carpeta_salida_img"]

            archivos = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            if not archivos:
                log_message("No se encontraron imágenes en la carpeta de entrada.", "yellow")
                mostrar_info("Sin imágenes", "No se encontraron imágenes en la carpeta de entrada.")
                return

            total_archivos = len(archivos)
            procesados = 0
            escala = _escala_numerica()
            modelo = Config.modelo
            formato = Config.formato_salida

            # F9: no saturar la GPU con un proceso Vulkan por núcleo de CPU.
            workers = max(1, min(int(Config.max_workers), 4))

            def _tarea(f):
                return proceso_ejecutar(
                    [
                        RUTA_EJECUTABLE,
                        "-i", os.path.join(input_folder, f),
                        "-o", os.path.join(output_folder, f"{os.path.splitext(f)[0]}_mejorado.{formato}"),
                        "-s", escala,
                        "-n", modelo
                    ],
                    f"Procesando {f}...",
                    paso_actual=0,
                    total_pasos=1
                )

            with ThreadPoolExecutor(max_workers=workers) as executor:
                for _ in executor.map(_tarea, archivos):
                    procesados += 1
                    set_progress((procesados / total_archivos) * 100)

            log_message(f"\n¡Éxito! Se mejoraron {total_archivos} imágenes y se guardaron en:\n{output_folder}", "green")
            mostrar_info("Proceso Completo", f"¡La mejora de {total_archivos} imágenes se completó con éxito!")
        except Exception as e:
            log_message(f"Ocurrió un error inesperado al procesar la carpeta:\n{e}", "red")
            mostrar_error("Error", f"No se pudo procesar la carpeta:\n{e}")
        finally:
            toggle_buttons_state("normal")

    threading.Thread(target=worker, daemon=True).start()

def detectar_fps(ruta_video):
    """F5: obtiene el FPS real del video de origen para no desincronizar el audio."""
    try:
        proc = subprocess.run(
            [RUTA_FFMPEG, "-hide_banner", "-i", ruta_video],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            shell=False, timeout=120
        )
        salida = f"{proc.stderr or ''}\n{proc.stdout or ''}"
        m = re.search(r"Video:.*?([\d.]+)\s*fps", salida)
        if m:
            return float(m.group(1))
        m = re.search(r"([\d.]+)\s*tbr", salida)
        if m:
            return float(m.group(1))
    except Exception:
        pass
    return None

def ejecutar_proceso_video():
    if not RUTA_EJECUTABLE or not RUTA_FFMPEG:
        mostrar_error("Error", "Por favor, configura las rutas de los ejecutables en la pestaña de Configuración.")
        return
    if not rutas_seleccionadas["video"]:
        log_message("Error: Por favor, selecciona un video primero.", "red")
        return

    def worker():
        toggle_buttons_state("disabled")
        try:
            base_dir = os.path.dirname(rutas_seleccionadas["video"])
            tmp_frames_dir = os.path.join(base_dir, "tmp_frames")
            out_frames_dir = os.path.join(base_dir, "out_frames")
            nombre_video_final = f"{os.path.splitext(os.path.basename(rutas_seleccionadas['video']))[0]}_mejorado.mp4"
            ruta_video_final = os.path.join(base_dir, nombre_video_final)

            # 1. Extraer frames
            os.makedirs(tmp_frames_dir, exist_ok=True)
            comando_extract = [
                RUTA_FFMPEG, "-y", "-i", rutas_seleccionadas["video"], "-qscale:v", "1", "-qmin", "1", "-qmax", "1", "-vsync", "0",
                os.path.join(tmp_frames_dir, "frame%08d.jpg")
            ]
            proceso_ejecutar(comando_extract, "Paso 1/3: Extrayendo frames del video...", 0, 3)

            # 2. Mejorar frames
            os.makedirs(out_frames_dir, exist_ok=True)
            comando_enhance = [
                RUTA_EJECUTABLE, "-i", tmp_frames_dir, "-o", out_frames_dir,
                "-n", Config.modelo,
                "-s", _escala_numerica(),
                "-f", "jpg"
            ]
            proceso_ejecutar(comando_enhance, "Paso 2/3: Mejorando los frames...", 1, 3)

            # 3. Fusionar frames y audio (heredando el FPS real del video original)
            fps = detectar_fps(rutas_seleccionadas["video"])
            comando_merge = [
                RUTA_FFMPEG, "-y",
                "-i", os.path.join(out_frames_dir, "frame%08d.jpg"), "-i", rutas_seleccionadas["video"],
                "-map", "0:v:0", "-map", "1:a:0?",
                "-c:a", "copy", "-c:v", "libx264",
                "-pix_fmt", "yuv420p"
            ]
            if fps:
                comando_merge += ["-r", f"{fps:g}"]
                log_message(f"FPS detected del video original: {fps:g}", "blue")
            else:
                log_message("No se pudo detectar el FPS; se usará el de FFmpeg por defecto.", "yellow")
            comando_merge.append(ruta_video_final)
            proceso_ejecutar(comando_merge, "Paso 3/3: Fusionando frames y audio...", 2, 3)

            log_message(f"\n¡Proceso Completado! El video mejorado se guardó en:\n{ruta_video_final}", "green")
            mostrar_info("Proceso Completo", "¡La mejora del video se completó con éxito!")

            # 4. Eliminar archivos temporales
            if Config.eliminar_temporales:
                log_message("Eliminando archivos temporales...", "blue")
                try:
                    for carpeta in (tmp_frames_dir, out_frames_dir):
                        if os.path.isdir(carpeta):
                            shutil.rmtree(carpeta)
                    log_message("Archivos temporales eliminados con éxito.", "green")
                except Exception as e:
                    log_message(f"Error al eliminar archivos temporales: {e}", "red")

        except Exception as e:
            log_message(f"Ocurrió un error inesperado en el proceso de video:\n{e}", "red")
            mostrar_error("Error", f"No se pudo mejorar el video:\n{e}")
        finally:
            toggle_buttons_state("normal")

    threading.Thread(target=worker, daemon=True).start()

def convertir_video():
    if not RUTA_FFMPEG:
        mostrar_error("Error", "Por favor, configura la ruta de FFmpeg en la pestaña de Configuración.")
        return
    if not rutas_seleccionadas["video_convertir"]:
        log_message("Error: Por favor, selecciona un video para convertir.", "red")
        return

    def worker():
        toggle_buttons_state("disabled")
        try:
            ruta_video = rutas_seleccionadas["video_convertir"]
            formato_salida = formato_video_var.get()
            nombre_base, _ = os.path.splitext(os.path.basename(ruta_video))
            ruta_salida = os.path.join(os.path.dirname(ruta_video), f"{nombre_base}_convertido.{formato_salida}")

            comando = [
                RUTA_FFMPEG, "-y",
                "-i", ruta_video,
                "-vcodec", "copy",
                "-acodec", "copy",
                ruta_salida
            ]

            proceso_ejecutar(comando, f"Convirtiendo a {formato_salida}...")

            log_message(f"\n¡Éxito! Video convertido y guardado en:\n{ruta_salida}", "green")
            mostrar_info("Proceso Completo", f"¡La conversión a {formato_salida} se completó con éxito!")

        except Exception as e:
            log_message(f"Ocurrió un error inesperado al convertir el video:\n{e}", "red")
            mostrar_error("Error", f"No se pudo convertir el video:\n{e}")
        finally:
            toggle_buttons_state("normal")

    threading.Thread(target=worker, daemon=True).start()

def recortar_video():
    if not RUTA_FFMPEG:
        mostrar_error("Error", "Por favor, configura la ruta de FFmpeg en la pestaña de Configuración.")
        return
    if not rutas_seleccionadas["video_recortar"]:
        log_message("Error: Por favor, selecciona un video para recortar.", "red")
        return

    # F4: los diálogos se piden en el hilo principal, antes de lanzar el worker.
    tiempo_inicio = simpledialog.askstring("Recortar Video", "Ingresa el tiempo de inicio (hh:mm:ss):", parent=ventana)
    if not tiempo_inicio:
        log_message("Proceso cancelado por el usuario.", "yellow")
        return
    duracion = simpledialog.askstring("Recortar Video", "Ingresa la duración (en segundos):", parent=ventana)
    if not duracion:
        log_message("Proceso cancelado por el usuario.", "yellow")
        return

    def worker():
        toggle_buttons_state("disabled")
        try:
            ruta_video = rutas_seleccionadas["video_recortar"]
            nombre_base, _ = os.path.splitext(os.path.basename(ruta_video))
            ruta_salida = os.path.join(os.path.dirname(ruta_video), f"{nombre_base}_recortado.mp4")

            comando = [
                RUTA_FFMPEG, "-y",
                "-ss", tiempo_inicio,
                "-i", ruta_video,
                "-t", duracion,
                "-c", "copy",
                ruta_salida
            ]
            proceso_ejecutar(comando, "Recortando video...")

            log_message(f"\n¡Éxito! Video recortado y guardado en:\n{ruta_salida}", "green")
            mostrar_info("Proceso Completo", "¡El video se recortó con éxito!")

        except Exception as e:
            log_message(f"Ocurrió un error inesperado al recortar el video:\n{e}", "red")
            mostrar_error("Error", f"No se pudo recortar el video:\n{e}")
        finally:
            toggle_buttons_state("normal")

    threading.Thread(target=worker, daemon=True).start()

# --- Funciones para el Visualizador de Video ---
video_reproductor = None
_reproduccion_activa = False


def detener_reproduccion():
    """Detiene la reproducción del visualizador de forma segura."""
    global _reproduccion_activa, video_reproductor
    _reproduccion_activa = False
    LOGGER = video_reproductor
    if LOGGER is not None:
        try:
            LOGGER.release()
        except Exception:
            pass
    video_reproductor = None
    _en_hilo_principal(etiqueta_visualizador.configure, image=None)
    log_message("Reproducción detenida.", "yellow")


def abrir_y_reproducir():
    global video_reproductor, _reproduccion_activa
    if not rutas_seleccionadas["video_recortar"]:
        log_message("Por favor, selecciona un video primero.", "red")
        return
    if cv2 is None:
        log_message("OpenCV no está instalado. Ejecuta: pip install opencv-python", "red")
        mensaje = "Falta OpenCV. Instálalo con:  pip install opencv-python"
        mostrar_error("Error", mensaje)
        return

    detener_reproduccion()

    video_reproductor = cv2.VideoCapture(rutas_seleccionadas["video_recortar"])
    if not video_reproductor.isOpened():
        log_message("Error: No se pudo abrir el archivo de video.", "red")
        mostrar_error("Error", "No se pudo abrir el archivo de video.")
        video_reproductor = None
        return

    _reproduccion_activa = True
    threading.Thread(target=reproducir_video, daemon=True).start()

def reproducir_video():
    global video_reproductor
    if video_reproductor is None:
        return

    ancho_max = 500
    alto_max = 300

    while _reproduccion_activa and video_reproductor is not None:
        ret, frame = video_reproductor.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)

        ancho_original, alto_original = img.size
        ratio = min(ancho_max / ancho_original, alto_max / alto_original)
        nuevo_ancho = max(1, int(ancho_original * ratio))
        nuevo_alto = max(1, int(alto_original * ratio))

        img_redimensionada = img.resize((nuevo_ancho, nuevo_alto), Image.Resampling.LANCZOS)
        img_tk = ctk.CTkImage(light_image=img_redimensionada, dark_image=img_redimensionada, size=(nuevo_ancho, nuevo_alto))

        # F4/A4: la actualización del widget se agenda en el hilo principal.
        def _mostrar(imagen=img_tk):
            try:
                etiqueta_visualizador.configure(image=imagen)
                etiqueta_visualizador.image = imagen
            except tk.TclError:
                pass

        _en_hilo_principal(_mostrar)
        time.sleep(1 / 30.0)

    detener_reproduccion()
    log_message("Reproducción finalizada.", "yellow")

def instalar_dependencias_pytorch():
    """Instala las dependencias de IA. Tras la instalación reimporta los módulos (F11)."""
    try:
        log_message("Iniciando la instalación de dependencias...", "blue")
        dependencias = [
            "torch", "torchvision", "basicsr", "gfpgan", "opencv-python", "numpy"
        ]
        log_message("Instalando dependencias de PyTorch y GFPGAN...")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install'] + dependencias)
        # F11: se reimporta con importlib (reload no funciona con torch).
        importlib.import_module("torch")
        importlib.import_module("gfpgan")
        _parche_torchvision()
        log_message("\n¡Instalación completada con éxito!", "green")
        return True
    except subprocess.CalledProcessError as e:
        log_message(f"La instalación falló. Código de error: {e.returncode}", "red")
        if e.stderr:
            log_message(str(e.stderr).strip(), "red")
        mostrar_error("Error", "La instalación de las dependencias falló. Revisa tu conexión a internet.")
        return False
    except Exception as e:
        log_message(f"Ocurrió un error inesperado durante la instalación:\n{e}", "red")
        mostrar_error("Error", f"Error durante la instalación de dependencias:\n{e}")
        return False

# --- Funciones de Configuración ---
def configurar_rutas():
    global RUTA_EJECUTABLE, RUTA_FFMPEG, RUTA_GFPGAN_PTH, RUTA_RESTOREFORMER_PTH

    ruta_realesrgan = filedialog.askopenfilename(
        title="Selecciona realesrgan-ncnn-vulkan.exe",
        filetypes=(("Ejecutables", "*.exe"), ("Todos los archivos", "*.*"))
    )
    if ruta_realesrgan:
        RUTA_EJECUTABLE = ruta_realesrgan
        etiqueta_realesrgan.configure(text=f"Real-ESRGAN: {os.path.basename(ruta_realesrgan)}")

    ruta_ffmpeg = filedialog.askopenfilename(
        title="Selecciona ffmpeg.exe",
        filetypes=(("Ejecutables", "*.exe"), ("Todos los archivos", "*.*"))
    )
    if ruta_ffmpeg:
        RUTA_FFMPEG = ruta_ffmpeg
        etiqueta_ffmpeg.configure(text=f"FFmpeg: {os.path.basename(ruta_ffmpeg)}")

    ruta_gfpgan = filedialog.askopenfilename(
        title="Selecciona el modelo GFPGANv1.4.pth",
        filetypes=(("Archivos de modelo", "*.pth"), ("Todos los archivos", "*.*"))
    )
    if ruta_gfpgan:
        RUTA_GFPGAN_PTH = ruta_gfpgan
        etiqueta_gfpgan.configure(text=f"Modelo GFPGAN: {os.path.basename(ruta_gfpgan)}")

    ruta_restoreformer = filedialog.askopenfilename(
        title="Selecciona el modelo RestoreFormer++.ckpt",
        filetypes=(("Archivos de modelo", "*.ckpt"), ("Todos los archivos", "*.*"))
    )
    if ruta_restoreformer:
        RUTA_RESTOREFORMER_PTH = ruta_restoreformer
        etiqueta_restoreformer.configure(text=f"Modelo RestoreFormer: {os.path.basename(ruta_restoreformer)}")

    # F7: persistir las rutas elegidas.
    Config.save()
    log_message("Rutas de ejecutables configuradas y guardadas. Ahora puedes usar la aplicación.", "green")


def autodetectar_rutas():
    """F6: busca ejecutables y modelos en la carpeta del proyecto / models/."""
    global RUTA_EJECUTABLE, RUTA_FFMPEG, RUTA_GFPGAN_PTH, RUTA_RESTOREFORMER_PTH

    candidatos = {
        "realesrgan": (
            "realesrgan-ncnn-vulkan.exe", "realesrgan-ncnn-vulkan",
            "realesrgan-ncnn-vulkan.exe", "realesrgan-ncnn-vulkan",
        ),
        "ffmpeg": ("ffmpeg.exe", "ffmpeg"),
        "gfpgan": ("GFPGANv1.4.pth",),
        "restoreformer": ("RestoreFormer++.ckpt",),
    }

    def _buscar(nombres):
        for carpeta in (MODELS_DIR, BASE_DIR):
            for nombre in nombres:
                ruta = os.path.join(carpeta, nombre)
                if os.path.isfile(ruta):
                    return ruta
        return ""

    if not RUTA_EJECUTABLE or not os.path.isfile(RUTA_EJECUTABLE):
        RUTA_EJECUTABLE = _buscar(candidatos["realesrgan"])
    if not RUTA_FFMPEG or not os.path.isfile(RUTA_FFMPEG):
        RUTA_FFMPEG = _buscar(candidatos["ffmpeg"])
    if not RUTA_GFPGAN_PTH or not os.path.isfile(RUTA_GFPGAN_PTH):
        RUTA_GFPGAN_PTH = _buscar(candidatos["gfpgan"])
    if not RUTA_RESTOREFORMER_PTH or not os.path.isfile(RUTA_RESTOREFORMER_PTH):
        RUTA_RESTOREFORMER_PTH = _buscar(candidatos["restoreformer"])


def verificar_dependencias_inicio():
    """A10: comprueba ejecutables y módulos de Python al arrancar."""
    todo_ok = True

    if RUTA_EJECUTABLE and os.path.isfile(RUTA_EJECUTABLE):
        log_message(f"✓ Real-ESRGAN: {RUTA_EJECUTABLE}", "green")
    else:
        log_message("✗ Real-ESRGAN no está configurado (configúralo en la pestaña Configuración).", "yellow")
        todo_ok = False

    if RUTA_FFMPEG and os.path.isfile(RUTA_FFMPEG):
        log_message(f"✓ FFmpeg: {RUTA_FFMPEG}", "green")
    else:
        log_message("✗ FFmpeg no está configurado (configúralo en la pestaña Configuración).", "yellow")
        todo_ok = False

    if RUTA_GFPGAN_PTH and os.path.isfile(RUTA_GFPGAN_PTH):
        log_message(f"✓ Modelo GFPGAN: {RUTA_GFPGAN_PTH}", "green")
    else:
        log_message("• Modelo GFPGAN no configurado (opcional).", "yellow")

    if RUTA_RESTOREFORMER_PTH and os.path.isfile(RUTA_RESTOREFORMER_PTH):
        log_message(f"✓ Modelo RestoreFormer++: {RUTA_RESTOREFORMER_PTH}", "green")
    else:
        log_message("• Modelo RestoreFormer++ no configurado (opcional).", "yellow")

    modulos = {"torch": "torch", "opencv-python (cv2)": "cv2", "numpy": "numpy"}
    faltantes = []
    for etiqueta, modulo in modulos.items():
        try:
            importlib.import_module(modulo)
            log_message(f"✓ Dependencia '{etiqueta}' instalada.", "green")
        except ImportError:
            faltantes.append(etiqueta)
            log_message(f"✗ Falta la dependencia '{etiqueta}'.", "yellow")

    if faltantes:
        log_message("Instala lo que falta con:  pip install -r requirements.txt", "yellow")

    if todo_ok:
        log_message("¡Listo! La aplicación está lista para usarse.", "green")
    else:
        log_message("Faltan ejecutables por configurar (pestaña Configuración).", "yellow")
    return todo_ok

def change_appearance_mode_event(new_mode):
    ctk.set_appearance_mode(new_mode)
    Config.tema = new_mode
    Config.save()

def _actualizar_configuracion():
    """Sincroniza la configuración con los controles de la interfaz y la guarda (F7)."""
    Config.factor_escala = escala_var.get()
    Config.modelo = modelo_var.get()
    Config.formato_salida = formato_var.get()
    Config.eliminar_temporales = bool(eliminar_temp_var.get())
    try:
        Config.max_workers = max(1, min(int(trabajos_var.get()), 8))
    except (TypeError, ValueError):
        Config.max_workers = 2
    Config.save()


# --- Configuración Principal de la GUI ---
Config.load()          # F7: recupera la configuración guardada
autodetectar_rutas()   # F6: detecta ejecutables y modelos disponibles

ctk.set_appearance_mode(Config.tema)
ctk.set_default_color_theme("blue")

ventana = ctk.CTk()
ventana.title("Herramienta de Mejora Visual")
ventana.geometry("800x700")
ventana.resizable(False, False)

ventana.protocol("WM_DELETE_WINDOW", lambda: (Config.save(), ventana.destroy()))

# --- Contenedor Principal ---
main_frame = ctk.CTkFrame(ventana, corner_radius=10)
main_frame.pack(expand=True, fill="both", padx=10, pady=10)

# --- Título de la Aplicación ---
titulo_app = ctk.CTkLabel(main_frame, text="Herramienta de Mejora Visual", font=ctk.CTkFont(size=24, weight="bold"))
titulo_app.pack(pady=(10, 5))

# --- Contenedor para Pestañas ---
notebook = ctk.CTkTabview(main_frame, width=650)
notebook.pack(pady=10, expand=True, fill="both")

# --- Pestaña Imagen Única ---
pestaña_unico = notebook.add("Imagen Única")
ctk.CTkLabel(pestaña_unico, text="Mejora la calidad de una sola imagen.", font=("Arial", 12)).pack(pady=10)
boton_seleccionar_unico = ctk.CTkButton(pestaña_unico, text="1. Seleccionar Imagen", command=lambda: seleccionar_archivo("imagen_unica"))
boton_seleccionar_unico.pack(pady=5)
etiqueta_ruta_unico = ctk.CTkLabel(pestaña_unico, text="Archivo: Ninguno seleccionado", font=("Arial", 10))
etiqueta_ruta_unico.pack()
boton_mejorar_unico = ctk.CTkButton(pestaña_unico, text="2. Mejorar Imagen", command=ejecutar_realesrgan_unico, fg_color="#2ECC71", hover_color="#27AE60")
boton_mejorar_unico.pack(pady=15)

# --- Pestaña Restaurar Rostros (con selector de modelo) ---
pestaña_gfpgan = notebook.add("Restaurar Rostros")
ctk.CTkLabel(pestaña_gfpgan, text="Restaura rostros en imágenes con diferentes modelos.", font=("Arial", 12)).pack(pady=10)
ctk.CTkLabel(pestaña_gfpgan, text="Selecciona el modelo a usar:").pack(pady=(5, 0))
modelo_restauracion_var = tk.StringVar(value="GFPGAN")
modelo_restauracion_menu = ctk.CTkOptionMenu(pestaña_gfpgan, values=["GFPGAN", "RestoreFormer++"], variable=modelo_restauracion_var)
modelo_restauracion_menu.pack(pady=5)
boton_seleccionar_gfpgan = ctk.CTkButton(pestaña_gfpgan, text="1. Seleccionar y Restaurar Rostros", command=restaurar_cara, fg_color="#3498DB", hover_color="#2980B9")
boton_seleccionar_gfpgan.pack(pady=15)

# --- Pestaña Carpeta de Imágenes ---
pestaña_carpeta = notebook.add("Carpeta")
ctk.CTkLabel(pestaña_carpeta, text="Mejora todas las imágenes en una carpeta.", font=("Arial", 12)).pack(pady=10)
boton_seleccionar_entrada = ctk.CTkButton(pestaña_carpeta, text="1. Seleccionar Carpeta de Entrada", command=lambda: seleccionar_carpeta("carpeta_entrada"))
boton_seleccionar_entrada.pack(pady=5)
etiqueta_entrada_carpeta = ctk.CTkLabel(pestaña_carpeta, text="Carpeta de entrada: Ninguna", font=("Arial", 10))
etiqueta_entrada_carpeta.pack()
boton_seleccionar_salida = ctk.CTkButton(pestaña_carpeta, text="2. Seleccionar Carpeta de Salida", command=lambda: seleccionar_carpeta("carpeta_salida_img"))
boton_seleccionar_salida.pack(pady=5)
etiqueta_salida_carpeta = ctk.CTkLabel(pestaña_carpeta, text="Carpeta de salida: Ninguna", font=("Arial", 10))
etiqueta_salida_carpeta.pack()
boton_mejorar_carpeta = ctk.CTkButton(pestaña_carpeta, text="3. Mejorar Imágenes de la Carpeta", command=ejecutar_realesrgan_carpeta, fg_color="#2ECC71", hover_color="#27AE60")
boton_mejorar_carpeta.pack(pady=15)

# --- Pestaña Video ---
pestaña_video = notebook.add("Video")
ctk.CTkLabel(pestaña_video, text="Proceso completo de mejora de video.", font=("Arial", 12)).pack(pady=10)
boton_seleccionar_video = ctk.CTkButton(pestaña_video, text="1. Seleccionar Video", command=lambda: seleccionar_archivo("video"))
boton_seleccionar_video.pack(pady=5)
etiqueta_ruta_video = ctk.CTkLabel(pestaña_video, text="Video: Ninguno seleccionado", font=("Arial", 10))
etiqueta_ruta_video.pack()
boton_iniciar_video = ctk.CTkButton(pestaña_video, text="2. Iniciar Proceso de Mejora", command=ejecutar_proceso_video, fg_color="#F1C40F", hover_color="#F39C12")
boton_iniciar_video.pack(pady=15)

# --- Pestaña Convertir Video ---
pestaña_convertir_video = notebook.add("Convertir Video")
ctk.CTkLabel(pestaña_convertir_video, text="Convierte video a otros formatos.", font=("Arial", 12)).pack(pady=10)
boton_seleccionar_video_convertir = ctk.CTkButton(pestaña_convertir_video, text="1. Seleccionar Video", command=lambda: seleccionar_archivo("video_convertir"))
boton_seleccionar_video_convertir.pack(pady=5)
etiqueta_ruta_convertir = ctk.CTkLabel(pestaña_convertir_video, text="Video a convertir: Ninguno", font=("Arial", 10))
etiqueta_ruta_convertir.pack()
ctk.CTkLabel(pestaña_convertir_video, text="2. Selecciona formato de salida:").pack(pady=(10, 0))
formato_video_var = tk.StringVar(value="mp4")
formato_video_menu = ctk.CTkOptionMenu(pestaña_convertir_video, values=["mp4", "avi", "mov", "mkv", "webm", "gif"], variable=formato_video_var)
formato_video_menu.pack()
boton_convertir_video_iniciar = ctk.CTkButton(pestaña_convertir_video, text="3. Iniciar Conversión", command=convertir_video, fg_color="#3498DB", hover_color="#2980B9")
boton_convertir_video_iniciar.pack(pady=15)

# --- Pestaña Recortar Video ---
pestaña_recortar_video = notebook.add("Recortar Video")
ctk.CTkLabel(pestaña_recortar_video, text="Recorta un video con un visualizador en tiempo real.", font=("Arial", 12)).pack(pady=10)
video_frame = ctk.CTkFrame(pestaña_recortar_video, width=500, height=300)
video_frame.pack(pady=10)
etiqueta_visualizador = ctk.CTkLabel(video_frame, text="Esperando video...", font=("Arial", 14))
etiqueta_visualizador.pack(expand=True, fill="both")
boton_seleccionar_video_recortar = ctk.CTkButton(pestaña_recortar_video, text="1. Seleccionar Video", command=lambda: (seleccionar_archivo("video_recortar"), abrir_y_reproducir()))
boton_seleccionar_video_recortar.pack(pady=5)
etiqueta_ruta_recortar = ctk.CTkLabel(pestaña_recortar_video, text="Video a recortar: Ninguno", font=("Arial", 10))
etiqueta_ruta_recortar.pack()
boton_detener_reproduccion = ctk.CTkButton(pestaña_recortar_video, text="Detener reproducción", command=detener_reproduccion, fg_color="#95A5A6", hover_color="#7F8C8D")
boton_detener_reproduccion.pack(pady=5)
boton_recortar_video_iniciar = ctk.CTkButton(pestaña_recortar_video, text="2. Recortar Video", command=recortar_video, fg_color="#F1C40F", hover_color="#F39C12")
boton_recortar_video_iniciar.pack(pady=15)

# --- Pestaña Configuración ---
pestaña_config = notebook.add("Configuración")
ctk.CTkLabel(pestaña_config, text="Configura las rutas y opciones de mejora.", font=("Arial", 12)).pack(pady=10)

ctk.CTkButton(pestaña_config, text="1. Configurar Rutas de Ejecutables", command=configurar_rutas).pack(pady=5)
etiqueta_realesrgan = ctk.CTkLabel(pestaña_config, text="Real-ESRGAN: No configurado")
etiqueta_realesrgan.pack()
etiqueta_ffmpeg = ctk.CTkLabel(pestaña_config, text="FFmpeg: No configurado")
etiqueta_ffmpeg.pack()
etiqueta_gfpgan = ctk.CTkLabel(pestaña_config, text="Modelo GFPGAN: No configurado")
etiqueta_gfpgan.pack()
etiqueta_restoreformer = ctk.CTkLabel(pestaña_config, text="Modelo RestoreFormer: No configurado")
etiqueta_restoreformer.pack()

ctk.CTkLabel(pestaña_config, text="Opciones de Mejora:").pack(pady=(15, 5))
# Factor de escala
ctk.CTkLabel(pestaña_config, text="Factor de escala:").pack()
escala_var = tk.StringVar(value=Config.factor_escala)
escala_menu = ctk.CTkOptionMenu(pestaña_config, values=["2x", "3x", "4x"], variable=escala_var, command=lambda x: _actualizar_configuracion())
escala_menu.pack()
# Modelo de mejora
ctk.CTkLabel(pestaña_config, text="Modelo:").pack(pady=(10, 0))
modelo_var = tk.StringVar(value=Config.modelo)
modelo_menu = ctk.CTkOptionMenu(pestaña_config, values=["realesr-animevideov3", "realesrgan-x4plus", "realesrgan-x4plus-anime"], variable=modelo_var, command=lambda x: _actualizar_configuracion())
modelo_menu.pack()
# Formato de salida
ctk.CTkLabel(pestaña_config, text="Formato de salida:").pack(pady=(10, 0))
formato_var = tk.StringVar(value=Config.formato_salida)
formato_menu = ctk.CTkOptionMenu(pestaña_config, values=["png", "jpg"], variable=formato_var, command=lambda x: _actualizar_configuracion())
formato_menu.pack()
# Trabajos simultáneos (F9)
ctk.CTkLabel(pestaña_config, text="Trabajos simultáneos (carpeta):").pack(pady=(10, 0))
trabajos_var = tk.StringVar(value=str(Config.max_workers))
trabajos_menu = ctk.CTkOptionMenu(pestaña_config, values=["1", "2", "3", "4"], variable=trabajos_var, command=lambda x: _actualizar_configuracion())
trabajos_menu.pack()
# Eliminar archivos temporales
eliminar_temp_var = tk.BooleanVar(value=Config.eliminar_temporales)
eliminar_temp_check = ctk.CTkCheckBox(pestaña_config, text="Eliminar archivos de video temporales", variable=eliminar_temp_var, command=lambda: _actualizar_configuracion())
eliminar_temp_check.pack(pady=15)
# Opción de Tema
ctk.CTkLabel(pestaña_config, text="Tema de la aplicación:").pack(pady=(15, 5))
tema_var = tk.StringVar(value=Config.tema)
tema_menu = ctk.CTkOptionMenu(pestaña_config, values=["Dark", "Light"], variable=tema_var, command=change_appearance_mode_event)
tema_menu.pack()

# --- Barra de progreso y área de registro ---
barra_progreso = ctk.CTkProgressBar(main_frame, orientation="horizontal", width=650)
barra_progreso.set(0)
barra_progreso.pack(fill="x", padx=10, pady=(10, 0))

log_text = ctk.CTkTextbox(main_frame, height=100, corner_radius=10, state="disabled")
log_text.pack(fill="both", expand=True, padx=10, pady=10)

def initialize_app():
    verificar_dependencias_inicio()

log_text.tag_config("green", foreground="#2ECC71")
log_text.tag_config("red", foreground="#E74C3C")
log_text.tag_config("blue", foreground="#3498DB")
log_text.tag_config("yellow", foreground="#F1C40F")

# Refleja en la interfaz las rutas autodetectadas o recuperadas de la config.
if RUTA_EJECUTABLE:
    etiqueta_realesrgan.configure(text=f"Real-ESRGAN: {os.path.basename(RUTA_EJECUTABLE)}")
if RUTA_FFMPEG:
    etiqueta_ffmpeg.configure(text=f"FFmpeg: {os.path.basename(RUTA_FFMPEG)}")
if RUTA_GFPGAN_PTH:
    etiqueta_gfpgan.configure(text=f"Modelo GFPGAN: {os.path.basename(RUTA_GFPGAN_PTH)}")
if RUTA_RESTOREFORMER_PTH:
    etiqueta_restoreformer.configure(text=f"Modelo RestoreFormer: {os.path.basename(RUTA_RESTOREFORMER_PTH)}")

ventana.after(100, initialize_app)

# REESCALLER_HEADLESS=1 permite importar la app en pruebas sin abrir el bucle principal.
if not os.environ.get("REESCALLER_HEADLESS"):
    ventana.mainloop()
