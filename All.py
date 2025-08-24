import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
import subprocess
import os
import threading
import sys
import importlib
from concurrent.futures import ThreadPoolExecutor

# --- Importaciones de Librerías y verificación ---
try:
    import customtkinter as ctk
    from PIL import Image, ImageTk
    import cv2
    import numpy as np
    # Importaciones condicionales que se verificarán más tarde
    from RestoreFormer.RestoreFormer import RestoreFormer
    from basicsr.utils import imwrite
    import torch
    import gfpgan
except ImportError:
    pass # No hay problema, se instalarán/verificarán en el uso

# --- Rutas de Ejecutables (se inicializan al inicio) ---
RUTA_EJECUTABLE = ""
RUTA_FFMPEG = ""
RUTA_GFPGAN_PTH = ""
RUTA_RESTOREFORMER_PTH = ""

# --- Variables Globales de la Aplicación ---
rutas_seleccionadas = {
    "imagen_unica": "",
    "carpeta_entrada": "",
    "carpeta_salida_img": "",
    "video": "",
    "video_convertir": "",
    "video_recortar": ""
}

PYTORCH_DISPONIBLE = False

# --- Configuración Personalizable ---
class Config:
    factor_escala = "4x"
    modelo = "realesrgan-x4plus"
    formato_salida = "png"
    eliminar_temporales = True
    tema = "Dark"

# --- Funciones de la Interfaz de Usuario ---
def log_message(message, color="white"):
    """Inserta un mensaje en el área de registro."""
    log_text.configure(state="normal")
    log_text.insert(tk.END, message + "\n", color)
    log_text.configure(state="disabled")
    log_text.see(tk.END)
    ventana.update_idletasks()

def set_progress(value):
    """Actualiza la barra de progreso."""
    barra_progreso.set(value / 100.0)
    ventana.update_idletasks()

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
    for button in buttons:
        button.configure(state=state)

def proceso_ejecutar(comando, mensaje_progreso, paso_actual=0, total_pasos=1):
    """Función genérica para ejecutar comandos con manejo de errores y progreso."""
    try:
        log_message(f"Iniciando: {mensaje_progreso}", "blue")
        set_progress((paso_actual / total_pasos) * 100)
        
        proceso = subprocess.run(
            comando,
            check=True,
            capture_output=True,
            text=True,
            shell=True
        )
        
        log_message("Comando ejecutado con éxito.", "green")
        log_message(f"Salida:\n{proceso.stdout.strip()}", "green")
        set_progress(((paso_actual + 1) / total_pasos) * 100)
        
    except subprocess.CalledProcessError as e:
        log_message(f"Error en la ejecución:\n{e.stderr.strip()}", "red")
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
        messagebox.showerror("Error", "Por favor, configura las rutas de los ejecutables en la pestaña de Configuración.")
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
                "-s", Config.factor_escala[0],
                "-n", Config.modelo
            ]
            proceso_ejecutar(comando, "Mejorando imagen...")
            log_message(f"\n¡Éxito! Imagen mejorada y guardada en:\n{ruta_salida}", "green")
            messagebox.showinfo("Proceso Completo", "¡La mejora de la imagen se completó con éxito!")
        except Exception:
            pass
        finally:
            toggle_buttons_state("normal")
    
    threading.Thread(target=worker).start()

# --- Lógica de la restauración facial con selector de modelo (GFPGAN y RestoreFormer++) ---
def restaurar_cara():
    global PYTORCH_DISPONIBLE
    
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
        except Exception as e:
            log_message(f"Ocurrió un error inesperado al usar el modelo:\n{e}", "red")
            set_progress(0)
        finally:
            toggle_buttons_state("normal")

    threading.Thread(target=worker).start()

def ejecutar_gfpgan(ruta_imagen):
    # Lógica de GFPGAN
    try:
        global PYTORCH_DISPONIBLE
        try:
            import torch
            import gfpgan
            PYTORCH_DISPONIBLE = True
        except ImportError:
            PYTORCH_DISPONIBLE = False
            
        if not PYTORCH_DISPONIBLE:
            log_message("PyTorch y GFPGAN no están instalados. Instalando dependencias...", "yellow")
            if not instalar_dependencias_pytorch():
                return
            try:
                import torch
                import gfpgan
                PYTORCH_DISPONIBLE = True
            except ImportError:
                log_message("La instalación de PyTorch falló.", "red")
                return

        if not RUTA_GFPGAN_PTH or not os.path.exists(RUTA_GFPGAN_PTH):
            log_message("Advertencia: No se encontró el modelo GFPGAN. Por favor, configúralo en la pestaña de Configuración.", "red")
            return
    
        from gfpgan import GFPGANer
        
        log_message("Iniciando restauración de rostros con GFPGAN...", "blue")
        set_progress(25)
        
        restaurador = GFPGANer(model_path=RUTA_GFPGAN_PTH, upscale=2, bg_upsampler=None, device='cuda' if torch.cuda.is_available() else 'cpu')
        log_message("Modelo GFPGAN cargado.", "green")
        set_progress(50)
        
        imagen_input = cv2.imread(ruta_imagen, cv2.IMREAD_COLOR)
        _, _, imagen_salida = restaurador.enhance(imagen_input, has_aligned=False, only_center_face=False)
        
        nombre_base, _ = os.path.splitext(os.path.basename(ruta_imagen))
        ruta_salida_gfpgan = os.path.join(os.path.dirname(ruta_imagen), f"{nombre_base}_gfpgan.png")
        cv2.imwrite(ruta_salida_gfpgan, imagen_salida)
        
        log_message(f"\n¡Éxito! Imagen con rostros restaurados (GFPGAN) guardada en:\n{ruta_salida_gfpgan}", "green")
        set_progress(100)
        messagebox.showinfo("Proceso Completo", "¡La restauración de rostros se completó con éxito!")
        
    except Exception as e:
        log_message(f"Ocurrió un error inesperado al usar GFPGAN:\n{e}", "red")
        set_progress(0)

def ejecutar_restoreformer(ruta_imagen):
    # Lógica para RestoreFormer++ integrada
    try:
        try:
            from RestoreFormer.RestoreFormer import RestoreFormer
            from basicsr.utils import imwrite
            import torch
            import cv2
            import numpy as np
        except ImportError as e:
            log_message(f"Error: La librería de RestoreFormer o sus dependencias no están instaladas: {e.name}", "red")
            log_message("Por favor, instala las dependencias necesarias con 'pip install basicsr'", "yellow")
            set_progress(0)
            return

        if not RUTA_RESTOREFORMER_PTH or not os.path.exists(RUTA_RESTOREFORMER_PTH):
            log_message("Advertencia: No se encontró el modelo RestoreFormer++. Por favor, configúralo en la pestaña de Configuración.", "red")
            set_progress(0)
            return
            
        log_message("Iniciando restauración de rostros con RestoreFormer++...", "blue")
        set_progress(25)

        restaurador = RestoreFormer(
            model_path=RUTA_RESTOREFORMER_PTH,
            upscale=2,
            arch='RestoreFormer++',
            bg_upsampler=None
        )
        log_message("Modelo RestoreFormer++ cargado.", "green")
        set_progress(50)

        input_img = cv2.imread(ruta_imagen, cv2.IMREAD_COLOR)
        cropped_faces, restored_faces, restored_img = restaurador.enhance(
            input_img,
            has_aligned=False,
            only_center_face=False,
            paste_back=True
        )

        if restored_img is not None:
            nombre_base, _ = os.path.splitext(os.path.basename(ruta_imagen))
            ruta_salida_rf = os.path.join(os.path.dirname(ruta_imagen), f"{nombre_base}_rf++.png")
            imwrite(restored_img, ruta_salida_rf)
            log_message(f"\n¡Éxito! Imagen con rostros restaurados (RestoreFormer++) guardada en:\n{ruta_salida_rf}", "green")
            set_progress(100)
            messagebox.showinfo("Proceso Completo", "¡La restauración de rostros se completó con éxito!")
        else:
            log_message("No se encontraron rostros en la imagen o la restauración falló.", "yellow")
            set_progress(0)

    except Exception as e:
        log_message(f"Ocurrió un error inesperado al usar RestoreFormer++:\n{e}", "red")
        set_progress(0)

def ejecutar_realesrgan_carpeta():
    if not RUTA_EJECUTABLE:
        messagebox.showerror("Error", "Por favor, configura las rutas de los ejecutables en la pestaña de Configuración.")
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
                return

            total_archivos = len(archivos)
            procesados = 0
            
            with ThreadPoolExecutor(max_workers=os.cpu_count() or 4) as executor:
                futures = {
                    executor.submit(
                        proceso_ejecutar, 
                        [RUTA_EJECUTABLE, "-i", os.path.join(input_folder, f), "-o", os.path.join(output_folder, f"{os.path.splitext(f)[0]}_mejorado.{Config.formato_salida}"), "-s", Config.factor_escala[0], "-n", Config.modelo], 
                        f"Procesando {f}...", 
                        paso_actual=procesados, 
                        total_pasos=total_archivos
                    ): f for f in archivos
                }

                for future in futures:
                    future.result()
                    procesados += 1
                    set_progress((procesados / total_archivos) * 100)

            log_message(f"\n¡Éxito! Se mejoraron {total_archivos} imágenes y se guardaron en:\n{output_folder}", "green")
            messagebox.showinfo("Proceso Completo", f"¡La mejora de {total_archivos} imágenes se completó con éxito!")
        except Exception as e:
            log_message(f"Ocurrió un error inesperado:\n{e}", "red")
        finally:
            toggle_buttons_state("normal")

    threading.Thread(target=worker).start()

def ejecutar_proceso_video():
    if not RUTA_EJECUTABLE or not RUTA_FFMPEG:
        messagebox.showerror("Error", "Por favor, configura las rutas de los ejecutables en la pestaña de Configuración.")
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
                RUTA_FFMPEG, "-i", rutas_seleccionadas["video"], "-qscale:v", "1", "-qmin", "1", "-qmax", "1", "-vsync", "0",
                os.path.join(tmp_frames_dir, "frame%08d.jpg")
            ]
            proceso_ejecutar(comando_extract, "Paso 1/3: Extrayendo frames del video...", 0, 3)
            
            # 2. Mejorar frames
            os.makedirs(out_frames_dir, exist_ok=True)
            comando_enhance = [
                RUTA_EJECUTABLE, "-i", tmp_frames_dir, "-o", out_frames_dir, 
                "-n", Config.modelo, 
                "-s", Config.factor_escala[0],
                "-f", "jpg"
            ]
            proceso_ejecutar(comando_enhance, "Paso 2/3: Mejorando los frames...", 1, 3)

            # 3. Fusionar frames y audio
            comando_merge = [
                RUTA_FFMPEG, "-i", os.path.join(out_frames_dir, "frame%08d.jpg"), "-i", rutas_seleccionadas["video"],
                "-map", "0:v:0", "-map", "1:a:0", "-c:a", "copy", "-c:v", "libx264", "-r", "23.98", "-pix_fmt", "yuv420p",
                ruta_video_final
            ]
            proceso_ejecutar(comando_merge, "Paso 3/3: Fusionando frames y audio...", 2, 3)
            
            log_message(f"\n¡Proceso Completado! El video mejorado se guardó en:\n{ruta_video_final}", "green")
            messagebox.showinfo("Proceso Completo", "¡La mejora del video se completó con éxito!")
            
            # 4. Eliminar archivos temporales
            if Config.eliminar_temporales:
                log_message("Eliminando archivos temporales...", "blue")
                try:
                    import shutil
                    shutil.rmtree(tmp_frames_dir)
                    shutil.rmtree(out_frames_dir)
                    log_message("Archivos temporales eliminados con éxito.", "green")
                except Exception as e:
                    log_message(f"Error al eliminar archivos temporales: {e}", "red")

        except Exception:
            pass
        finally:
            toggle_buttons_state("normal")

    threading.Thread(target=worker).start()

def convertir_video():
    if not RUTA_FFMPEG:
        messagebox.showerror("Error", "Por favor, configura la ruta de FFmpeg en la pestaña de Configuración.")
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
                RUTA_FFMPEG, 
                "-i", ruta_video,
                "-vcodec", "copy",
                "-acodec", "copy",
                ruta_salida
            ]

            proceso_ejecutar(comando, f"Convirtiendo a {formato_salida}...")
            
            log_message(f"\n¡Éxito! Video convertido y guardado en:\n{ruta_salida}", "green")
            messagebox.showinfo("Proceso Completo", f"¡La conversión a {formato_salida} se completó con éxito!")

        except Exception:
            pass
        finally:
            toggle_buttons_state("normal")

def recortar_video():
    if not RUTA_FFMPEG:
        messagebox.showerror("Error", "Por favor, configura la ruta de FFmpeg en la pestaña de Configuración.")
        return
    if not rutas_seleccionadas["video_recortar"]:
        log_message("Error: Por favor, selecciona un video para recortar.", "red")
        return

    def worker():
        toggle_buttons_state("disabled")
        try:
            ruta_video = rutas_seleccionadas["video_recortar"]
            nombre_base, _ = os.path.splitext(os.path.basename(ruta_video))
            ruta_salida = os.path.join(os.path.dirname(ruta_video), f"{nombre_base}_recortado.mp4")

            tiempo_inicio = simpledialog.askstring("Recortar Video", "Ingresa el tiempo de inicio (hh:mm:ss):")
            duracion = simpledialog.askstring("Recortar Video", "Ingresa la duración (en segundos):")

            if not tiempo_inicio or not duracion:
                log_message("Proceso cancelado por el usuario.", "yellow")
                return

            comando = [
                RUTA_FFMPEG, 
                "-ss", tiempo_inicio, 
                "-i", ruta_video, 
                "-t", duracion, 
                "-c", "copy", 
                ruta_salida
            ]
            proceso_ejecutar(comando, f"Recortando video...")
            
            log_message(f"\n¡Éxito! Video recortado y guardado en:\n{ruta_salida}", "green")
            messagebox.showinfo("Proceso Completo", "¡El video se recortó con éxito!")

        except Exception as e:
            log_message(f"Ocurrió un error inesperado al recortar el video:\n{e}", "red")
        finally:
            toggle_buttons_state("normal")

    threading.Thread(target=worker).start()

# --- Funciones para el Visualizador de Video ---
video_reproductor = None
def abrir_y_reproducir():
    global video_reproductor
    if not rutas_seleccionadas["video_recortar"]:
        log_message("Por favor, selecciona un video primero.", "red")
        return

    if video_reproductor:
        video_reproductor.release()
        etiqueta_visualizador.configure(image=None)

    video_reproductor = cv2.VideoCapture(rutas_seleccionadas["video_recortar"])
    if not video_reproductor.isOpened():
        log_message("Error: No se pudo abrir el archivo de video.", "red")
        return

    threading.Thread(target=reproducir_video, daemon=True).start()

def reproducir_video():
    global video_reproductor
    if video_reproductor is None:
        return

    ancho_max = 500
    alto_max = 300
    
    while video_reproductor.isOpened():
        ret, frame = video_reproductor.read()
        if not ret:
            break
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        
        ancho_original, alto_original = img.size
        ratio = min(ancho_max / ancho_original, alto_max / alto_original)
        nuevo_ancho = int(ancho_original * ratio)
        nuevo_alto = int(alto_original * ratio)
        
        img_redimensionada = img.resize((nuevo_ancho, nuevo_alto), Image.Resampling.LANCZOS)
        
        img_tk = ctk.CTkImage(light_image=img_redimensionada, dark_image=img_redimensionada, size=(nuevo_ancho, nuevo_alto))
        
        etiqueta_visualizador.configure(image=img_tk)
        etiqueta_visualizador.image = img_tk
        
        ventana.update()
        
    video_reproductor.release()
    etiqueta_visualizador.configure(image=None)
    log_message("Reproducción finalizada.", "yellow")

def instalar_dependencias_pytorch():
    try:
        log_message("Iniciando la instalación de dependencias...", "blue")
        dependencias = [
            "torch", "torchvision", "basicsr", "gfpgan", "opencv-python", "numpy"
        ]
        log_message("Instalando dependencias de PyTorch y GFPGAN...")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install'] + dependencias)
        log_message("\n¡Instalación completada con éxito!", "green")
        importlib.reload(sys.modules['torch'])
        importlib.reload(sys.modules['gfpgan'])
    except subprocess.CalledProcessError as e:
        log_message(f"La instalación falló. Código de error: {e.returncode}", "red")
        log_message(e.stderr.strip(), "red")
        return False
    except Exception as e:
        log_message(f"Ocurrió un error inesperado durante la instalación:\n{e}", "red")
        return False
    return True

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
    
    log_message("Rutas de ejecutables configuradas. Ahora puedes usar la aplicación.", "green")

def verificar_dependencias_inicio():
    if RUTA_EJECUTABLE and RUTA_FFMPEG:
        log_message("¡Listo! La aplicación está lista para usarse.", "green")
        return True
    else:
        log_message("Por favor, configura las rutas de los ejecutables en la pestaña de Configuración.", "yellow")
        return False

def change_appearance_mode_event(new_mode):
    ctk.set_appearance_mode(new_mode)
    Config.tema = new_mode

# --- Configuración Principal de la GUI ---
ctk.set_appearance_mode(Config.tema)
ctk.set_default_color_theme("blue")

ventana = ctk.CTk()
ventana.title("Herramienta de Mejora Visual")
ventana.geometry("800x700")
ventana.resizable(False, False)

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
escala_menu = ctk.CTkOptionMenu(pestaña_config, values=["2x", "3x", "4x"], variable=escala_var, command=lambda x: setattr(Config, 'factor_escala', x))
escala_menu.pack()
# Modelo de mejora
ctk.CTkLabel(pestaña_config, text="Modelo:").pack(pady=(10, 0))
modelo_var = tk.StringVar(value=Config.modelo)
modelo_menu = ctk.CTkOptionMenu(pestaña_config, values=["realesr-animevideov3", "realesrgan-x4plus", "realesrgan-x4plus-anime"], variable=modelo_var, command=lambda x: setattr(Config, 'modelo', x))
modelo_menu.pack()
# Formato de salida
ctk.CTkLabel(pestaña_config, text="Formato de salida:").pack(pady=(10, 0))
formato_var = tk.StringVar(value=Config.formato_salida)
formato_menu = ctk.CTkOptionMenu(pestaña_config, values=["png", "jpg"], variable=formato_var, command=lambda x: setattr(Config, 'formato_salida', x))
formato_menu.pack()
# Eliminar archivos temporales
eliminar_temp_var = tk.BooleanVar(value=Config.eliminar_temporales)
eliminar_temp_check = ctk.CTkCheckBox(pestaña_config, text="Eliminar archivos de video temporales", variable=eliminar_temp_var, command=lambda: setattr(Config, 'eliminar_temporales', eliminar_temp_var.get()))
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

ventana.after(100, initialize_app)
ventana.mainloop()