"""Genera imágenes de prueba para los casos límite del motor nativo de rostros."""
import os
import numpy as np
import cv2
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
S = os.path.join(HERE, "samples")
OUT = os.path.join(HERE, "edge")
os.makedirs(OUT, exist_ok=True)

src = cv2.imread(os.path.join(S, "Blake_Lively.jpg"))          # 700x464, 2 caras
h, w = src.shape[:2]

# 1) escala de grises (JPEG de 1 canal)
cv2.imwrite(os.path.join(OUT, "gris.jpg"), cv2.cvtColor(src, cv2.COLOR_BGR2GRAY))

# 2) RGBA con alfa (marco transparente)
rgba = cv2.cvtColor(src, cv2.COLOR_BGR2BGRA)
rgba[:, :, 3] = 255
rgba[:20, :, 3] = 0
rgba[:, :20, 3] = 0
cv2.imwrite(os.path.join(OUT, "alfa.png"), rgba)

# 3) PNG de 16 bits
cv2.imwrite(os.path.join(OUT, "16bits.png"), (src.astype(np.uint16) * 257))

# 4) cara cortada por el borde: desplaza la imagen para que una cara quede a medias fuera
canvas = np.full((h, w, 3), 128, np.uint8)
canvas[:, : w - 170] = src[:, 170:]   # el contenido se desplaza a la izquierda: queda una cara cortada
cv2.imwrite(os.path.join(OUT, "borde.jpg"), canvas)

# 5) cara pequeña: la imagen entera reducida (caras de pocos píxeles) y una media
cv2.imwrite(os.path.join(OUT, "pequena_160.jpg"), cv2.resize(src, (160, int(160 * h / w)), interpolation=cv2.INTER_AREA))
cv2.imwrite(os.path.join(OUT, "pequena_300.jpg"), cv2.resize(src, (300, int(300 * h / w)), interpolation=cv2.INTER_AREA))

# 6) fotos grandes
cv2.imwrite(os.path.join(OUT, "grande_4000.jpg"), cv2.resize(src, (4000, int(4000 * h / w)), interpolation=cv2.INTER_LANCZOS4), [cv2.IMWRITE_JPEG_QUALITY, 92])
cv2.imwrite(os.path.join(OUT, "grande_6000.jpg"), cv2.resize(src, (6000, int(6000 * h / w)), interpolation=cv2.INTER_LANCZOS4), [cv2.IMWRITE_JPEG_QUALITY, 92])

# 7) orientación EXIF: píxeles girados 90° y etiqueta «gira 90° a la derecha» (6)
pil = Image.fromarray(cv2.cvtColor(src, cv2.COLOR_BGR2RGB)).rotate(90, expand=True)
exif = Image.Exif()
exif[0x0112] = 6
pil.save(os.path.join(OUT, "exif_giro.jpg"), exif=exif, quality=92)

for f in sorted(os.listdir(OUT)):
    im = Image.open(os.path.join(OUT, f))
    print(f"{f}: {im.size} modo={im.mode}")
