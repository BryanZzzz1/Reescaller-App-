"""Genera las imágenes del instalador (estilo de la app: vidrio oscuro con manchas de color y el logo de Reescaller).

Uso:  python installer\\make_assets.py      (requiere Pillow)
Crea installer\\assets\\wizard-*.bmp (panel lateral) y small-*.bmp (icono de la cabecera) en varios tamaños (DPI).
"""
from pathlib import Path
from PIL import Image, ImageDraw, ImageFilter, ImageFont

OUT = Path(__file__).parent / "assets"
OUT.mkdir(exist_ok=True)

BASE = (10, 13, 20)
VIOLET = (139, 92, 246)
CYAN = (34, 211, 238)
PINK = (244, 114, 182)


def blobs(w, h):
    """Fondo oscuro con manchas difuminadas (el «cristal» de la app)."""
    img = Image.new("RGB", (w, h), BASE)
    layer = Image.new("RGB", (w, h), BASE)
    d = ImageDraw.Draw(layer)
    d.ellipse((-w * .5, -h * .08, w * .95, h * .36), fill=VIOLET)
    d.ellipse((w * .15, h * .30, w * 1.5, h * .72), fill=CYAN)
    d.ellipse((-w * .6, h * .62, w * .8, h * 1.1), fill=PINK)
    layer = layer.filter(ImageFilter.GaussianBlur(w * .22))
    return Image.blend(img, layer, .55)


def gradient_square(size, radius_frac=9 / 32):
    s = size
    g = Image.new("RGB", (s, s))
    px = g.load()
    for y in range(s):
        for x in range(s):
            t = (x + y) / (2 * s)
            px[x, y] = tuple(int(VIOLET[i] * (1 - t) + CYAN[i] * t) for i in range(3))
    mask = Image.new("L", (s, s), 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, s - 1, s - 1), radius=int(s * radius_frac), fill=255)
    return g, mask


def logo(size):
    """El logo de la app (cuadrado degradado + píxel sólido y dos esquinas) con suavizado."""
    ss = 4
    s = size * ss
    g, mask = gradient_square(s)
    d = ImageDraw.Draw(g)
    u = s / 32.0
    d.rounded_rectangle((8 * u, 18 * u, 14 * u, 24 * u), radius=1.2 * u, fill=(255, 255, 255))
    def poly(points, alpha):
        layer = Image.new("RGBA", (s, s), (0, 0, 0, 0))
        ld = ImageDraw.Draw(layer)
        w = int(2 * u)
        pts = [(x * u, y * u) for x, y in points]
        ld.line(pts, fill=(255, 255, 255, int(255 * alpha)), width=w, joint="curve")
        for x, y in (pts[0], pts[-1]):
            ld.ellipse((x - w / 2, y - w / 2, x + w / 2, y + w / 2), fill=(255, 255, 255, int(255 * alpha)))
        return layer
    base = g.convert("RGBA")
    base = Image.alpha_composite(base, poly([(15, 12), (21, 12), (21, 18)], 1.0))
    base = Image.alpha_composite(base, poly([(15, 6), (26, 6), (26, 17)], .6))
    base.putalpha(mask)
    return base.resize((size, size), Image.LANCZOS)


def font(name, size):
    for f in (name, "segoeui.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(f"C:/Windows/Fonts/{f}", size)
        except OSError:
            continue
    return ImageFont.load_default()


def wizard_image(w, h):
    img = blobs(w, h).convert("RGBA")
    # panel de cristal
    glass = Image.new("RGBA", (w, h), (255, 255, 255, 0))
    gd = ImageDraw.Draw(glass)
    pad = int(w * .09)
    gd.rounded_rectangle((pad, pad, w - pad, h - pad), radius=int(w * .09), fill=(255, 255, 255, 20), outline=(255, 255, 255, 46), width=max(1, w // 160))
    img = Image.alpha_composite(img, glass)
    lg = logo(int(w * .42))
    img.alpha_composite(lg, ((w - lg.width) // 2, int(h * .17)))
    d = ImageDraw.Draw(img)
    f1 = font("segoeuib.ttf", int(w * .125))
    f2 = font("segoeui.ttf", int(w * .066))
    def center(text, y, f, fill):
        bb = d.textbbox((0, 0), text, font=f)
        d.text(((w - (bb[2] - bb[0])) / 2, y), text, font=f, fill=fill)
    center("Reescaller", int(h * .17) + lg.height + int(h * .05), f1, (255, 255, 255, 255))
    center("Mejora tus imágenes", int(h * .60), f2, (214, 220, 240, 255))
    center("y videos con IA", int(h * .60) + int(w * .09), f2, (214, 220, 240, 255))
    # barra de color inferior
    bar_w, bar_h = int(w * .5), max(3, int(h * .008))
    bar = Image.new("RGB", (bar_w, bar_h))
    bp = bar.load()
    for x in range(bar_w):
        t = x / max(1, bar_w - 1)
        c = tuple(int(VIOLET[i] * (1 - t) + CYAN[i] * t) for i in range(3))
        for y in range(bar_h):
            bp[x, y] = c
    img.paste(bar, ((w - bar_w) // 2, int(h * .86)))
    return img.convert("RGB")


def small_image(w, h):
    img = Image.new("RGB", (w, h), (32, 32, 36))
    lg = logo(int(min(w, h) * .82))
    img.paste(lg, ((w - lg.width) // 2, (h - lg.height) // 2), lg)
    return img


if __name__ == "__main__":
    for (w, h) in ((164, 314), (192, 366), (246, 471), (328, 628)):
        wizard_image(w * 2, h * 2).resize((w, h), Image.LANCZOS).save(OUT / f"wizard-{w}.bmp")
    for (w, h) in ((55, 58), (64, 68), (80, 85), (110, 116)):
        small_image(w * 4, h * 4).resize((w, h), Image.LANCZOS).save(OUT / f"small-{w}.bmp")
    print("imágenes creadas en", OUT)
