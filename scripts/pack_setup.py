"""Crea el instalador propio de Reescaller: lanzador (reescaller-setup-launcher.exe) + ZIP con todo al final.

Uso:  python -I scripts\\pack_setup.py <lanzador.exe> <interfaz.exe> <loader.dll> <carpeta_app> <webview2_setup.exe> <salida.exe>

El ZIP lleva: `ui/reescaller-setup.exe` y `ui/WebView2Loader.dll` (interfaz que el lanzador extrae a una carpeta temporal; la
interfaz también se copia como `Uninstall.exe` al instalar), los archivos de la app (rutas relativas) y
`redist/MicrosoftEdgeWebview2Setup.exe`.
"""
import os
import shutil
import sys
import tempfile
import zipfile


def main():
    if len(sys.argv) != 7:
        sys.exit(__doc__)
    stub, ui, loader, app, wv2, out = sys.argv[1:]
    for p in (stub, ui, loader, app, wv2):
        if not os.path.exists(p):
            sys.exit(f"No existe: {p}")

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".zip")
    tmp.close()
    n = 0
    with zipfile.ZipFile(tmp.name, "w", zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True) as z:
        z.write(ui, "ui/reescaller-setup.exe")
        z.write(loader, "ui/WebView2Loader.dll")
        z.write(wv2, "redist/MicrosoftEdgeWebview2Setup.exe")
        for root, _dirs, files in os.walk(app):
            for f in sorted(files):
                full = os.path.join(root, f)
                rel = os.path.relpath(full, app).replace(os.sep, "/")
                # los .onnx/.bin ya vienen comprimidos o casi: deflate no gana nada y tarda
                comp = zipfile.ZIP_STORED if rel.lower().endswith((".onnx", ".bin", ".png", ".jpg")) else zipfile.ZIP_DEFLATED
                z.write(full, rel, compress_type=comp)
                n += 1

    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "wb") as o:
        with open(stub, "rb") as s:
            shutil.copyfileobj(s, o, 1 << 20)
        with open(tmp.name, "rb") as z:
            shutil.copyfileobj(z, o, 1 << 20)
    os.unlink(tmp.name)
    print(f"{n} archivos -> {out} ({os.path.getsize(out) / 1048576:.1f} MB)")


if __name__ == "__main__":
    main()
