//! Incrusta el icono y la información de versión en el ejecutable de Windows.

fn main() {
    #[cfg(windows)]
    {
        println!("cargo:rerun-if-changed=assets/reescaller.ico");
        let mut res = winres::WindowsResource::new();
        res.set_icon("assets/reescaller.ico");
        res.set("ProductName", "Reescaller");
        res.set(
            "FileDescription",
            "Reescaller - Remasterización de imágenes con IA",
        );
        res.set("CompanyName", "Reescaller");
        res.set("LegalCopyright", "MIT License");
        res.set("OriginalFilename", "Reescaller.exe");
        // Si no hay windres/rc disponible la compilación continúa sin icono.
        match res.compile() {
            Err(e) => println!("cargo:warning=No se pudo incrustar el icono: {e}"),
            Ok(()) => {
                // Con la cadena GNU, el recurso va en una biblioteca estática y el enlazador descarta un objeto que nadie
                // referencia (el .exe se quedaba sin icono ni versión). Se pasa el objeto directamente a los ejecutables.
                if let Ok(out) = std::env::var("OUT_DIR") {
                    let obj = std::path::Path::new(&out).join("resource.o");
                    if obj.is_file() {
                        println!("cargo:rustc-link-arg-bins={}", obj.display());
                    }
                }
            }
        }
    }
}
