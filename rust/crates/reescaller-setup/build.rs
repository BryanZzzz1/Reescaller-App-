//! Incrusta el icono y la información de versión en el instalador.

fn main() {
    #[cfg(windows)]
    {
        println!("cargo:rerun-if-changed=../reescaller-app/assets/reescaller.ico");
        let mut res = winres::WindowsResource::new();
        res.set_icon("../reescaller-app/assets/reescaller.ico");
        res.set("ProductName", "Reescaller");
        res.set("FileDescription", "Instalador de Reescaller");
        res.set("CompanyName", "Reescaller");
        res.set("OriginalFilename", "Reescaller-Setup.exe");
        match res.compile() {
            Err(e) => println!("cargo:warning=No se pudo incrustar el icono: {e}"),
            Ok(()) => {
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
