//! One PNG of what this renderer last put on screen.
//!
//! `bevy_render`'s screenshot does not exist on this path, so until now a
//! question about what the native build looks like could only be answered by
//! someone holding the device. `CAPTURE_SCREENSHOT_TO=<path>` writes a single
//! frame and then costs nothing.
//!
//! The layer has to be created with `framebufferOnly = false` for the readback
//! to be allowed at all; the host does that when the variable is set.

use metal::{MTLOrigin, MTLRegion, MTLSize, TextureRef};

pub struct ScreenCapture {
    path: String,
    due: std::time::Instant,
}

impl ScreenCapture {
    /// `CAPTURE_SCREENSHOT_AFTER` is seconds, matching the windowed build's
    /// variable of the same name, so one command works on either.
    pub fn from_env() -> Option<Self> {
        let path = std::env::var("CAPTURE_SCREENSHOT_TO").ok()?;
        let after: f32 = std::env::var("CAPTURE_SCREENSHOT_AFTER")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(8.0);
        println!("[capture] will write {path} after {after}s");
        Some(Self { path, due: std::time::Instant::now() + std::time::Duration::from_secs_f32(after) })
    }

    pub fn due(&self) -> bool {
        std::time::Instant::now() >= self.due
    }

    /// Call only on a buffer that has already completed, or the texture still
    /// holds the previous frame.
    pub fn write(&self, texture: &TextureRef) {
        let (width, height) = (texture.width() as usize, texture.height() as usize);
        let row = width * 4;
        let mut bytes = vec![0u8; row * height];
        texture.get_bytes(
            bytes.as_mut_ptr() as *mut std::ffi::c_void,
            row as u64,
            MTLRegion {
                origin: MTLOrigin { x: 0, y: 0, z: 0 },
                size: MTLSize { width: width as u64, height: height as u64, depth: 1 },
            },
            0,
        );
        // Drawables are BGRA; PNG wants RGBA.
        for pixel in bytes.chunks_exact_mut(4) {
            pixel.swap(0, 2);
        }
        match image::RgbaImage::from_raw(width as u32, height as u32, bytes) {
            Some(img) => match img.save(&self.path) {
                Ok(()) => println!("[capture] wrote {} ({width}x{height})", self.path),
                Err(err) => println!("[capture] could not write {}: {err}", self.path),
            },
            None => println!("[capture] readback was the wrong size"),
        }
    }
}
