//! Writing a `.gputrace` the way Xcode's Metal Debugger would, from inside the
//! app.
//!
//! Everything this crate measures by hand — timers around phases, command
//! buffer timestamps — exists because the terminal cannot open Xcode. A capture
//! sidesteps that: the file carries per-draw timings, the shader profiler's
//! per-line costs and the vertex/fragment split, and whoever has Xcode opens it.
//!
//! Capturing needs the app to allow it, which on iOS means
//! `MetalCaptureEnabled` in Info.plist.

use metal::{CaptureDescriptor, CaptureManager, Device, MTLCaptureDestination};

/// Captures the next frames to `<app container>/Documents/<name>.gputrace`,
/// where `devicectl device copy from` can fetch it.
pub struct FrameCapture {
    frames_left: u32,
    path: std::path::PathBuf,
}

impl FrameCapture {
    /// `GFX_CAPTURE=<n>` captures the first n frames after startup.
    pub fn from_env(device: &Device, queue: &metal::CommandQueueRef) -> Option<Self> {
        let frames: u32 = std::env::var("GFX_CAPTURE").ok()?.parse().ok()?;
        if frames == 0 {
            return None;
        }

        let manager = CaptureManager::shared();
        if !manager.supports_destination(MTLCaptureDestination::GpuTraceDocument) {
            println!("[capture] this build cannot write a .gputrace — is MetalCaptureEnabled set?");
            return None;
        }

        let path = documents_dir()?.join("frame.gputrace");
        // Metal refuses to overwrite, so clear any trace from a previous run.
        let _ = std::fs::remove_dir_all(&path);

        let descriptor = CaptureDescriptor::new();
        descriptor.set_capture_command_queue(queue);
        descriptor.set_destination(MTLCaptureDestination::GpuTraceDocument);
        descriptor.set_output_url(&path);
        let _ = device;

        match manager.start_capture(&descriptor) {
            Ok(()) => {
                println!("[capture] capturing {frames} frames to {}", path.display());
                Some(Self { frames_left: frames, path })
            }
            Err(err) => {
                println!("[capture] could not start: {err}");
                None
            }
        }
    }

    /// Call once per frame. Stops the capture when the requested frames are in.
    pub fn frame_done(&mut self) -> bool {
        if self.frames_left == 0 {
            return true;
        }
        self.frames_left -= 1;
        if self.frames_left == 0 {
            CaptureManager::shared().stop_capture();
            println!("[capture] written to {}", self.path.display());
            return true;
        }
        false
    }
}

fn documents_dir() -> Option<std::path::PathBuf> {
    let home = std::env::var("HOME").ok()?;
    let documents = std::path::Path::new(&home).join("Documents");
    std::fs::create_dir_all(&documents).ok()?;
    Some(documents)
}
