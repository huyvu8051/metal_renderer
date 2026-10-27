
//! Apple GPUs are tile-based: a target that is cleared at the start of a pass
//! and never stored or sampled afterwards lives entirely in tile memory and
//! needs no DRAM behind it. Depth here is Clear + DontCare and the multisample
//! colour resolves into the scene texture, so all three qualify.

use metal::{Device, MTLPixelFormat};

pub(crate) struct MsaaTargets {
    sample_count: u64,
    depth: Option<metal::Texture>,
    depth_no_msaa: Option<metal::Texture>,
    color: Option<metal::Texture>,
}

/// Memoryless exists only on Apple-family GPUs; an Intel Mac has to keep the
/// DRAM behind these.
fn tile_only_storage(device: &Device) -> metal::MTLStorageMode {
    if device.supports_family(metal::MTLGPUFamily::Apple1) {
        metal::MTLStorageMode::Memoryless
    } else {
        metal::MTLStorageMode::Private
    }
}

impl MsaaTargets {
    pub(crate) fn new(sample_count: u64) -> Self {
        Self { sample_count, depth: None, depth_no_msaa: None, color: None }
    }

    pub(crate) fn ensure_depth(&mut self, device: &Device, width: u64, height: u64) -> metal::Texture {
        if let Some(tex) = &self.depth {
            if tex.width() == width && tex.height() == height {
                return tex.clone();
            }
        }
        let descriptor = metal::TextureDescriptor::new();
        descriptor.set_texture_type(metal::MTLTextureType::D2Multisample);
        descriptor.set_pixel_format(MTLPixelFormat::Depth32Float);
        descriptor.set_width(width);
        descriptor.set_height(height);
        descriptor.set_sample_count(self.sample_count);
        descriptor.set_storage_mode(tile_only_storage(device));
        descriptor.set_usage(metal::MTLTextureUsage::RenderTarget);
        let tex = device.new_texture(&descriptor);
        self.depth = Some(tex.clone());
        tex
    }

    pub(crate) fn ensure_color(&mut self, device: &Device, width: u64, height: u64) -> metal::Texture {
        if let Some(tex) = &self.color {
            if tex.width() == width && tex.height() == height {
                return tex.clone();
            }
        }
        let descriptor = metal::TextureDescriptor::new();
        descriptor.set_texture_type(metal::MTLTextureType::D2Multisample);
        descriptor.set_pixel_format(MTLPixelFormat::BGRA8Unorm);
        descriptor.set_width(width);
        descriptor.set_height(height);
        descriptor.set_sample_count(self.sample_count);
        descriptor.set_storage_mode(tile_only_storage(device));
        descriptor.set_usage(metal::MTLTextureUsage::RenderTarget);
        let tex = device.new_texture(&descriptor);
        self.color = Some(tex.clone());
        tex
    }

    pub(crate) fn ensure_depth_no_msaa(&mut self, device: &Device, width: u64, height: u64) -> metal::Texture {
        if let Some(tex) = &self.depth_no_msaa {
            if tex.width() == width && tex.height() == height {
                return tex.clone();
            }
        }
        let descriptor = metal::TextureDescriptor::new();
        descriptor.set_pixel_format(MTLPixelFormat::Depth32Float);
        descriptor.set_width(width);
        descriptor.set_height(height);
        descriptor.set_storage_mode(tile_only_storage(device));
        descriptor.set_usage(metal::MTLTextureUsage::RenderTarget);
        let tex = device.new_texture(&descriptor);
        self.depth_no_msaa = Some(tex.clone());
        tex
    }
}
