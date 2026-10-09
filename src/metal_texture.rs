
use metal::{Device, MTLPixelFormat, MTLRegion};

/// Uploads RGBA8 pixels, with a full mip chain unless the caller opts out.
///
/// Mips are not cosmetic here: a minified texture sampled at level 0 misses the
/// GPU's texture cache on nearly every fragment, and that bandwidth — not the
/// shader's arithmetic — is what a map of this size runs out of first.
pub fn upload_texture(device: &Device, width: u32, height: u32, rgba: &[u8]) -> metal::Texture {
    upload_texture_with_mips(device, width, height, rgba, true)
}

pub fn upload_texture_with_mips(
    device: &Device,
    width: u32,
    height: u32,
    rgba: &[u8],
    mipmapped: bool,
) -> metal::Texture {
    let levels = if mipmapped { mip_level_count(width, height) } else { 1 };

    let descriptor = metal::TextureDescriptor::new();
    descriptor.set_pixel_format(MTLPixelFormat::RGBA8Unorm);
    descriptor.set_width(width as u64);
    descriptor.set_height(height as u64);
    descriptor.set_mipmap_level_count(levels as u64);
    descriptor.set_storage_mode(metal::MTLStorageMode::Shared);
    descriptor.set_usage(metal::MTLTextureUsage::ShaderRead);
    let texture = device.new_texture(&descriptor);

    let region = MTLRegion::new_2d(0, 0, width as u64, height as u64);
    texture.replace_region(region, 0, rgba.as_ptr() as *const _, (width * 4) as u64);

    if levels > 1 {
        let mut level_width = width;
        let mut level_height = height;
        let mut source = rgba.to_vec();
        for level in 1..levels {
            let (next, next_width, next_height) = downsample_half(&source, level_width, level_height);
            let region = MTLRegion::new_2d(0, 0, next_width as u64, next_height as u64);
            texture.replace_region(region, level as u64, next.as_ptr() as *const _, (next_width * 4) as u64);
            source = next;
            level_width = next_width;
            level_height = next_height;
        }
    }

    texture
}

fn mip_level_count(width: u32, height: u32) -> u32 {
    32 - width.max(height).max(1).leading_zeros()
}

/// Plain 2x2 box filter. `replace_region` cannot generate mips on its own and a
/// blit encoder would need a command buffer here, so the chain is built on the
/// CPU once at load.
fn downsample_half(source: &[u8], width: u32, height: u32) -> (Vec<u8>, u32, u32) {
    let next_width = (width / 2).max(1);
    let next_height = (height / 2).max(1);
    let mut out = vec![0u8; (next_width * next_height * 4) as usize];

    for y in 0..next_height {
        for x in 0..next_width {
            let x0 = (x * 2).min(width - 1);
            let y0 = (y * 2).min(height - 1);
            let x1 = (x0 + 1).min(width - 1);
            let y1 = (y0 + 1).min(height - 1);
            let texel = |px: u32, py: u32, channel: usize| source[(((py * width) + px) * 4) as usize + channel] as u32;
            for channel in 0..4 {
                let sum = texel(x0, y0, channel) + texel(x1, y0, channel) + texel(x0, y1, channel) + texel(x1, y1, channel);
                out[(((y * next_width) + x) * 4) as usize + channel] = (sum / 4) as u8;
            }
        }
    }

    (out, next_width, next_height)
}
