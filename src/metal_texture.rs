
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
    let levels = if mipmapped { mip_level_count_for(width, height) } else { 1 };

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

fn mip_level_count_for(width: u32, height: u32) -> u32 {
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

/// Uploads ASTC blocks untouched. Apple GPUs sample ASTC natively, so there is
/// nothing to decode -- and decoding would undo the reason the textures were
/// compressed in the first place.
///
/// No mip chain is generated: the blocks are already a finished level and the
/// GPU cannot filter compressed data into smaller ones. A KTX2 that carries
/// its own mips would need those passed level by level; the game's do not.
pub fn upload_astc(
    device: &Device,
    width: u32,
    height: u32,
    blocks: &[u8],
    block: crate::snapshot::AstcBlock,
    mip_level_count: u32,
) -> metal::Texture {
    use crate::snapshot::AstcBlock as B;
    let (format, bw, bh) = match block {
        B::B4x4 { srgb } => (
            if srgb { MTLPixelFormat::ASTC_4x4_sRGB } else { MTLPixelFormat::ASTC_4x4_LDR },
            4, 4,
        ),
        B::B6x6 { srgb } => (
            if srgb { MTLPixelFormat::ASTC_6x6_sRGB } else { MTLPixelFormat::ASTC_6x6_LDR },
            6, 6,
        ),
        B::B8x8 { srgb } => (
            if srgb { MTLPixelFormat::ASTC_8x8_sRGB } else { MTLPixelFormat::ASTC_8x8_LDR },
            8, 8,
        ),
    };

    // KTX2 ships these with a full chain and Bevy hands all of it over
    // mip-major. Uploading level zero alone left every minified surface
    // sampling the top level -- see upload_texture's own note on what that
    // costs in texture-cache bandwidth.
    let levels = mip_level_count.max(1).min(mip_level_count_for(width, height));
    let descriptor = metal::TextureDescriptor::new();
    descriptor.set_pixel_format(format);
    descriptor.set_width(width as u64);
    descriptor.set_height(height as u64);
    descriptor.set_mipmap_level_count(levels as u64);
    descriptor.set_usage(metal::MTLTextureUsage::ShaderRead);
    let texture = device.new_texture(&descriptor);

    let mut offset = 0usize;
    let (mut level_width, mut level_height) = (width, height);
    let mut slices: Vec<&[u8]> = Vec::with_capacity(levels as usize);
    for _ in 0..levels {
        // ASTC is 16 bytes a block whatever the block size, so a row of blocks
        // is 16 * ceil(width / block_width).
        let level_bytes =
            16 * level_width.div_ceil(bw) as usize * level_height.div_ceil(bh) as usize;
        // A truncated chain is a broken asset, not a reason to read past the
        // end of it.
        if offset + level_bytes > blocks.len() {
            break;
        }
        slices.push(&blocks[offset..offset + level_bytes]);
        offset += level_bytes;
        level_width = (level_width / 2).max(1);
        level_height = (level_height / 2).max(1);
    }
    write_astc_levels(&texture, width, height, bw, bh, &slices);
    texture
}

/// Uploads a chain whose levels are not contiguous -- KTX2 keeps a byte offset
/// per level and is free to store them in any order.
pub fn upload_astc_from_levels(
    device: &Device,
    width: u32,
    height: u32,
    block: crate::snapshot::AstcBlock,
    levels: &[&[u8]],
) -> metal::Texture {
    use crate::snapshot::AstcBlock as B;
    let (format, bw, bh) = match block {
        B::B4x4 { srgb } => (
            if srgb { MTLPixelFormat::ASTC_4x4_sRGB } else { MTLPixelFormat::ASTC_4x4_LDR },
            4, 4,
        ),
        B::B6x6 { srgb } => (
            if srgb { MTLPixelFormat::ASTC_6x6_sRGB } else { MTLPixelFormat::ASTC_6x6_LDR },
            6, 6,
        ),
        B::B8x8 { srgb } => (
            if srgb { MTLPixelFormat::ASTC_8x8_sRGB } else { MTLPixelFormat::ASTC_8x8_LDR },
            8, 8,
        ),
    };
    let descriptor = metal::TextureDescriptor::new();
    descriptor.set_pixel_format(format);
    descriptor.set_width(width as u64);
    descriptor.set_height(height as u64);
    descriptor.set_mipmap_level_count(levels.len().max(1) as u64);
    descriptor.set_usage(metal::MTLTextureUsage::ShaderRead);
    let texture = device.new_texture(&descriptor);
    write_astc_levels(&texture, width, height, bw, bh, levels);
    texture
}

fn write_astc_levels(
    texture: &metal::Texture,
    width: u32,
    height: u32,
    bw: u32,
    bh: u32,
    levels: &[&[u8]],
) {
    let (mut level_width, mut level_height) = (width, height);
    for (level, bytes) in levels.iter().enumerate() {
        let bytes_per_row = 16 * level_width.div_ceil(bw) as u64;
        if bytes.len() < bytes_per_row as usize * level_height.div_ceil(bh) as usize {
            break;
        }
        texture.replace_region(
            metal::MTLRegion {
                origin: metal::MTLOrigin { x: 0, y: 0, z: 0 },
                size: metal::MTLSize {
                    width: level_width as u64,
                    height: level_height as u64,
                    depth: 1,
                },
            },
            level as u64,
            bytes.as_ptr() as *const _,
            bytes_per_row,
        );
        level_width = (level_width / 2).max(1);
        level_height = (level_height / 2).max(1);
    }
}

/// What a KTX2 file says about itself, enough to hand its blocks to Metal.
pub struct Ktx2 {
    pub width: u32,
    pub height: u32,
    pub block: crate::snapshot::AstcBlock,
    /// Byte range of each mip level, largest first -- the order KTX2's own
    /// level index uses, whatever order the data itself sits in. Metal wants
    /// each level uploaded on its own, so they stay separate.
    pub levels: Vec<std::ops::Range<usize>>,
}

/// Reads the header rather than decoding anything: these are ASTC blocks the
/// GPU samples directly, and decoding them would undo the compression.
///
/// Returns None for anything that is not uncompressed ASTC 4x4/6x6/8x8 --
/// including a supercompressed file, which would need inflating first.
pub fn parse_ktx2(data: &[u8]) -> Option<Ktx2> {
    const MAGIC: &[u8; 12] = b"\xabKTX 20\xbb\r\n\x1a\n";
    if data.len() < 80 || &data[..12] != MAGIC {
        return None;
    }
    let u32_at = |o: usize| -> Option<u32> {
        Some(u32::from_le_bytes(data.get(o..o + 4)?.try_into().ok()?))
    };
    let u64_at = |o: usize| -> Option<u64> {
        Some(u64::from_le_bytes(data.get(o..o + 8)?.try_into().ok()?))
    };

    let vk_format = u32_at(12)?;
    let width = u32_at(20)?;
    let height = u32_at(24)?;
    // Field order after the 12-byte identifier: vkFormat, typeSize,
    // pixelWidth, pixelHeight, pixelDepth, layerCount, faceCount, levelCount,
    // supercompressionScheme -- so levelCount is at 44 from the start of the
    // file and the scheme at 48. Reading these one field late made every file
    // look supercompressed and turned every character grey.
    let level_count = u32_at(12 + 7 * 4)?.max(1);
    let supercompression = u32_at(12 + 8 * 4)?;
    if supercompression != 0 {
        return None;
    }

    // VK_FORMAT_ASTC_*_{UNORM,SRGB}_BLOCK, which run in pairs from 157.
    let block = match vk_format {
        157 => crate::snapshot::AstcBlock::B4x4 { srgb: false },
        158 => crate::snapshot::AstcBlock::B4x4 { srgb: true },
        165 => crate::snapshot::AstcBlock::B6x6 { srgb: false },
        166 => crate::snapshot::AstcBlock::B6x6 { srgb: true },
        171 => crate::snapshot::AstcBlock::B8x8 { srgb: false },
        172 => crate::snapshot::AstcBlock::B8x8 { srgb: true },
        _ => return None,
    };

    // The level index starts after the 80-byte header, three u64 per level,
    // and level 0 is the full-size one.
    let index = 80usize;
    if data.len() < index + level_count as usize * 24 {
        return None;
    }
    let mut levels = Vec::with_capacity(level_count as usize);
    for level in 0..level_count as usize {
        let offset = u64_at(index + level * 24)? as usize;
        let length = u64_at(index + level * 24 + 8)? as usize;
        if offset + length > data.len() {
            return None;
        }
        levels.push(offset..offset + length);
    }
    Some(Ktx2 { width, height, block, levels })
}
