//! A KTX2 of ASTC blocks, as a PNG you can look at.
//!
//! Nothing here decodes anything. The blocks go to the GPU exactly as the game
//! sends them, a full-screen triangle samples that texture into an RGBA target
//! and the result is read back -- so what lands in the PNG is what the sampler
//! would have handed the shader, lossy compression and all.
//!
//!   cargo run -p metal_renderer --bin ktx2_png -- <in.ktx2> [out.png]
//!
//! Only level 0. The mip chain is in the file but nothing here wants it.

#[cfg(target_vendor = "apple")]
fn main() {
    use metal::{MTLOrigin, MTLPixelFormat, MTLRegion, MTLSize};
    use metal_renderer::snapshot::AstcBlock;

    let mut args = std::env::args().skip(1);
    let Some(input) = args.next() else {
        eprintln!("usage: ktx2_png <in.ktx2> [out.png]");
        std::process::exit(2);
    };
    let output = args.next().unwrap_or_else(|| {
        std::path::Path::new(&input).with_extension("png").to_string_lossy().into_owned()
    });

    let bytes = std::fs::read(&input).unwrap_or_else(|err| {
        eprintln!("cannot read {input}: {err}");
        std::process::exit(1);
    });
    let Some(ktx) = metal_renderer::metal_texture::parse_ktx2(&bytes) else {
        eprintln!("{input} is not uncompressed ASTC 4x4/6x6/8x8");
        std::process::exit(1);
    };
    let (block_name, srgb) = match ktx.block {
        AstcBlock::B4x4 { srgb } => ("4x4", srgb),
        AstcBlock::B6x6 { srgb } => ("6x6", srgb),
        AstcBlock::B8x8 { srgb } => ("8x8", srgb),
    };
    let bits_per_texel = match ktx.block {
        AstcBlock::B4x4 { .. } => 128.0 / 16.0,
        AstcBlock::B6x6 { .. } => 128.0 / 36.0,
        AstcBlock::B8x8 { .. } => 128.0 / 64.0,
    };
    println!(
        "[ktx2_png] {}x{} ASTC {block_name} {} -- {bits_per_texel:.2} bits a texel",
        ktx.width,
        ktx.height,
        if srgb { "sRGB" } else { "linear" },
    );

    let device = metal::Device::system_default().expect("no Metal device");
    let source = metal_renderer::metal_texture::upload_astc(
        &device,
        ktx.width,
        ktx.height,
        &bytes[ktx.levels[0].clone()],
        ktx.block,
        1,
    );

    // Matching the source's colour space on the way out. An sRGB texture
    // samples to linear, so an sRGB target encodes it back and the PNG carries
    // the bytes that were authored; a linear one -- a metallic-roughness map --
    // must not go through that, or its channels come out meaning nothing.
    let format = if srgb { MTLPixelFormat::RGBA8Unorm_sRGB } else { MTLPixelFormat::RGBA8Unorm };
    let descriptor = metal::TextureDescriptor::new();
    descriptor.set_pixel_format(format);
    descriptor.set_width(ktx.width as u64);
    descriptor.set_height(ktx.height as u64);
    descriptor.set_storage_mode(metal::MTLStorageMode::Shared);
    descriptor.set_usage(metal::MTLTextureUsage::RenderTarget | metal::MTLTextureUsage::ShaderRead);
    let target = device.new_texture(&descriptor);

    const SHADER: &str = r#"
#include <metal_stdlib>
using namespace metal;
struct Varying { float4 position [[position]]; float2 uv; };
vertex Varying blit_vertex(uint id [[vertex_id]]) {
    // One triangle larger than the viewport, so there is no vertex buffer
    // and no seam down the middle of a two-triangle quad.
    float2 corner = float2((id << 1) & 2, id & 2);
    Varying out;
    out.position = float4(corner * 2.0 - 1.0, 0.0, 1.0);
    out.uv = float2(corner.x, 1.0 - corner.y);
    return out;
}
fragment float4 blit_fragment(Varying in [[stage_in]],
                              texture2d<float> src [[texture(0)]],
                              sampler smp [[sampler(0)]]) {
    return src.sample(smp, in.uv);
}
"#;
    let library = device
        .new_library_with_source(SHADER, &metal::CompileOptions::new())
        .expect("blit shader failed to compile");
    let pipeline_descriptor = metal::RenderPipelineDescriptor::new();
    pipeline_descriptor.set_vertex_function(Some(&library.get_function("blit_vertex", None).unwrap()));
    pipeline_descriptor.set_fragment_function(Some(&library.get_function("blit_fragment", None).unwrap()));
    pipeline_descriptor.color_attachments().object_at(0).unwrap().set_pixel_format(format);
    let pipeline = device.new_render_pipeline_state(&pipeline_descriptor).expect("no pipeline");

    // Nearest, so the PNG shows the texels the file actually holds rather than
    // a smoothed read of them -- the question being asked of it is usually how
    // much detail survived the compression.
    let sampler_descriptor = metal::SamplerDescriptor::new();
    sampler_descriptor.set_min_filter(metal::MTLSamplerMinMagFilter::Nearest);
    sampler_descriptor.set_mag_filter(metal::MTLSamplerMinMagFilter::Nearest);
    sampler_descriptor.set_address_mode_s(metal::MTLSamplerAddressMode::ClampToEdge);
    sampler_descriptor.set_address_mode_t(metal::MTLSamplerAddressMode::ClampToEdge);
    let sampler = device.new_sampler(&sampler_descriptor);

    let queue = device.new_command_queue();
    let command_buffer = queue.new_command_buffer();
    let pass = metal::RenderPassDescriptor::new();
    let attachment = pass.color_attachments().object_at(0).unwrap();
    attachment.set_texture(Some(&target));
    attachment.set_load_action(metal::MTLLoadAction::DontCare);
    attachment.set_store_action(metal::MTLStoreAction::Store);
    let encoder = command_buffer.new_render_command_encoder(pass);
    encoder.set_render_pipeline_state(&pipeline);
    encoder.set_fragment_texture(0, Some(&source));
    encoder.set_fragment_sampler_state(0, Some(&sampler));
    encoder.draw_primitives(metal::MTLPrimitiveType::Triangle, 0, 3);
    encoder.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();

    let (width, height) = (ktx.width as usize, ktx.height as usize);
    let row = width * 4;
    let mut pixels = vec![0u8; row * height];
    target.get_bytes(
        pixels.as_mut_ptr() as *mut std::ffi::c_void,
        row as u64,
        MTLRegion {
            origin: MTLOrigin { x: 0, y: 0, z: 0 },
            size: MTLSize { width: width as u64, height: height as u64, depth: 1 },
        },
        0,
    );
    match image::RgbaImage::from_raw(width as u32, height as u32, pixels) {
        Some(image) => match image.save(&output) {
            Ok(()) => println!("[ktx2_png] wrote {output}"),
            Err(err) => {
                eprintln!("cannot write {output}: {err}");
                std::process::exit(1);
            }
        },
        None => {
            eprintln!("readback was the wrong size");
            std::process::exit(1);
        }
    }
}

#[cfg(not(target_vendor = "apple"))]
fn main() {
    eprintln!("ktx2_png needs Metal");
    std::process::exit(1);
}
