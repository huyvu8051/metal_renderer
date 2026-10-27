//! Loading a glTF straight into Metal buffers, with node transforms baked into
//! world-space vertices at load time. This is the path the renderer used before
//! it was driven from ECS data, kept because it is the control case: the same
//! shader and the same draw loop, fed by a list built once instead of a
//! snapshot rebuilt every frame.

use metal::{Device, MTLResourceOptions};

#[repr(C)]
#[derive(Copy, Clone)]
pub struct GltfVertex {
    pub position: [f32; 3],
    /// snorm16: one part in 32767 of a unit vector, three orders finer than
    /// the lighting resolves, for a third of the bytes f32x3 took.
    pub normal: [i16; 4],
    pub uv: [f32; 2],
    /// xyz the tangent direction as snorm16, w glTF's handedness sign.
    pub tangent: [i16; 4],
    /// Lightmap coordinates, already normalised to 0..1 by the baker, as
    /// unorm16 -- 1/65535 against a 2048-texel atlas. Zero for anything the
    /// bake did not cover, which the shader reads as "no lightmap here" and
    /// falls back to the flat ambient term.
    pub uv2: [u16; 2],
}

/// A unit vector plus a sign, as the shader reads it back.
pub fn snorm16(v: [f32; 3], w: f32) -> [i16; 4] {
    let q = |x: f32| (x.clamp(-1.0, 1.0) * 32767.0).round() as i16;
    [q(v[0]), q(v[1]), q(v[2]), q(w)]
}

pub fn unorm16(v: [f32; 2]) -> [u16; 2] {
    let q = |x: f32| (x.clamp(0.0, 1.0) * 65535.0).round() as u16;
    [q(v[0]), q(v[1])]
}

/// The whole map in one vertex buffer and one index buffer, each mesh an
/// offset into them.
///
/// One buffer per mesh was worth measuring and was worth 0.44 ms a frame on
/// desert_city's 2363 primitives. The driver keeps an NSMutableSet of the
/// resources resident on an encoder, so every *distinct* buffer is a hash and
/// an insert; `sample` showed IOGPUResourceListAddResource and __rehashs near
/// the top of the profile, and both vanish once there is only one of each.
pub struct GltfScene {
    pub vertices: metal::Buffer,
    pub indices: metal::Buffer,
    pub meshes: Vec<GltfMesh>,
}

pub struct GltfMesh {
    pub emissive: [f32; 4],
    /// From the material's `alphaMode`. The ECS path reads this too, and a
    /// control case that draws transparent surfaces as opaque is not drawing
    /// the same frame.
    pub is_blend: bool,
    /// World-space centre, for sorting the transparent draws back to front.
    pub centre: [f32; 3],
    /// Reported so the two loaders' vertex counts can be compared.
    /// Distance from `centre` to the furthest vertex. Measured here because
    /// the walk already has the positions in hand, and a frustum test needs a
    /// radius as much as it needs a centre.
    pub radius: f32,
    pub vertex_count: usize,
    /// Byte offsets into [`GltfScene`]'s two buffers. Indices stay relative to
    /// the mesh's own first vertex, which is what binding the vertex buffer at
    /// an offset already means.
    pub vertex_offset: u64,
    pub index_offset: u64,
    pub index_count: usize,
    pub base_color_texture: Option<metal::Texture>,
    pub base_color_factor: [f32; 4],
}

type Mat4 = [[f32; 4]; 4];

fn mat4_identity() -> Mat4 {
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
}

fn mat4_mul(a: Mat4, b: Mat4) -> Mat4 {
    let mut out = [[0.0f32; 4]; 4];
    for column in 0..4 {
        for row in 0..4 {
            out[column][row] = (0..4).map(|k| a[k][row] * b[column][k]).sum();
        }
    }
    out
}

fn mat4_from_quat_translation_scale(t: [f32; 3], r: [f32; 4], s: [f32; 3]) -> Mat4 {
    let [x, y, z, w] = r;
    let (x2, y2, z2) = (x + x, y + y, z + z);
    let (xx, xy, xz) = (x * x2, x * y2, x * z2);
    let (yy, yz, zz) = (y * y2, y * z2, z * z2);
    let (wx, wy, wz) = (w * x2, w * y2, w * z2);
    [
        [(1.0 - (yy + zz)) * s[0], (xy + wz) * s[0], (xz - wy) * s[0], 0.0],
        [(xy - wz) * s[1], (1.0 - (xx + zz)) * s[1], (yz + wx) * s[1], 0.0],
        [(xz + wy) * s[2], (yz - wx) * s[2], (1.0 - (xx + yy)) * s[2], 0.0],
        [t[0], t[1], t[2], 1.0],
    ]
}

fn transform_point(m: Mat4, p: [f32; 3]) -> [f32; 3] {
    [
        m[0][0] * p[0] + m[1][0] * p[1] + m[2][0] * p[2] + m[3][0],
        m[0][1] * p[0] + m[1][1] * p[1] + m[2][1] * p[2] + m[3][1],
        m[0][2] * p[0] + m[1][2] * p[1] + m[2][2] * p[2] + m[3][2],
    ]
}

fn transform_direction(m: Mat4, n: [f32; 3]) -> [f32; 3] {
    let out = [
        m[0][0] * n[0] + m[1][0] * n[1] + m[2][0] * n[2],
        m[0][1] * n[0] + m[1][1] * n[1] + m[2][1] * n[2],
        m[0][2] * n[0] + m[1][2] * n[1] + m[2][2] * n[2],
    ];
    let length = (out[0] * out[0] + out[1] * out[1] + out[2] * out[2]).sqrt();
    if length > 1e-6 { [out[0] / length, out[1] / length, out[2] / length] } else { [0.0, 1.0, 0.0] }
}

pub fn to_rgba8(image: &gltf::image::Data) -> Vec<u8> {
    use gltf::image::Format;
    let pixel_count = (image.width * image.height) as usize;
    let mut out = Vec::with_capacity(pixel_count * 4);
    match image.format {
        Format::R8 => {
            for &r in &image.pixels {
                out.extend_from_slice(&[r, r, r, 255]);
            }
        }
        Format::R8G8 => {
            for chunk in image.pixels.chunks_exact(2) {
                out.extend_from_slice(&[chunk[0], chunk[1], 0, 255]);
            }
        }
        Format::R8G8B8 => {
            for chunk in image.pixels.chunks_exact(3) {
                out.extend_from_slice(&[chunk[0], chunk[1], chunk[2], 255]);
            }
        }
        Format::R8G8B8A8 => {
            out.extend_from_slice(&image.pixels);
        }
        _ => {

            out.resize(pixel_count * 4, 180);
        }
    }
    out
}

fn walk_node(
    node: &gltf::Node,
    parent_transform: Mat4,
    buffers: &[gltf::buffer::Data],
    textures: &[Option<metal::Texture>],
    meshes: &mut Vec<GltfMesh>,
    vertex_bytes: &mut Vec<u8>,
    index_bytes: &mut Vec<u8>,
) {
    let (t, r, s) = node.transform().decomposed();
    let local = mat4_from_quat_translation_scale(t, r, s);
    let world = mat4_mul(parent_transform, local);

    if let Some(mesh) = node.mesh() {
        for primitive in mesh.primitives() {
            let reader = primitive.reader(|buffer| Some(&buffers[buffer.index()]));
            let Some(positions) = reader.read_positions() else { continue };
            let positions: Vec<[f32; 3]> = positions.collect();
            let normals: Vec<[f32; 3]> = reader
                .read_normals()
                .map(|n| n.collect())
                .unwrap_or_else(|| vec![[0.0, 1.0, 0.0]; positions.len()]);
            let uvs: Vec<[f32; 2]> = reader
                .read_tex_coords(0)
                .map(|uv| uv.into_f32().collect())
                .unwrap_or_else(|| vec![[0.0, 0.0]; positions.len()]);
            let tangents: Vec<[f32; 4]> = reader
                .read_tangents()
                .map(|t| t.collect())
                .unwrap_or_else(|| vec![[1.0, 0.0, 0.0, 1.0]; positions.len()]);
            let Some(indices) = reader.read_indices() else { continue };
            let indices: Vec<u32> = indices.into_u32().collect();
            if positions.is_empty() || indices.is_empty() {
                continue;
            }

            let vertices: Vec<GltfVertex> = (0..positions.len())
                .map(|i| {
                    let tangent = transform_direction(world, [tangents[i][0], tangents[i][1], tangents[i][2]]);
                    GltfVertex {
                        // Skinned and scene glTFs carry no lightmap; only the
                        // map does, and it comes in through the host.
                        uv2: [0, 0],
                        position: transform_point(world, positions[i]),
                        normal: snorm16(transform_direction(world, normals[i]), 0.0),
                        uv: uvs[i],
                        tangent: snorm16(tangent, tangents[i][3]),
                    }
                })
                .collect();

            let vertex_offset = vertex_bytes.len() as u64;
            let index_offset = index_bytes.len() as u64;
            vertex_bytes.extend_from_slice(unsafe {
                std::slice::from_raw_parts(
                    vertices.as_ptr() as *const u8,
                    vertices.len() * std::mem::size_of::<GltfVertex>(),
                )
            });
            index_bytes.extend_from_slice(unsafe {
                std::slice::from_raw_parts(
                    indices.as_ptr() as *const u8,
                    indices.len() * std::mem::size_of::<u32>(),
                )
            });

            let pbr = primitive.material().pbr_metallic_roughness();
            let base_color_texture = pbr
                .base_color_texture()
                .and_then(|info| textures.get(info.texture().source().index()).cloned().flatten());

            let centre = {
                let mut sum = [0.0f64; 3];
                for v in &vertices {
                    sum[0] += v.position[0] as f64;
                    sum[1] += v.position[1] as f64;
                    sum[2] += v.position[2] as f64;
                }
                let n = vertices.len().max(1) as f64;
                [(sum[0] / n) as f32, (sum[1] / n) as f32, (sum[2] / n) as f32]
            };
            let radius = vertices
                .iter()
                .map(|v| {
                    let d = [
                        v.position[0] - centre[0],
                        v.position[1] - centre[1],
                        v.position[2] - centre[2],
                    ];
                    (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt()
                })
                .fold(0.0f32, f32::max);

            meshes.push(GltfMesh {
                vertex_count: vertices.len(),
                vertex_offset,
                index_offset,
                radius,
                index_count: indices.len(),
                base_color_texture,
                base_color_factor: pbr.base_color_factor(),
                centre,
                is_blend: matches!(
                    primitive.material().alpha_mode(),
                    gltf::material::AlphaMode::Blend
                ),
                emissive: {
                    let e = primitive.material().emissive_factor();
                    [e[0], e[1], e[2], 0.0]
                },
            });
        }
    }

    for child in node.children() {
        walk_node(&child, world, buffers, textures, meshes, vertex_bytes, index_bytes);
    }
}

pub fn load_gltf_scene(device: &Device, path: &std::path::Path) -> GltfScene {
    let (document, buffers, images) = match gltf::import(path) {
        Ok(result) => result,
        Err(err) => {
            eprintln!("[gltf_scene] failed to load {path:?}: {err}");
            return GltfScene::empty(device);
        }
    };

    let textures: Vec<Option<metal::Texture>> = images
        .iter()
        .map(|image| {
            Some(crate::metal_texture::upload_texture(device, image.width, image.height, &to_rgba8(image)))
        })
        .collect();

    let mut meshes = Vec::new();
    let mut vertex_bytes: Vec<u8> = Vec::new();
    let mut index_bytes: Vec<u8> = Vec::new();
    let Some(scene) = document.default_scene().or_else(|| document.scenes().next()) else {
        eprintln!("[gltf_scene] {path:?} has no scenes");
        return GltfScene::empty(device);
    };
    for node in scene.nodes() {
        walk_node(
            &node,
            mat4_identity(),
            &buffers,
            &textures,
            &mut meshes,
            &mut vertex_bytes,
            &mut index_bytes,
        );
    }
    let indices: usize = meshes.iter().map(|m| m.index_count).sum();
    let vertices: usize = meshes.iter().map(|m| m.vertex_count).sum();
    println!(
        "[gltf_scene] loaded {} meshes ({} textures), {vertices} unique vertices, {indices} indices from {path:?}",
        meshes.len(),
        textures.len()
    );
    GltfScene {
        vertices: new_buffer(device, &vertex_bytes),
        indices: new_buffer(device, &index_bytes),
        meshes,
    }
}

/// Metal rejects a zero-length buffer, and an empty scene has to stay a valid
/// one -- the loader already returns early on a bad file.
fn new_buffer(device: &Device, bytes: &[u8]) -> metal::Buffer {
    device.new_buffer_with_data(
        if bytes.is_empty() { [0u8; 4].as_ptr() } else { bytes.as_ptr() } as *const _,
        bytes.len().max(4) as u64,
        MTLResourceOptions::StorageModeShared,
    )
}

impl GltfScene {
    fn empty(device: &Device) -> Self {
        Self { vertices: new_buffer(device, &[]), indices: new_buffer(device, &[]), meshes: Vec::new() }
    }
}
