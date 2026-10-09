
use bevy::asset::{AssetId, Assets, Handle};
use bevy::camera::visibility::InheritedVisibility;
use bevy::ecs::query::{Changed, Or, With, Without};
use bevy::ecs::resource::Resource;
use bevy::ecs::system::{Query, Res, ResMut};
use bevy::image::Image;
use bevy::material::AlphaMode;
use bevy::mesh::skinning::SkinnedMesh;
use bevy::mesh::{Mesh, Mesh3d};
use bevy::pbr::{MeshMaterial3d, StandardMaterial};
use bevy::render::render_resource::TextureFormat;
use bevy::transform::components::GlobalTransform;

use crate::snapshot::{DecodedTextureCache, ExtractedMesh, ExtractedTexture};
pub type Q<'w, 's, D, F = ()> = Query<'w, 's, D, F>;

pub type MeshChanged = (
    Without<SkinnedMesh>,
    bevy::ecs::query::Or<(
        bevy::ecs::query::Changed<GlobalTransform>,
        bevy::ecs::query::Changed<MeshMaterial3d<StandardMaterial>>,
        bevy::ecs::query::Changed<InheritedVisibility>,
        bevy::ecs::query::Changed<Mesh3d>,
    )>,
);

pub type MeshExtractParams<'w, 's, F> = (
    Q<
        'w,
        's,
        (
            bevy::ecs::entity::Entity,
            &'static Mesh3d,
            &'static MeshMaterial3d<StandardMaterial>,
            &'static GlobalTransform,
            &'static InheritedVisibility,
        ),
        F,
    >,
    Res<'w, Assets<Mesh>>,
    Res<'w, Assets<StandardMaterial>>,
    Res<'w, Assets<Image>>,
    ResMut<'w, DecodedTextureCache>,
);

#[derive(Resource, Default)]
pub struct StaticMeshCache {
    pub entries: Vec<ExtractedMesh>,
    pub slot_of: std::collections::HashMap<bevy::ecs::entity::Entity, usize>,
    pub free_slots: Vec<usize>,
    pub awaiting_textures: std::collections::HashSet<bevy::ecs::entity::Entity>,
}

pub fn decode_texture(
    handle: &Handle<Image>,
    images: &Assets<Image>,
    texture_cache: &mut DecodedTextureCache,
    used: &mut std::collections::HashSet<AssetId<Image>>,
) -> Option<ExtractedTexture> {
    used.insert(handle.id());
    if let Some(cached) = texture_cache.0.get(&handle.id()) {
        return Some(cached.clone());
    }
    let image = images.get(handle)?;
    let data = image.data.as_ref()?;
    let rgba: std::sync::Arc<[u8]> = match image.texture_descriptor.format {
        TextureFormat::Rgba8UnormSrgb | TextureFormat::Rgba8Unorm => std::sync::Arc::from(data.as_slice()),
        TextureFormat::Bgra8UnormSrgb | TextureFormat::Bgra8Unorm => {
            let mut swapped = data.clone();
            for pixel in swapped.chunks_exact_mut(4) {
                pixel.swap(0, 2);
            }
            std::sync::Arc::from(swapped.as_slice())
        }
        _ => return None,
    };
    let extracted = ExtractedTexture {
        id: handle.id(),
        width: image.texture_descriptor.size.width,
        height: image.texture_descriptor.size.height,
        rgba,
    };
    texture_cache.0.insert(handle.id(), extracted.clone());
    Some(extracted)
}

#[allow(clippy::too_many_arguments)]
#[allow(clippy::too_many_arguments)]
pub fn store_extracted_mesh(
    cache: &mut StaticMeshCache,
    entity: bevy::ecs::entity::Entity,
    mesh3d: &Mesh3d,
    material3d: &MeshMaterial3d<StandardMaterial>,
    transform: &GlobalTransform,
    visibility: &InheritedVisibility,
    meshes: &Assets<Mesh>,
    materials: &Assets<StandardMaterial>,
    images: &Assets<Image>,
    texture_cache: &mut DecodedTextureCache,
    used_texture_ids: &mut std::collections::HashSet<AssetId<Image>>,
    geometry_already_uploaded: &impl Fn(AssetId<Mesh>) -> bool,
) {
    let slot = cache.slot_of.get(&entity).copied();
    let previous = slot.and_then(|slot| cache.entries.get(slot));
    let Some((extracted, textures_pending)) = extract_one_mesh(
        mesh3d, material3d, transform, visibility, meshes, materials, images,
        texture_cache, used_texture_ids, geometry_already_uploaded, previous,
    ) else {
        return;
    };
    if textures_pending {
        cache.awaiting_textures.insert(entity);
    } else {
        cache.awaiting_textures.remove(&entity);
    }
    match slot {
        Some(slot) => cache.entries[slot] = extracted,
        None => match cache.free_slots.pop() {
            Some(slot) => {
                cache.slot_of.insert(entity, slot);
                cache.entries[slot] = extracted;
            }
            None => {
                cache.slot_of.insert(entity, cache.entries.len());
                cache.entries.push(extracted);
            }
        },
    }
}

pub fn extract_one_mesh(
    mesh3d: &Mesh3d,
    material3d: &MeshMaterial3d<StandardMaterial>,
    transform: &GlobalTransform,
    visibility: &InheritedVisibility,
    meshes: &Assets<Mesh>,
    materials: &Assets<StandardMaterial>,
    images: &Assets<Image>,
    texture_cache: &mut DecodedTextureCache,
    used_texture_ids: &mut std::collections::HashSet<AssetId<Image>>,
    geometry_already_uploaded: &impl Fn(AssetId<Mesh>) -> bool,
    previous: Option<&ExtractedMesh>,
) -> Option<(ExtractedMesh, bool)> {
    let material = materials.get(&material3d.0)?;
    let mesh_id = mesh3d.id();

    let (positions, normals, uvs, tangents, indices) = if geometry_already_uploaded(mesh_id) {
        (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new())
    } else {
        let mesh = meshes.get(&mesh3d.0)?;
        let positions = match mesh.attribute(Mesh::ATTRIBUTE_POSITION) {
            Some(bevy::mesh::VertexAttributeValues::Float32x3(p)) => p.clone(),
            _ => return None,
        };
        let normals = match mesh.attribute(Mesh::ATTRIBUTE_NORMAL) {
            Some(bevy::mesh::VertexAttributeValues::Float32x3(n)) => n.clone(),
            _ => Vec::new(),
        };
        let uvs = match mesh.attribute(Mesh::ATTRIBUTE_UV_0) {
            Some(bevy::mesh::VertexAttributeValues::Float32x2(uv)) => uv.clone(),
            _ => Vec::new(),
        };
        let tangents = match mesh.attribute(Mesh::ATTRIBUTE_TANGENT) {
            Some(bevy::mesh::VertexAttributeValues::Float32x4(t)) => t.clone(),
            _ => Vec::new(),
        };
        let indices = mesh.indices().map(|i| i.iter().map(|v| v as u32).collect()).unwrap_or_default();
        (positions, normals, uvs, tangents, indices)
    };

    let has_tangents = if positions.is_empty() {
        previous.is_some_and(|p| p.normal_texture.is_some())
    } else {
        !tangents.is_empty()
    };
    let texture = material
        .base_color_texture
        .as_ref()
        .and_then(|handle| decode_texture(handle, images, texture_cache, used_texture_ids));
    let normal_texture = has_tangents
        .then(|| {
            material
                .normal_map_texture
                .as_ref()
                .and_then(|handle| decode_texture(handle, images, texture_cache, used_texture_ids))
        })
        .flatten();

    let textures_pending = material.base_color_texture.is_some() && texture.is_none()
        || has_tangents && material.normal_map_texture.is_some() && normal_texture.is_none();

    Some((ExtractedMesh {
        positions,
        normals,
        uvs,
        indices,
        base_color: {
            let s = material.base_color.to_srgba();
            [s.red, s.green, s.blue, s.alpha]
        },
        emissive: [
            material.emissive.red,
            material.emissive.green,
            material.emissive.blue,
            material.emissive.alpha,
        ],
        texture,
        tangents,
        normal_texture,
        transform: transform.compute_transform(),
        visible: visibility.get(),
        is_blend: matches!(
            material.alpha_mode,
            AlphaMode::Blend | AlphaMode::Premultiplied | AlphaMode::Add | AlphaMode::Multiply
        ),
        mesh_id,
    }, textures_pending))
}



