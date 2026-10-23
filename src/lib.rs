//! A hand-written Metal renderer driven by plain Bevy data, with no knowledge
//! of any particular game. A host builds a [`snapshot::RenderSnapshot`] from
//! its own world each frame and hands it to [`native_metal_renderer::NativeMetalRenderer::render_frame`].

pub mod gltf_materials;
#[cfg(target_vendor = "apple")]
pub mod gltf_scene;
#[cfg(target_vendor = "apple")]
pub mod gpu_capture;
pub mod metal_math;
pub mod metal_msaa;
pub mod metal_texture;
pub mod extract;
pub mod snapshot;
pub mod skinned_actor;

#[cfg(target_vendor = "apple")]
pub mod native_metal_renderer;
#[cfg(target_vendor = "apple")]
pub mod skinned_gltf;
