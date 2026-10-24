
use bevy::asset::AssetId;
use bevy::ecs::resource::Resource;
use bevy::image::Image;
use bevy::mesh::Mesh;
use bevy::transform::components::Transform;
#[derive(Resource, Default)]
pub struct DecodedTextureCache(pub std::collections::HashMap<AssetId<Image>, ExtractedTexture>);

#[derive(Clone)]
pub struct ExtractedTexture {

    pub id: AssetId<Image>,
    pub width: u32,
    pub height: u32,

    /// Either RGBA8 or, when `astc` is set, the ASTC blocks exactly as they
    /// came out of the KTX2 file. Apple GPUs sample ASTC directly, so handing
    /// it over untouched is both cheaper and the only way these load at all --
    /// decoding was never the point of compressing them.
    pub rgba: std::sync::Arc<[u8]>,
    pub astc: Option<AstcBlock>,
}

/// The block size, which is all the Metal pixel format needs beyond knowing
/// it is ASTC and whether it is sRGB.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum AstcBlock {
    B4x4 { srgb: bool },
    B6x6 { srgb: bool },
    B8x8 { srgb: bool },
}

#[derive(Clone)]
pub struct ExtractedMesh {
    pub positions: Vec<[f32; 3]>,
    pub normals: Vec<[f32; 3]>,

    pub uvs: Vec<[f32; 2]>,
    /// Lightmap coordinates, one per vertex, or empty. Filled in by the host
    /// from the map's baked atlas -- nothing in the glTF carries it.
    pub uv2: Vec<[f32; 2]>,

    pub indices: Vec<u32>,
    pub base_color: [f32; 4],

    pub emissive: [f32; 4],
    pub texture: Option<ExtractedTexture>,

    pub tangents: Vec<[f32; 4]>,

    pub normal_texture: Option<ExtractedTexture>,

    pub transform: Transform,

    pub visible: bool,
    /// True for the weapon and the hands, which the windowed build keeps on
    /// VIEWMODEL_LAYER. The scope looks at the world from a second point of
    /// view and the gun is not part of that world: drawn there it appears
    /// inside its own eyepiece.
    pub viewmodel: bool,
    /// True for the reticle, on SCOPE_RETICLE_LAYER. The mirror image of the
    /// above -- it belongs to the scope view only, and the main camera does
    /// not carry that layer, so a red mark floating in the middle of the
    /// screen while hip firing is what happens when nobody tells it.
    pub reticle: bool,

    pub is_blend: bool,

    pub mesh_id: AssetId<Mesh>,
}

pub struct CameraSnapshot {
    pub eye: [f32; 3],
    pub rotation: [f32; 4],
    pub yaw: f32,
    pub pitch: f32,
    /// Vertical, radians, straight off the camera's own projection. The
    /// renderer used to hardcode 60 degrees against the game's 45, which makes
    /// everything -- the weapon most noticeably -- sit further away than it
    /// should.
    pub fov: f32,

    pub ads_blend: f32,

    pub scope_zoom: f32,

    pub wall_raise: f32,
}

pub struct ScopeCameraSnapshot {
    pub eye: [f32; 3],
    pub rotation: [f32; 4],

    pub active: bool,

    pub fov: f32,

    pub lens_world_pos: [f32; 3],

    pub lens_local_rotation: [f32; 4],
}

#[derive(Default)]
pub struct RenderSnapshot {
    pub camera: Option<CameraSnapshot>,
    pub scope_camera: Option<ScopeCameraSnapshot>,
    pub skinned_actors: Vec<SkinnedRenderState>,
    pub static_meshes: Vec<ExtractedMesh>,
    pub light: Option<LightSnapshot>,
    pub sun: Option<SunSnapshot>,
    pub map_msaa_on: bool,

    pub shadow_quality: u8,

    pub light_quality: u8,

    pub point_light_quality: u8,

    pub render_resolution_scale: f32,


    pub generic_ui_nodes: Vec<GenericUiNode>,
    pub debug_lines: Vec<DebugLine>,
}

pub struct DebugLine {
    pub a: [f32; 3],
    pub b: [f32; 3],
    pub color: [f32; 4],
}

pub struct GenericUiNode {
    pub center: [f32; 2],
    pub size: [f32; 2],
    /// Clockwise radians, matching `UiTransform::rotation`.
    pub rotation: f32,
    pub background_color: Option<[f32; 4]>,
    pub border_color: Option<[f32; 4]>,
    pub border_width: f32,
    /// Treated as a circle, not a rounded rect, when >= half the shorter side.
    pub corner_radius: f32,
    pub text: Option<GenericUiText>,
    pub image: Option<ExtractedTexture>,
    /// `[x, y, w, h]`, inherited from the nearest clipping ancestor (not self).
    pub clip_rect: Option<[f32; 4]>,
    /// Bevy's own computed paint order (honours `GlobalZIndex`/hierarchy); nodes
    /// are emitted sorted by it so painting follows real UI stacking, not spawn order.
    pub stack_index: u32,
}

pub struct GenericUiText {
    pub content: String,
    pub color: [f32; 4],
    pub font_size: f32,
}

pub struct SunSnapshot {

    pub direction: [f32; 3],
    pub color: [f32; 3],
}

pub struct LightSnapshot {
    pub position: [f32; 3],

    pub color: [f32; 3],
    pub range: f32,
}

pub struct SkinnedRenderState {
    pub transform: Transform,
    pub joint_matrices: Vec<[[f32; 4]; 4]>,
    pub model: &'static str,
}

impl Default for CameraSnapshot {
    fn default() -> Self {
        Self {
            eye: [0.0, 0.0, 0.0],
            rotation: [0.0, 0.0, 0.0, 1.0],
            yaw: 0.0,
            pitch: 0.0,
            fov: std::f32::consts::FRAC_PI_4,
            ads_blend: 0.0,
            scope_zoom: 8.0,
            wall_raise: 0.0,
        }
    }
}
