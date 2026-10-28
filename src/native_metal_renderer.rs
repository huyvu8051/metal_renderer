
use bevy::asset::AssetId;
use bevy::image::Image;
use bevy::mesh::Mesh;
use core_graphics::geometry::{CGRect, CGSize};
use metal::foreign_types::ForeignType;
use metal::{
    CAMetalLayer, CommandQueue, CompileOptions, Device, MTLClearColor, MTLIndexType, MTLLoadAction, MTLPixelFormat,
    MTLPrimitiveType, MTLResourceOptions, MTLScissorRect, MTLStoreAction, MetalLayer, RenderPassDescriptor,
    RenderPipelineDescriptor, RenderPipelineState,
};
use objc::{runtime::Object, *};

use super::metal_math::{look_at, mat_mul, orthographic, perspective};
use super::snapshot::{ExtractedMesh, LightSnapshot, RenderSnapshot, SunSnapshot};

/// Per-pixel work compiled into the scene shader. These are compile-time
/// rather than uniforms because the cost is the fetches and the ALU, which a
/// runtime flag would still pay for.
#[derive(Copy, Clone)]
pub struct ShaderFeatures {
    pub shadows: bool,
    pub normal_map: bool,
    pub specular: bool,
    pub point_light: bool,
    /// Replace the fragment shader with a constant, to measure the floor.
    pub flat: bool,
}

impl Default for ShaderFeatures {
    fn default() -> Self {
        Self { shadows: true, normal_map: true, specular: true, point_light: true, flat: false }
    }
}

impl ShaderFeatures {
    pub fn defines(&self) -> String {
        format!(
            "#define FEATURE_SHADOWS {}\n#define FEATURE_NORMAL_MAP {}\n#define FEATURE_SPECULAR {}\n#define FEATURE_POINT_LIGHT {}\n#define FEATURE_FLAT {}\n",
            self.shadows as u8, self.normal_map as u8, self.specular as u8, self.point_light as u8, self.flat as u8
        )
    }
}

/// A skinned glTF for the renderer to preload at startup.
pub struct SkinnedModelSpec {
    pub name: &'static str,
    /// Relative to the bundle's `assets/` directory.
    pub asset_path: &'static str,
    pub scale: f32,
    pub y_offset: f32,
    pub facing_radians: f32,
}

#[derive(Copy, Clone)]
struct LightUniform {
    position: [f32; 3],
    range: f32,

    color: [f32; 4],

    sun_dir: [f32; 3],
    _sun_pad: f32,
    sun_color: [f32; 4],
}

impl LightUniform {

    fn from_snapshots(light: Option<&LightSnapshot>, sun: Option<&SunSnapshot>, sun_fraction: f32, point_fraction: f32) -> Self {
        let (position, range, color_base) = match light {
            Some(l) => (l.position, l.range, [l.color[0], l.color[1], l.color[2], 0.0]),
            None => (FALLBACK_LIGHT.position, FALLBACK_LIGHT.range, FALLBACK_LIGHT.color),
        };
        let color = [color_base[0] * point_fraction, color_base[1] * point_fraction, color_base[2] * point_fraction, color_base[3]];
        let (sun_dir, sun_color_base) = match sun {
            Some(s) => (s.direction, [s.color[0], s.color[1], s.color[2], 0.0]),
            None => (FALLBACK_LIGHT.sun_dir, FALLBACK_LIGHT.sun_color),
        };
        // Once here rather than per fragment: the snapshot's own direction is a
        // unit vector already, but FALLBACK_LIGHT's is 0.816 long, so the
        // shaders had to normalize a value that never changes within a pass.
        let sun_len = (sun_dir[0] * sun_dir[0] + sun_dir[1] * sun_dir[1] + sun_dir[2] * sun_dir[2]).sqrt();
        let sun_dir = if sun_len > 1e-6 {
            [sun_dir[0] / sun_len, sun_dir[1] / sun_len, sun_dir[2] / sun_len]
        } else {
            sun_dir
        };
        let sun_color = [
            sun_color_base[0] * sun_fraction,
            sun_color_base[1] * sun_fraction,
            sun_color_base[2] * sun_fraction,
            sun_color_base[3],
        ];
        LightUniform { position, range, color, sun_dir, _sun_pad: 0.0, sun_color }
    }
}

const FALLBACK_LIGHT: LightUniform = LightUniform {
    position: [400.0, 800.0, 400.0],
    range: 4000.0,
    color: [1.0, 1.0, 1.0, 0.0],
    sun_dir: [-0.33, -0.67, -0.33],
    _sun_pad: 0.0,
    sun_color: [1.0, 1.0, 1.0, 0.0],
};

/// Where `assets/` lives.
///
/// `BEVY_ASSET_ROOT` first, because the rest of the game already honours it
/// and this loader did not: outside a bundle NSBundle hands back the
/// executable's own directory, so a release build found assets only through a
/// symlink someone had left in target/release and a debug build found nothing
/// at all. Zombies still spawned -- they were being shot at -- but every
/// skinned model failed to load and they had no mesh.
/// Where `assets/` sits, as a directory to join onto.
///
/// On iOS that is the app bundle, and NSBundle is the only thing that knows
/// where that was put. Everywhere else it is whatever Bevy decided, which is
/// the point: the renderer loads a handful of models itself rather than
/// through the AssetServer, and the two have to agree about where the assets
/// are or one of them finds nothing.
///
/// They did not agree. A plain executable has no bundle, so resourcePath
/// handed back the directory the binary sits in -- target/debug -- and every
/// zombie failed to load on the desktop build with a path nobody recognised.
/// Asking Bevy is not a workaround for that; it is the fix, because Bevy's
/// answer is the one the rest of the game is already using.
pub fn bundle_resource_path() -> Option<String> {
    #[cfg(target_os = "ios")]
    {
        // BEVY_ASSET_ROOT first, the same order Bevy itself checks in.
        if let Ok(root) = std::env::var("BEVY_ASSET_ROOT") {
            if !root.is_empty() {
                return Some(root);
            }
        }
        unsafe {
            let bundle_class = class!(NSBundle);
            let bundle: *mut Object = msg_send![bundle_class, mainBundle];
            let path: *mut Object = msg_send![bundle, resourcePath];
            if path.is_null() {
                return None;
            }
            let utf8: *const std::os::raw::c_char = msg_send![path, UTF8String];
            if utf8.is_null() {
                return None;
            }
            Some(std::ffi::CStr::from_ptr(utf8).to_string_lossy().into_owned())
        }
    }
    #[cfg(not(target_os = "ios"))]
    {
        // BEVY_ASSET_ROOT, then CARGO_MANIFEST_DIR, then the executable's
        // own directory -- and it is Bevy's function doing the deciding, so
        // it cannot drift from what the AssetServer resolved.
        Some(
            bevy::asset::io::file::FileAssetReader::get_base_path()
                .to_string_lossy()
                .into_owned(),
        )
    }
}

pub const SPONZA_SHADER_SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

// Each of these costs real per-pixel work, so they are compile-time switches
// rather than uniforms: a quality setting that only shrinks a texture or
// scales a factor still pays for the fetches and the ALU. ShaderFeatures
// prepends the overrides.
#ifndef FEATURE_SHADOWS
#define FEATURE_SHADOWS 1
#endif
#ifndef FEATURE_NORMAL_MAP
#define FEATURE_NORMAL_MAP 1
#endif
#ifndef FEATURE_SPECULAR
#define FEATURE_SPECULAR 1
#endif
#ifndef FEATURE_POINT_LIGHT
#define FEATURE_POINT_LIGHT 1
#endif
// Measurement floor: returns a constant without reading a texture or lighting
// anything, so whatever fragment time remains is the cost of shading that many
// pixels rather than the cost of this shader.
#ifndef FEATURE_FLAT
#define FEATURE_FLAT 0
#endif

struct Vertex {
    packed_float3 position;
    // snorm16/unorm16, packed_ so the offsets match Rust's repr(C) instead of
    // short4's own 8-byte alignment. See GltfVertex (gltf_scene.rs).
    packed_short4 normal;
    packed_float2 uv;
    // xyz = tangent direction, w = handedness — see GltfVertex's own doc
    // comment (gltf_scene.rs) for where this gets baked (CPU-side, world
    // space, same as position/normal above).
    packed_short4 tangent;
    packed_ushort2 uv2;
};

// Split so the per-pass half is bound once instead of copied into every
// draw: of the 432 bytes this used to be, only the 160 in MeshUniforms vary
// per mesh.
struct PassUniforms {
    float4x4 viewProjection;
    packed_float3 lightPos;
    float lightRange;
    float4 lightColor;
    packed_float3 sunDir;
    float _sunPad;
    float4 sunColor;
    // Shadow-mapping/specular pass (see SHADOW_SHADER_SOURCE's own doc
    // comment) — lightViewProjection/pointLightViewProjection re-project
    // worldPos into the sun's and the point light's own shadow maps to
    // sample occlusion; cameraPos is needed to reconstruct a view
    // direction per-fragment for the Blinn-Phong specular term, since this
    // shader has no other way to know where the eye is.
    float4x4 lightViewProjection;
    float4x4 pointLightViewProjection;
    packed_float3 cameraPos;
    float _cameraPad;
};

struct MeshUniforms {
    float4 baseColorFactor;
    float4 emissive;
    float4x4 modelMatrix;
    float4x4 normalMatrix;
};

struct VertexOut {
    float4 position [[position]];
    float3 worldPos;
    float3 normal;
    float2 uv;
    // Only the normal-mapping variant reads this, and everything declared
    // here is written into the tile parameter buffer for every vertex in the
    // frame -- 16 of 72 bytes, on a map whose materials carry no normal map.
#if FEATURE_NORMAL_MAP
    float4 tangent;
#endif
    float2 uv2;
};

vertex VertexOut sponza_vertex(uint vid [[vertex_id]],
                                const device Vertex *verts [[buffer(0)]],
                                constant PassUniforms &pass [[buffer(1)]],
                                constant MeshUniforms &mesh [[buffer(2)]]) {
    VertexOut out;
    Vertex v = verts[vid];
    float4 worldPos4 = mesh.modelMatrix * float4(float3(v.position), 1.0);
    out.position = pass.viewProjection * worldPos4;
    out.worldPos = worldPos4.xyz;
    out.normal = (mesh.normalMatrix * float4(float3(short3(v.normal.xyz)) * (1.0 / 32767.0), 0.0)).xyz;
    out.uv = float2(v.uv);
#if FEATURE_NORMAL_MAP
    // xyz rotates like the normal above (w=0 drops translation); w itself
    // is glTF's handedness sign, not a spatial component, so it carries
    // straight through unchanged — same convention bake_static_mesh_
    // vertices/bake_local_mesh_vertices already use for tangent.w.
    float3 tangentDir = float3(short3(v.tangent.xyz)) * (1.0 / 32767.0);
    float3 rotatedTangent = (mesh.modelMatrix * float4(tangentDir, 0.0)).xyz;
    out.tangent = float4(rotatedTangent, float(v.tangent.w) * (1.0 / 32767.0));
#endif
    out.uv2 = float2(ushort2(v.uv2)) * (1.0 / 65535.0);
    return out;
}

// Shininess exponent for the Blinn-Phong specular term — one fixed value
// for every material (this renderer has no per-material roughness/
// metallic map to drive it from, unlike bevy_render's real PBR path).
// Was 48 initially, which looked fine on the zombie's more irregular
// low-poly normals but washed the gun's large flat receiver faces into a
// smooth gray gradient wide enough to bury its own brushed-metal albedo
// texture underneath — those faces interpolate one near-constant normal
// across a much bigger area than the zombie's, so the same "reasonably
// tight" exponent produced a much wider hotspot there. 90 narrows the
// falloff enough to stay a small glint even on a big flat face.
constant float kShininess = 90.0;
// Specular intensity multiplier — see kShininess's own comment for why
// this also came down (0.6 -> 0.35): a wide AND bright highlight is what
// was burying the gun's texture detail, not just the width alone.
constant float kSpecularStrength = 0.35;

// Reads `shadowMap` (already-normalized-to-texture-space `lightViewProjection`)
// at `worldPos`, returning 1.0 (fully lit) .. 0.0 (fully shadowed) via
// Metal's hardware `sample_compare` (free bilinear = 2x2 PCF). Shared by
// both the sun and point-light shadow lookups below — same convention,
// different light_view_projection/texture per call.
static inline float sample_shadow(float4x4 light_view_projection, float3 worldPos, depth2d<float> shadowMap, sampler shadowSampler) {
    float4 lightClip = light_view_projection * float4(worldPos, 1.0);
    float3 lightNdc = lightClip.xyz / lightClip.w;
    float2 shadowUv = lightNdc.xy * 0.5 + 0.5;
    shadowUv.y = 1.0 - shadowUv.y;
    // A small constant bias (subtracted so a fragment reads slightly
    // nearer-to-light than it actually is) avoids shadow acne — self-
    // shadowing false positives from the shadow map's own limited
    // resolution. Doubled vs. a typical [0,1]-range bias since this
    // renderer's depth range spans [-1,1] (twice as wide).
    float shadowBias = 0.004;
    if (shadowUv.x < 0.0 || shadowUv.x > 1.0 || shadowUv.y < 0.0 || shadowUv.y > 1.0 || lightNdc.z > 1.0) {
        return 1.0;
    }
    return shadowMap.sample_compare(shadowSampler, shadowUv, lightNdc.z - shadowBias);
}

fragment float4 sponza_fragment(VertexOut in [[stage_in]],
                                 constant PassUniforms &pass [[buffer(1)]],
                                 constant MeshUniforms &mesh [[buffer(2)]],
                                 texture2d<float> baseColorTexture [[texture(0)]],
                                 sampler texSampler [[sampler(0)]],
                                 depth2d<float> shadowMap [[texture(1)]],
                                 sampler shadowSampler [[sampler(1)]],
                                 depth2d<float> pointShadowMap [[texture(2)]],
                                 texture2d<float> normalTexture [[texture(3)]],
                                 texture2d<float> lightmap [[texture(4)]]) {
#if FEATURE_FLAT
    return float4(0.6, 0.6, 0.6, 1.0);
#else
    // Geometric (per-vertex-interpolated) normal, reoriented by the
    // sampled normal map below when this mesh actually has one — see
    // ExtractedMesh::normal_texture's own doc comment (native_game_
    // extract.rs) for why the AK47's receiver looked flat/textureless
    // without this: its "brushed metal" scratches are a lighting response
    // to a normal map, not a base-color pattern, and this renderer never
    // read normal maps at all before now. normalTexture is always bound
    // to SOMETHING (see flat_normal_texture's own doc comment) — a flat
    // "pointing straight up in tangent space" 1x1 fallback for meshes
    // with no real one, so this math runs unconditionally instead of
    // branching on an extra "has a normal map" uniform flag.
    float3 geometricNormal = normalize(in.normal);
#if FEATURE_NORMAL_MAP
    float3 tangent = normalize(in.tangent.xyz - geometricNormal * dot(in.tangent.xyz, geometricNormal));
    float3 bitangent = cross(geometricNormal, tangent) * in.tangent.w;
    float3x3 TBN = float3x3(tangent, bitangent, geometricNormal);
    float3 sampledNormal = normalTexture.sample(texSampler, in.uv).xyz * 2.0 - 1.0;
    float3 normal = normalize(TBN * sampledNormal);
#else
    float3 normal = geometricNormal;
#endif

    float3 viewDir = normalize(float3(pass.cameraPos) - in.worldPos);
    // Normalized host-side in LightUniform::from_snapshots.
    float3 toSun = -float3(pass.sunDir);

    // See SHADOW_SHADER_SOURCE's own doc comment for how these were
    // written — same lightViewProjection convention used to read them back
    // keeps the comparison self-consistent regardless of Metal's "native"
    // depth range (this renderer never uses it, see metal_math::
    // perspective's own doc comment). Both the sun (parallel rays, hence
    // an orthographic lightViewProjection) and the arena's overhead point
    // light (a real position, hence a perspective one — see render_frame's
    // "Pass 0b") cast real shadows here; bevy_render's own PBR path does
    // too for both, which is what an earlier version of this shader was
    // missing (only the sun was covered) — visible as a zombie/pillar
    // having two *different* real shadows on desktop that this renderer
    // only drew one of.
#if FEATURE_SHADOWS
    float sunShadow = sample_shadow(pass.lightViewProjection, in.worldPos, shadowMap, shadowSampler);
#else
    float sunShadow = 1.0;
#endif

    float sunDiffuse = max(dot(normal, toSun), 0.0) * sunShadow;
#if FEATURE_SPECULAR
    float sunSpecular = sunDiffuse > 0.0 ? pow(max(dot(normal, normalize(toSun + viewDir)), 0.0), kShininess) * sunShadow : 0.0;
#else
    float sunSpecular = 0.0;
#endif

#if FEATURE_POINT_LIGHT
    float3 toLight = float3(pass.lightPos) - in.worldPos;
    float dist = length(toLight);
    float atten = clamp(1.0 - dist / max(pass.lightRange, 0.0001), 0.0, 1.0);
    atten *= atten;
    float pointDiffuse = 0.0;
    float pointSpecular = 0.0;
    // Both terms below are multiplied by atten, so out of the light's reach
    // every one of them is zero and the shadow lookup that feeds them is a
    // fetch whose result is thrown away. There is no point light at all for
    // most of a frame -- it is the muzzle flash.
    if (atten > 0.0) {
#if FEATURE_SHADOWS
        float pointShadowFactor = sample_shadow(pass.pointLightViewProjection, in.worldPos, pointShadowMap, shadowSampler);
#else
        float pointShadowFactor = 1.0;
#endif
        float3 pointDir = toLight / max(dist, 0.0001);
        pointDiffuse = max(dot(normal, pointDir), 0.0) * atten * pointShadowFactor;
#if FEATURE_SPECULAR
        pointSpecular = pointDiffuse > 0.0 ? pow(max(dot(normal, normalize(pointDir + viewDir)), 0.0), kShininess) * atten * pointShadowFactor : 0.0;
#endif
    }
#else
    float pointDiffuse = 0.0;
    float pointSpecular = 0.0;
#endif

    // Diffuse modulates the surface's own base color (as before); specular
    // is intentionally kept SEPARATE and added on top unmultiplied by
    // baseColor — a real (even non-PBR) specular highlight is the light's
    // own color reflecting off the surface, not tinted by its albedo, and
    // folding it into the same term that gets multiplied by baseColor.rgb
    // below crushed it to near-invisible on dark surfaces (the gun's black
    // metal) even though the math upstream was already computing it
    // correctly — this was the actual bug, not a missing specular term.
    float3 diffuseTerm = clamp(pass.sunColor.rgb * sunDiffuse + pass.lightColor.rgb * pointDiffuse, 0.0, 1.0);
    float3 specularTerm = (pass.sunColor.rgb * sunSpecular + pass.lightColor.rgb * pointSpecular) * kSpecularStrength;

    float4 texColor = baseColorTexture.sample(texSampler, in.uv);
    float4 baseColor = texColor * mesh.baseColorFactor;

    // The 0.35 this replaces was the whole of indirect light: a corner and an
    // open wall were handed the same fill. R is baked ambient occlusion, G is
    // whether the sun reaches here at all.
    //
    // uv2 is exactly zero for anything the bake did not cover -- unmapped
    // geometry, and every mesh that is not the map. Those keep the old
    // constant rather than reading texel zero, which is a corner of the atlas
    // and means nothing.
    float ambient = 0.35;
    float sunMask = 1.0;
    if (in.uv2.x > 0.0 || in.uv2.y > 0.0) {
        float2 baked = lightmap.sample(texSampler, in.uv2).rg;
        // Not all the way to black. Occlusion this strong is geometry a metre
        // away, not a sealed box, and a floor that reads zero is a hole.
        ambient = mix(0.06, 0.42, baked.r);
        sunMask = baked.g;
    }
    float3 lit = baseColor.rgb * (ambient + 0.65 * diffuseTerm * sunMask) + specularTerm * sunMask;
    // Additive — no real Bloom pass exists in this renderer (see
    // ExtractedMesh::emissive's own doc comment for why these values are
    // authored well above 1.0), but adding the raw emissive term straight
    // onto the lit result at least makes muzzle flash/reticle/sparks read
    // as bright/glowing instead of flat, matching bevy_render's own
    // emissive-is-additive-to-lit-color behavior minus the actual bloom
    // blur/threshold pass.
    lit += mesh.emissive.rgb;
    return float4(lit, baseColor.a);
#endif
}
"#;

/// Mirrors PassUniforms in SPONZA_SHADER_SOURCE. Bound once per pass.
#[repr(C)]
#[derive(Copy, Clone)]
pub struct SponzaPassUniforms {
    pub view_projection: [[f32; 4]; 4],
    pub light_pos: [f32; 3],
    pub light_range: f32,
    pub light_color: [f32; 4],
    pub sun_dir: [f32; 3],
    pub _sun_pad: f32,
    pub sun_color: [f32; 4],
    pub light_view_projection: [[f32; 4]; 4],
    pub point_light_view_projection: [[f32; 4]; 4],
    pub camera_pos: [f32; 3],
    pub _camera_pad: f32,
}

/// Mirrors MeshUniforms in SPONZA_SHADER_SOURCE. The only half that changes
/// between draws.
#[repr(C)]
#[derive(Copy, Clone)]
pub struct SponzaMeshUniforms {
    pub base_color_factor: [f32; 4],
    pub emissive: [f32; 4],
    pub model_matrix: [[f32; 4]; 4],
    pub normal_matrix: [[f32; 4]; 4],
}

const SKINNED_SHADER_SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct Vertex {
    packed_float3 position;
    packed_float3 normal;
    packed_float2 uv;
    ushort4 joints;
    packed_float4 weights;
};

// Split for the same reason as SPONZA_SHADER_SOURCE's: only modelMatrix and
// baseColorFactor change between draws.
struct PassUniforms {
    float4x4 viewProjection;
    packed_float3 lightPos;
    float lightRange;
    float4 lightColor;
    packed_float3 sunDir;
    float _sunPad;
    float4 sunColor;
    // Shadow-mapping/specular pass — see SPONZA_SHADER_SOURCE's own
    // Uniforms doc comment, identical reasoning applies here.
    float4x4 lightViewProjection;
    float4x4 pointLightViewProjection;
    packed_float3 cameraPos;
    float _cameraPad;
    // xyz the probe grid's world origin, w its spacing; then its size in
    // probes. float4 rather than a float3 and a scalar on purpose -- a lone
    // float3 would shift every offset after it against Rust's repr(C).
    float4 probeOriginSpacing;
    float4 probeDims;
};

struct MeshUniforms {
    float4x4 modelMatrix;
    float4 baseColorFactor;
};

struct VertexOut {
    float4 position [[position]];
    float3 worldPos;
    float3 normal;
    float2 uv;
};

vertex VertexOut skinned_vertex(uint vid [[vertex_id]],
                                 const device Vertex *verts [[buffer(0)]],
                                 constant PassUniforms &pass [[buffer(1)]],
                                 const device float4x4 *jointMatrices [[buffer(2)]],
                                 constant MeshUniforms &mesh [[buffer(3)]]) {
    Vertex v = verts[vid];
    float4x4 skin =
        v.weights.x * jointMatrices[v.joints.x] +
        v.weights.y * jointMatrices[v.joints.y] +
        v.weights.z * jointMatrices[v.joints.z] +
        v.weights.w * jointMatrices[v.joints.w];

    float4 skinnedPos = skin * float4(float3(v.position), 1.0);
    float3 skinnedNormal = normalize((skin * float4(float3(v.normal), 0.0)).xyz);

    VertexOut out;
    float4 worldPos4 = mesh.modelMatrix * skinnedPos;
    out.position = pass.viewProjection * worldPos4;
    out.worldPos = worldPos4.xyz;
    out.normal = (mesh.modelMatrix * float4(skinnedNormal, 0.0)).xyz;
    out.uv = float2(v.uv);
    return out;
}

// Same fixed shininess/specular-strength constants as SPONZA_SHADER_SOURCE
// — duplicated, not shared, since each shader source string in this file
// compiles as its own independent Metal library.
constant float kShininess = 48.0;
constant float kSpecularStrength = 0.6;

// Same shadow-sampling helper as SPONZA_SHADER_SOURCE — duplicated, not
// shared, since each shader source string in this file compiles as its
// own independent Metal library (see kShininess/kSpecularStrength's own
// "duplicated, not shared" comment above).
static inline float sample_shadow(float4x4 light_view_projection, float3 worldPos, depth2d<float> shadowMap, sampler shadowSampler) {
    float4 lightClip = light_view_projection * float4(worldPos, 1.0);
    float3 lightNdc = lightClip.xyz / lightClip.w;
    float2 shadowUv = lightNdc.xy * 0.5 + 0.5;
    shadowUv.y = 1.0 - shadowUv.y;
    float shadowBias = 0.004;
    if (shadowUv.x < 0.0 || shadowUv.x > 1.0 || shadowUv.y < 0.0 || shadowUv.y > 1.0 || lightNdc.z > 1.0) {
        return 1.0;
    }
    return shadowMap.sample_compare(shadowSampler, shadowUv, lightNdc.z - shadowBias);
}

fragment float4 skinned_fragment(VertexOut in [[stage_in]],
                                  constant PassUniforms &pass [[buffer(1)]],
                                 constant MeshUniforms &mesh [[buffer(3)]],
                                  texture2d<float> baseColorTexture [[texture(0)]],
                                  sampler texSampler [[sampler(0)]],
                                  depth2d<float> shadowMap [[texture(1)]],
                                  sampler shadowSampler [[sampler(1)]],
                                  depth2d<float> pointShadowMap [[texture(2)]],
                                  texture3d<float> probesPositive [[texture(3)]],
                                  texture3d<float> probesNegative [[texture(4)]],
                                  sampler probeSampler [[sampler(2)]]) {
    // Root cause of an earlier version of this struct (kept for the next
    // person who hits this class of bug) found via a 3-step device/
    // Simulator console+shader bisection: (1) raw texColor sample alone →
    // colorful — rules out texture binding/decode; (2) texColor *
    // mesh.baseColorFactor → solid black — isolates the bug to reading
    // baseColorFactor itself; (3) a lone `float3 lightDir` field (MSL
    // default alignment 16 per-member — NOT the GLSL std140 "vec3 forces
    // the next struct member to a 16-byte boundary" rule some docs
    // conflate this with, but the net effect here was the same) placed
    // whatever followed it at a different byte offset than Rust's
    // tightly-packed repr(C) struct, since a lone `float3` still consumes
    // a rounded-up 16-byte slot when a scalar or vec4 follows. Every
    // `packed_float3`/plain-float4 field below is chosen specifically so
    // each field's offset lands on the same byte in both Rust's repr(C)
    // layout and MSL's (see LightUniform's own doc comment) instead of
    // re-deriving this by trial and error again.
    float3 normal = normalize(in.normal);
    float3 viewDir = normalize(float3(pass.cameraPos) - in.worldPos);
    // Normalized host-side in LightUniform::from_snapshots.
    float3 toSun = -float3(pass.sunDir);

    // See SHADOW_SHADER_SOURCE's own doc comment for how these were
    // written, and SPONZA_SHADER_SOURCE's matching block for why both the
    // sun AND the point light get their own real shadow here (not
    // repeated per-shader).
    float sunShadow = sample_shadow(pass.lightViewProjection, in.worldPos, shadowMap, shadowSampler);
    float pointShadowFactor = sample_shadow(pass.pointLightViewProjection, in.worldPos, pointShadowMap, shadowSampler);

    float sunDiffuse = max(dot(normal, toSun), 0.0) * sunShadow;
    float sunSpecular = sunDiffuse > 0.0 ? pow(max(dot(normal, normalize(toSun + viewDir)), 0.0), kShininess) * sunShadow : 0.0;

    float3 toLight = float3(pass.lightPos) - in.worldPos;
    float dist = length(toLight);
    float atten = clamp(1.0 - dist / max(pass.lightRange, 0.0001), 0.0, 1.0);
    atten *= atten;
    float3 pointDir = toLight / max(dist, 0.0001);
    float pointDiffuse = max(dot(normal, pointDir), 0.0) * atten * pointShadowFactor;
    float pointSpecular = pointDiffuse > 0.0 ? pow(max(dot(normal, normalize(pointDir + viewDir)), 0.0), kShininess) * atten * pointShadowFactor : 0.0;

    // See SPONZA_SHADER_SOURCE's matching block for why specular is kept
    // separate from baseColor's multiply instead of folded into the same
    // term (that crushed it to near-invisible on dark surfaces).
    float3 diffuseTerm = clamp(pass.sunColor.rgb * sunDiffuse + pass.lightColor.rgb * pointDiffuse, 0.0, 1.0);
    float3 specularTerm = (pass.sunColor.rgb * sunSpecular + pass.lightColor.rgb * pointSpecular) * kSpecularStrength;

    float4 texColor = baseColorTexture.sample(texSampler, in.uv);
    float4 baseColor = texColor * mesh.baseColorFactor;

    // The light this thing is standing in, read out of the probe grid.
    //
    // Nothing that moves can hold a lightmap, so a zombie inside a building
    // used to be lit exactly like one in the street. These six numbers are
    // how open the sky is along each axis at this point in the air; the
    // normal picks three of them and weights by how much it faces each.
    float ambient = 0.35;
    float sunMask = 1.0;
    if (pass.probeOriginSpacing.w > 0.0) {
        float3 cell = (in.worldPos - pass.probeOriginSpacing.xyz)
                    / pass.probeOriginSpacing.w;
        // Texel centres, not corners: probe i sits at origin + i*spacing, and
        // sampling at i/dims would read half a cell short of it.
        float3 uvw = (cell + 0.5) / pass.probeDims.xyz;
        if (all(uvw > 0.0) && all(uvw < 1.0)) {
            float4 pos = probesPositive.sample(probeSampler, uvw);
            float4 neg = probesNegative.sample(probeSampler, uvw);
            float3 n = normalize(in.normal);
            float3 weight = n * n;
            float3 openness = float3(
                n.x >= 0.0 ? pos.r : neg.r,
                n.y >= 0.0 ? pos.g : neg.g,
                n.z >= 0.0 ? pos.b : neg.b);
            ambient = mix(0.06, 0.42, dot(weight, openness));
            sunMask = pos.a;
        }
    }
    float3 lit = baseColor.rgb * (ambient + 0.65 * diffuseTerm * sunMask) + specularTerm * sunMask;
    return float4(lit, baseColor.a);
}
"#;

const SHADOW_SHADER_SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct SponzaVertex {
    packed_float3 position;
    packed_short4 normal;
    packed_float2 uv;
    // Unused here (this pass only needs position) but MUST stay present —
    // this struct's buffer(0) is the exact same GltfVertex-laid-out buffer
    // builds one buffer per mesh, shared by both the shadow and main
    // passes — see draw_shadow_casters/draw_scene). Metal computes this
    // struct's own stride from ITS OWN size to index verts[vid], so
    // dropping this field would silently read every vertex 16 bytes short
    // — position/normal/uv would land right, but every vertex AFTER the
    // first would already be reading from the wrong offset.
    packed_short4 tangent;
    // Mirrors GltfVertex, which grew a lightmap uv. Unused in this pass but
    // the stride has to match or every vertex after the first is read short.
    packed_ushort2 uv2;
};

struct SponzaShadowUniforms {
    float4x4 lightViewProjection;
    // Same role as SPONZA_SHADER_SOURCE's own Uniforms::modelMatrix —
    // identity for CPU-baked-to-world-space meshes (walls/gun), the real
    // per-instance transform for cars (cached LOCAL-space vertex buffer).
    // SkinnedShadowUniforms above already has this exact field for the
    // same reason (zombies' shadow pass).
    float4x4 modelMatrix;
};

vertex float4 shadow_vertex_sponza(uint vid [[vertex_id]],
                                    const device SponzaVertex *verts [[buffer(0)]],
                                    constant SponzaShadowUniforms &uniforms [[buffer(1)]]) {
    SponzaVertex v = verts[vid];
    return uniforms.lightViewProjection * uniforms.modelMatrix * float4(float3(v.position), 1.0);
}

struct SkinnedVertex {
    packed_float3 position;
    packed_float3 normal;
    packed_float2 uv;
    ushort4 joints;
    packed_float4 weights;
};

struct SkinnedShadowUniforms {
    float4x4 lightViewProjection;
    float4x4 modelMatrix;
};

vertex float4 shadow_vertex_skinned(uint vid [[vertex_id]],
                                     const device SkinnedVertex *verts [[buffer(0)]],
                                     constant SkinnedShadowUniforms &uniforms [[buffer(1)]],
                                     const device float4x4 *jointMatrices [[buffer(2)]]) {
    SkinnedVertex v = verts[vid];
    float4x4 skin =
        v.weights.x * jointMatrices[v.joints.x] +
        v.weights.y * jointMatrices[v.joints.y] +
        v.weights.z * jointMatrices[v.joints.z] +
        v.weights.w * jointMatrices[v.joints.w];
    float4 skinnedPos = skin * float4(float3(v.position), 1.0);
    float4 worldPos = uniforms.modelMatrix * skinnedPos;
    return uniforms.lightViewProjection * worldPos;
}
"#;

#[repr(C)]
#[derive(Copy, Clone)]
struct SponzaShadowUniforms {
    pub light_view_projection: [[f32; 4]; 4],
    pub model_matrix: [[f32; 4]; 4],
}

#[repr(C)]
#[derive(Copy, Clone)]
struct SkinnedShadowUniforms {
    pub light_view_projection: [[f32; 4]; 4],
    pub model_matrix: [[f32; 4]; 4],
}

const SHADOW_MAP_SIZE: u64 = 2048;

const SHADOW_MAP_SIZES: [u64; 6] = [512, 512, 1024, 1536, 2048, 3072];

const RENDER_SCALE_MIN_PX: u64 = 64;

const SHADOW_HALF_EXTENT: f32 = 22.0;
const SHADOW_DISTANCE: f32 = 40.0;


const MSAA_SAMPLE_COUNT: u64 = 4;

const LINE_SHADER_SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct LineVertex {
    packed_float3 position;
    packed_float4 color;
};

struct LineUniforms {
    float4x4 viewProjection;
};

struct LineVertexOut {
    float4 position [[position]];
    float4 color;
};

vertex LineVertexOut line_vertex(uint vid [[vertex_id]],
                                 const device LineVertex *verts [[buffer(0)]],
                                 constant LineUniforms &uniforms [[buffer(1)]]) {
    LineVertexOut out;
    out.position = uniforms.viewProjection * float4(float3(verts[vid].position), 1.0);
    out.color = float4(verts[vid].color);
    return out;
}

fragment float4 line_fragment(LineVertexOut in [[stage_in]]) {
    return in.color;
}
"#;

#[repr(C)]
#[derive(Clone, Copy)]
struct LineVertex {
    position: [f32; 3],
    color: [f32; 4],
}

#[repr(C)]
struct LineUniforms {
    pub view_projection: [[f32; 4]; 4],
}



const UI_SHADER_SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct UiVertex {
    packed_float2 position;
    packed_float2 uv;
};

struct UiVertexOut {
    float4 position [[position]];
    float2 uv;
};

struct UiUniforms {
    float4 tint;
};

vertex UiVertexOut ui_vertex(uint vid [[vertex_id]],
                              const device UiVertex *verts [[buffer(0)]]) {
    UiVertexOut out;
    UiVertex v = verts[vid];
    out.position = float4(v.position, 0.0, 1.0);
    out.uv = v.uv;
    return out;
}

fragment float4 ui_fragment(UiVertexOut in [[stage_in]],
                             constant UiUniforms &uniforms [[buffer(1)]],
                             texture2d<float> tex [[texture(0)]],
                             sampler texSampler [[sampler(0)]]) {
    float4 texColor = tex.sample(texSampler, in.uv);
    return texColor * uniforms.tint;
}
"#;

#[repr(C)]
#[derive(Copy, Clone)]
struct UiVertex {
    position: [f32; 2],
    uv: [f32; 2],
}

#[repr(C)]
#[derive(Copy, Clone)]
struct UiUniforms {
    tint: [f32; 4],
}

const BLIT_SHADER_SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct BlitVertex {
    packed_float2 position;
    packed_float2 uv;
};

struct BlitVertexOut {
    float4 position [[position]];
    float2 uv;
};

vertex BlitVertexOut blit_vertex(uint vid [[vertex_id]],
                                  const device BlitVertex *verts [[buffer(0)]]) {
    BlitVertexOut out;
    BlitVertex v = verts[vid];
    out.position = float4(v.position, 0.0, 1.0);
    out.uv = v.uv;
    return out;
}

fragment float4 blit_fragment(BlitVertexOut in [[stage_in]],
                               texture2d<float> tex [[texture(0)]],
                               sampler texSampler [[sampler(0)]]) {
    return tex.sample(texSampler, in.uv);
}
"#;

#[repr(C)]
#[derive(Copy, Clone)]
struct BlitVertex {
    position: [f32; 2],
    uv: [f32; 2],
}

const GLYPH_W: usize = 5;
const GLYPH_H: usize = 7;
const GLYPH_COUNT: usize = 45;

const GLYPH_BITMAPS: [[u8; GLYPH_H]; GLYPH_COUNT] = [
    [0b01110, 0b10001, 0b10011, 0b10101, 0b11001, 0b10001, 0b01110],
    [0b00100, 0b01100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110],
    [0b01110, 0b10001, 0b00001, 0b00010, 0b00100, 0b01000, 0b11111],
    [0b11111, 0b00010, 0b00100, 0b00010, 0b00001, 0b10001, 0b01110],
    [0b00010, 0b00110, 0b01010, 0b10010, 0b11111, 0b00010, 0b00010],
    [0b11111, 0b10000, 0b11110, 0b00001, 0b00001, 0b10001, 0b01110],
    [0b00110, 0b01000, 0b10000, 0b11110, 0b10001, 0b10001, 0b01110],
    [0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b01000, 0b01000],
    [0b01110, 0b10001, 0b10001, 0b01110, 0b10001, 0b10001, 0b01110],
    [0b01110, 0b10001, 0b10001, 0b01111, 0b00001, 0b00010, 0b01100],
    [0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b01100, 0b01100],
    [0b00001, 0b00010, 0b00100, 0b01000, 0b10000, 0b00000, 0b00000],
    [0b01110, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001, 0b10001],
    [0b01111, 0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b01111],
    [0b11110, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b11110],
    [0b11111, 0b10000, 0b10000, 0b11110, 0b10000, 0b10000, 0b11111],
    [0b10001, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001, 0b10001],
    [0b01110, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110],
    [0b00111, 0b00010, 0b00010, 0b00010, 0b00010, 0b10010, 0b01100],
    [0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b11111],
    [0b10001, 0b11011, 0b10101, 0b10101, 0b10001, 0b10001, 0b10001],
    [0b10001, 0b11001, 0b10101, 0b10101, 0b10011, 0b10001, 0b10001],
    [0b01110, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01110],
    [0b11110, 0b10001, 0b10001, 0b11110, 0b10000, 0b10000, 0b10000],
    [0b11110, 0b10001, 0b10001, 0b11110, 0b10100, 0b10010, 0b10001],
    [0b01111, 0b10000, 0b10000, 0b01110, 0b00001, 0b00001, 0b11110],
    [0b11111, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100],
    [0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01110],
    [0b00001, 0b00010, 0b00100, 0b01000, 0b00100, 0b00010, 0b00001],
    [0b10000, 0b01000, 0b00100, 0b00010, 0b00100, 0b01000, 0b10000],
    [0b00000, 0b00000, 0b11111, 0b00000, 0b11111, 0b00000, 0b00000],
    [0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b00000],
    [0b01110, 0b10001, 0b10000, 0b10111, 0b10001, 0b10001, 0b01111],
    [0b00000, 0b00000, 0b01100, 0b01100, 0b00000, 0b01100, 0b01100],
    [0b11110, 0b10001, 0b10001, 0b11110, 0b10001, 0b10001, 0b11110],
    [0b11111, 0b10000, 0b10000, 0b11110, 0b10000, 0b10000, 0b10000],
    [0b10001, 0b10010, 0b10100, 0b11000, 0b10100, 0b10010, 0b10001],
    [0b01110, 0b10001, 0b10001, 0b10001, 0b10101, 0b10010, 0b01101],
    [0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01010, 0b00100],
    [0b10001, 0b10001, 0b10001, 0b10101, 0b10101, 0b10101, 0b01010],
    [0b10001, 0b10001, 0b01010, 0b00100, 0b01010, 0b10001, 0b10001],
    [0b10001, 0b10001, 0b01010, 0b00100, 0b00100, 0b00100, 0b00100],
    [0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b10000, 0b11111],
    // Hyphen and comma. Added for the credits screen: a licence named
    // CCBY4.0 is not the licence the asset was given under, and a credit
    // that cannot spell the licence is a weak credit.
    [0b00000, 0b00000, 0b00000, 0b01110, 0b00000, 0b00000, 0b00000],
    [0b00000, 0b00000, 0b00000, 0b00000, 0b00110, 0b00100, 0b01000],
];

fn glyph_index(c: char) -> Option<usize> {
    match c {
        '0'..='9' => Some(c as usize - '0' as usize),
        '.' => Some(10),
        '/' => Some(11),
        'A' => Some(12),
        'C' => Some(13),
        'D' => Some(14),
        'E' => Some(15),
        'H' => Some(16),
        'I' => Some(17),
        'J' => Some(18),
        'L' => Some(19),
        'M' => Some(20),
        'N' => Some(21),
        'O' => Some(22),
        'P' => Some(23),
        'R' => Some(24),
        'S' => Some(25),
        'T' => Some(26),
        'U' => Some(27),
        '<' => Some(28),
        '>' => Some(29),
        '=' => Some(30),
        ' ' => Some(31),
        'G' => Some(32),
        ':' => Some(33),
        'B' => Some(34),
        'F' => Some(35),
        'K' => Some(36),
        'Q' => Some(37),
        'V' => Some(38),
        'W' => Some(39),
        'X' => Some(40),
        'Y' => Some(41),
        'Z' => Some(42),
        '-' => Some(43),
        ',' => Some(44),
        _ => None,
    }
}

fn glyph_uv_rect(index: usize) -> (f32, f32) {
    let u0 = index as f32 / GLYPH_COUNT as f32;
    let u1 = (index + 1) as f32 / GLYPH_COUNT as f32;
    (u0, u1)
}

fn build_font_atlas(device: &Device) -> metal::Texture {
    let width = (GLYPH_W * GLYPH_COUNT) as u64;
    let height = GLYPH_H as u64;
    let mut pixels = vec![0u8; (width * height * 4) as usize];
    for (index, bitmap) in GLYPH_BITMAPS.iter().enumerate() {
        for (row, bits) in bitmap.iter().enumerate() {
            for col in 0..GLYPH_W {
                let lit = (bits >> (GLYPH_W - 1 - col)) & 1 != 0;
                let x = index * GLYPH_W + col;
                let y = row;
                let offset = (y * width as usize + x) * 4;
                let value = if lit { 255 } else { 0 };
                pixels[offset..offset + 4].copy_from_slice(&[value, value, value, value]);
            }
        }
    }
    let descriptor = metal::TextureDescriptor::new();
    descriptor.set_pixel_format(MTLPixelFormat::RGBA8Unorm);
    descriptor.set_width(width);
    descriptor.set_height(height);
    descriptor.set_storage_mode(metal::MTLStorageMode::Shared);
    let texture = device.new_texture(&descriptor);
    texture.replace_region(metal::MTLRegion::new_2d(0, 0, width, height), 0, pixels.as_ptr() as *const _, width * 4);
    texture
}

#[repr(C)]
#[derive(Copy, Clone)]
struct SkinnedPassUniforms {
    pub view_projection: [[f32; 4]; 4],
    pub light_pos: [f32; 3],
    pub light_range: f32,
    pub light_color: [f32; 4],
    pub sun_dir: [f32; 3],
    pub _sun_pad: f32,
    pub sun_color: [f32; 4],
    pub light_view_projection: [[f32; 4]; 4],
    pub point_light_view_projection: [[f32; 4]; 4],
    pub camera_pos: [f32; 3],
    pub _camera_pad: f32,
    /// xyz the probe grid's world origin, w its spacing. A float4 rather than
    /// a float3 and a scalar on purpose -- a lone float3 in the MSL mirror
    /// consumes a rounded-up sixteen bytes and shifts everything after it
    /// against Rust's repr(C).
    pub probe_origin_spacing: [f32; 4],
    /// xyz the grid's size in probes, w unused.
    pub probe_dims: [f32; 4],
}

/// Mirrors MeshUniforms in SKINNED_SHADER_SOURCE.
#[repr(C)]
#[derive(Copy, Clone)]
struct SkinnedMeshUniforms {
    pub model_matrix: [[f32; 4]; 4],
    pub base_color_factor: [f32; 4],
}

pub struct NativeMetalRenderer {
    device: Device,
    layer: MetalLayer,
    queue: CommandQueue,
    depth_state: metal::DepthStencilState,

    blend_depth_state: metal::DepthStencilState,

    ui_depth_state: metal::DepthStencilState,

    msaa_targets: super::metal_msaa::MsaaTargets,

    msaa_enabled: bool,

    shadow_quality: u8,

    render_resolution_scale: f32,


    scene_color_size: (u64, u64),
    scene_color_texture: metal::Texture,

    blit_pipeline: Option<RenderPipelineState>,
    blit_vertex_buffer: metal::Buffer,
    sponza_pipeline: Option<RenderPipelineState>,
    sponza_pipeline_no_msaa: Option<RenderPipelineState>,
    sponza_blend_pipeline: Option<RenderPipelineState>,
    sponza_blend_pipeline_no_msaa: Option<RenderPipelineState>,
    sponza_pipeline_no_shadows: Option<RenderPipelineState>,
    sponza_pipeline_no_shadows_no_msaa: Option<RenderPipelineState>,
    /// Same shader with FEATURE_NORMAL_MAP off, for meshes that carry no
    /// normal map. Chosen per draw; opaque_draw_order groups by it so the
    /// pass switches pipeline once.
    sponza_pipeline_flat_normal: Option<RenderPipelineState>,
    sponza_pipeline_flat_normal_no_msaa: Option<RenderPipelineState>,
    sponza_pipeline_no_shadows_flat_normal: Option<RenderPipelineState>,
    sponza_pipeline_no_shadows_flat_normal_no_msaa: Option<RenderPipelineState>,
    sponza_sampler: metal::SamplerState,

    shadow_map: metal::Texture,

    shadow_map_point: metal::Texture,
    shadow_pipeline_sponza: Option<RenderPipelineState>,
    shadow_pipeline_skinned: Option<RenderPipelineState>,

    shadow_depth_state: metal::DepthStencilState,

    shadow_sampler: metal::SamplerState,
    white_texture: metal::Texture,

    flat_normal_texture: metal::Texture,
    camera_position: [f32; 3],
    camera_yaw: f32,
    camera_pitch: f32,
    skinned_pipeline: Option<RenderPipelineState>,
    skinned_pipeline_no_msaa: Option<RenderPipelineState>,

    skinned_models: super::skinned_actor::SkinnedModelRegistry,


    debug_lines: Vec<super::snapshot::DebugLine>,
    line_pipeline: Option<RenderPipelineState>,
    line_pipeline_no_msaa: Option<RenderPipelineState>,
    start_time: std::time::Instant,




    real_game_textures: std::collections::HashMap<AssetId<Image>, metal::Texture>,
    /// Uploaded assets no draw claims, oldest first. See release_draw_assets.
    retained: std::collections::VecDeque<(RetainedAsset, u64)>,
    retained_bytes: u64,

    /// Buffers, counts, and the mesh's own bounding sphere. The sphere lives
    /// with the geometry because that is the only time the positions are in
    /// hand -- an already-uploaded mesh arrives with its position list empty.
    gpu_transformed_mesh_cache: std::collections::HashMap<
        AssetId<Mesh>,
        (metal::Buffer, metal::Buffer, usize, usize, [f32; 3], f32, bool),
    >,

    /// The draw list, indexed by the host's slot and kept between frames. Only
    /// the slots the host reports as changed are rebuilt, so a still scene
    /// prepares nothing. `None` is a slot whose entity is gone or hidden; the
    /// slot stays so every other index keeps its meaning.
    real_game_draws: Vec<Option<RealGameMeshDraw>>,

    /// How many live slots name each asset. A slot emptying frees nothing
    /// while another still names the same id -- sixty-four bullet holes share
    /// one mesh and one texture between them, so "an entity left" and "an
    /// asset stopped being used" are different questions. Counting answers
    /// the second one for the price of the slots that actually changed,
    /// instead of sweeping the whole list on the first.
    mesh_refs: std::collections::HashMap<AssetId<Mesh>, u32>,
    texture_refs: std::collections::HashMap<AssetId<Image>, u32>,

    /// Which slots hold a transparent draw. The blend pass used to find them
    /// by filtering the whole list, every frame and every pass, to come back
    /// with a handful -- the last walk that was searching for something it
    /// could have been told. Ordered, so the pass still reaches them in slot
    /// order before sorting them back to front.
    blend_slots: std::collections::BTreeSet<usize>,
    /// Opaque slots in draw order, grouped by texture pair. The pass itself
    /// never sorts, and a spawning casing only appends.
    opaque_order: Vec<usize>,
    /// Membership by slot, so an update already in the list costs one lookup
    /// instead of a scan.
    opaque_in_order: Vec<bool>,
    /// Appends and swap_removes since the last regroup, both of which leave
    /// the tail out of texture order.
    opaque_pending: usize,

    real_game_camera_rotation: [f32; 4],

    real_game_ads_blend: f32,
    real_game_wall_raise: f32,

    screen_capture: Option<crate::screen_capture::ScreenCapture>,
    joint_ring: JointRing,
    /// Kept so the cull count can be reported against the same planes the
    /// scene pass used. A Cell because draw_scene only has &self.
    last_main_view_projection: std::cell::Cell<[[f32; 4]; 4]>,
    /// The map's baked ambient occlusion and sun visibility. One texture for
    /// the whole map, supplied by the host -- nothing in the glTF carries it.
    lightmap: Option<metal::Texture>,
    /// The baked probe grid: six directions of sky openness plus sun
    /// visibility, split across two RGBA8 volumes because seven channels do
    /// not fit in four.
    probes: Option<(metal::Texture, metal::Texture)>,
    probe_origin_spacing: [f32; 4],
    probe_dims: [f32; 4],
    probe_sampler: Option<metal::SamplerState>,
    /// The main view's vertical fov, taken from the snapshot each frame.
    real_game_fov: f32,



    real_game_light: LightUniform,

    ui_pipeline: Option<RenderPipelineState>,
    ui_pipeline_no_msaa: Option<RenderPipelineState>,
    ui_sampler: metal::SamplerState,
    font_atlas_texture: metal::Texture,


    ui_generic_nodes: Vec<super::snapshot::GenericUiNode>,
    /// Force the offscreen target back on. It only earns its keep when the
    /// scene renders at a different size than the drawable, or when MSAA has to
    /// be resolved; otherwise it costs a full-screen store and a full-screen
    /// load for nothing, so the default is to draw into the drawable.
    pub force_offscreen: bool,
    /// Draw every mesh as opaque, ignoring its alpha mode. desert_rust_metal
    /// does this implicitly by never reading alphaMode at all.
    pub force_opaque: bool,
    /// Whether to count draws, culls and vertices. The counting is five walks
    /// over the whole draw list and nothing reads the result unless the host
    /// is printing its stats line.
    stats_enabled: bool,
    /// Kept one frame so its GPU timestamps can be read after it completes.
    /// Last frame's submissions, labelled, kept so their GPU timestamps can be
    /// read once they have completed.
    last_command_buffers: Vec<(&'static str, metal::CommandBuffer)>,
    shadow_quality_requested: u8,
    shadow_static: bool,
    shadow_dynamic: bool,
    frame_capture: Option<super::gpu_capture::FrameCapture>,
}

fn normal_matrix_for(model: [[f32; 4]; 4]) -> [[f32; 4]; 4] {
    bevy::math::Mat4::from_cols_array_2d(&model).inverse().transpose().to_cols_array_2d()
}

/// Vertices as authored, plus the index list that references them. Expanding
/// the indices instead would run the vertex shader once per index rather than
/// once per vertex, and give the post-transform cache nothing to reuse.
fn bake_local_mesh_vertices(mesh: &ExtractedMesh) -> (Vec<super::gltf_scene::GltfVertex>, Vec<u32>) {
    let vertex_at = |idx: usize| -> super::gltf_scene::GltfVertex {
        let position = mesh.positions[idx];
        let normal = if mesh.normals.is_empty() { [0.0, 1.0, 0.0] } else { mesh.normals[idx] };
        let uv = if mesh.uvs.is_empty() { [0.0, 0.0] } else { mesh.uvs[idx] };
        let tangent = if mesh.tangents.is_empty() { [1.0, 0.0, 0.0, 1.0] } else { mesh.tangents[idx] };
        let uv2 = if mesh.uv2.is_empty() { [0.0, 0.0] } else { mesh.uv2[idx] };
        super::gltf_scene::GltfVertex {
            position,
            normal: super::gltf_scene::snorm16(normal, 0.0),
            uv,
            tangent: super::gltf_scene::snorm16([tangent[0], tangent[1], tangent[2]], tangent[3]),
            uv2: super::gltf_scene::unorm16(uv2),
        }
    };
    let verts: Vec<_> = (0..mesh.positions.len()).map(vertex_at).collect();
    let indices = if mesh.indices.is_empty() {
        (0..mesh.positions.len() as u32).collect()
    } else {
        mesh.indices.clone()
    };
    (verts, indices)
}


fn ndc_ui_rect(x: f32, y: f32, w: f32, h: f32, view_w: f32, view_h: f32, uv: [[f32; 2]; 4]) -> [UiVertex; 6] {
    let ndc_x0 = (x / view_w) * 2.0 - 1.0;
    let ndc_x1 = ((x + w) / view_w) * 2.0 - 1.0;
    let ndc_y0 = 1.0 - (y / view_h) * 2.0;
    let ndc_y1 = 1.0 - ((y + h) / view_h) * 2.0;
    [
        UiVertex { position: [ndc_x0, ndc_y1], uv: uv[0] },
        UiVertex { position: [ndc_x1, ndc_y1], uv: uv[1] },
        UiVertex { position: [ndc_x1, ndc_y0], uv: uv[2] },
        UiVertex { position: [ndc_x0, ndc_y1], uv: uv[0] },
        UiVertex { position: [ndc_x1, ndc_y0], uv: uv[2] },
        UiVertex { position: [ndc_x0, ndc_y0], uv: uv[3] },
    ]
}

fn ndc_ui_rect_rotated(center_x: f32, center_y: f32, w: f32, h: f32, rotation: f32, view_w: f32, view_h: f32, uv: [[f32; 2]; 4]) -> [UiVertex; 6] {
    let (sin, cos) = rotation.sin_cos();
    let hw = w / 2.0;
    let hh = h / 2.0;
    let to_ndc = |lx: f32, ly: f32| -> [f32; 2] {
        let px = center_x + lx * cos - ly * sin;
        let py = center_y + lx * sin + ly * cos;
        [(px / view_w) * 2.0 - 1.0, 1.0 - (py / view_h) * 2.0]
    };
    let bottom_left = to_ndc(-hw, hh);
    let bottom_right = to_ndc(hw, hh);
    let top_right = to_ndc(hw, -hh);
    let top_left = to_ndc(-hw, -hh);
    [
        UiVertex { position: bottom_left, uv: uv[0] },
        UiVertex { position: bottom_right, uv: uv[1] },
        UiVertex { position: top_right, uv: uv[2] },
        UiVertex { position: bottom_left, uv: uv[0] },
        UiVertex { position: top_right, uv: uv[2] },
        UiVertex { position: top_left, uv: uv[3] },
    ]
}

/// setVertexBytes caps at 4KB, which is 42 of these six-vertex glyphs.
const GLYPHS_PER_UI_BATCH: usize = 42;

/// The pass rebinds the same font atlas for every glyph and the same white
/// texture for every rect, and re-sets a scissor that rarely changes between
/// nodes. Both are tracked here and skipped rather than re-recorded per draw.
struct UiEncoder<'a> {
    encoder: &'a metal::RenderCommandEncoderRef,
    bound_texture: usize,
    scissor: Option<(u64, u64, u64, u64)>,
}

impl UiEncoder<'_> {
    fn set_scissor(&mut self, rect: MTLScissorRect) {
        let key = (rect.x, rect.y, rect.width, rect.height);
        if self.scissor == Some(key) {
            return;
        }
        self.scissor = Some(key);
        self.encoder.set_scissor_rect(rect);
    }

    fn bind(&mut self, texture: &metal::Texture) {
        let key = texture.as_ptr() as usize;
        if self.bound_texture == key {
            return;
        }
        self.bound_texture = key;
        self.encoder.set_fragment_texture(0, Some(texture));
    }
}

fn draw_ui_quad(ui: &mut UiEncoder<'_>, verts: &[UiVertex; 6], tint: [f32; 4], texture: &metal::Texture) {
    draw_ui_verts(ui, verts, MTLPrimitiveType::Triangle, tint, texture);
}

fn draw_ui_verts(ui: &mut UiEncoder<'_>, verts: &[UiVertex], primitive: MTLPrimitiveType, tint: [f32; 4], texture: &metal::Texture) {
    if verts.is_empty() {
        return;
    }
    ui.bind(texture);
    ui.encoder.set_vertex_bytes(0, std::mem::size_of_val(verts) as u64, verts.as_ptr() as *const _);
    let uniforms = UiUniforms { tint };
    ui.encoder.set_fragment_bytes(1, std::mem::size_of::<UiUniforms>() as u64, &uniforms as *const _ as *const _);
    ui.encoder.draw_primitives(primitive, 0, verts.len() as u64);
}

/// Angle 0 = +X, increasing clockwise (+Y is down). `0.0, TAU` sweeps a full circle.
fn arc_fill_verts(center: [f32; 2], radius: f32, start_angle: f32, sweep: f32, view_w: f32, view_h: f32, segments: u32) -> Vec<UiVertex> {
    let to_ndc = |x: f32, y: f32| -> [f32; 2] { [(x / view_w) * 2.0 - 1.0, 1.0 - (y / view_h) * 2.0] };
    let center_ndc = to_ndc(center[0], center[1]);
    let point_at = |theta: f32| {
        let (sin, cos) = theta.sin_cos();
        to_ndc(center[0] + cos * radius, center[1] + sin * radius)
    };
    let mut verts = Vec::with_capacity(segments as usize * 3);
    let mut prev = point_at(start_angle);
    for i in 1..=segments {
        let edge = point_at(start_angle + sweep * (i as f32 / segments as f32));
        verts.push(UiVertex { position: center_ndc, uv: [0.5, 0.5] });
        verts.push(UiVertex { position: prev, uv: [0.5, 0.5] });
        verts.push(UiVertex { position: edge, uv: [0.5, 0.5] });
        prev = edge;
    }
    verts
}

fn arc_stroke_verts(center: [f32; 2], outer_radius: f32, inner_radius: f32, start_angle: f32, sweep: f32, view_w: f32, view_h: f32, segments: u32) -> Vec<UiVertex> {
    let to_ndc = |x: f32, y: f32| -> [f32; 2] { [(x / view_w) * 2.0 - 1.0, 1.0 - (y / view_h) * 2.0] };
    let mut verts = Vec::with_capacity((segments as usize + 1) * 2);
    for i in 0..=segments {
        let theta = start_angle + sweep * (i as f32 / segments as f32);
        let (sin, cos) = theta.sin_cos();
        let outer = to_ndc(center[0] + cos * outer_radius, center[1] + sin * outer_radius);
        let inner = to_ndc(center[0] + cos * inner_radius, center[1] + sin * inner_radius);
        verts.push(UiVertex { position: outer, uv: [0.5, 0.5] });
        verts.push(UiVertex { position: inner, uv: [0.5, 0.5] });
    }
    verts
}

/// Corner (arc-center, start-angle) pairs, top-left/top-right/bottom-right/bottom-left, each a quarter turn.
fn rounded_rect_corners(x: f32, y: f32, w: f32, h: f32, r: f32) -> [([f32; 2], f32); 4] {
    use std::f32::consts::PI;
    [
        ([x + r, y + r], PI),
        ([x + w - r, y + r], -PI / 2.0),
        ([x + w - r, y + h - r], 0.0),
        ([x + r, y + h - r], PI / 2.0),
    ]
}

fn draw_rounded_rect_fill(ui: &mut UiEncoder<'_>, center: [f32; 2], size: [f32; 2], radius: f32, color: [f32; 4], view_w: f32, view_h: f32, white_texture: &metal::Texture) {
    const FULL_UV: [[f32; 2]; 4] = [[0.0, 1.0], [1.0, 1.0], [1.0, 0.0], [0.0, 0.0]];
    const CORNER_SEGMENTS: u32 = 10;
    let [cx, cy] = center;
    let [w, h] = size;
    let r = radius.min(w / 2.0).min(h / 2.0);
    let x = cx - w / 2.0;
    let y = cy - h / 2.0;

    let middle = ndc_ui_rect(x + r, y, w - 2.0 * r, h, view_w, view_h, FULL_UV);
    draw_ui_quad(ui, &middle, color, white_texture);
    let left = ndc_ui_rect(x, y + r, r, h - 2.0 * r, view_w, view_h, FULL_UV);
    draw_ui_quad(ui, &left, color, white_texture);
    let right = ndc_ui_rect(x + w - r, y + r, r, h - 2.0 * r, view_w, view_h, FULL_UV);
    draw_ui_quad(ui, &right, color, white_texture);

    for (corner_center, start_angle) in rounded_rect_corners(x, y, w, h, r) {
        let verts = arc_fill_verts(corner_center, r, start_angle, std::f32::consts::FRAC_PI_2, view_w, view_h, CORNER_SEGMENTS);
        draw_ui_verts(ui, &verts, MTLPrimitiveType::Triangle, color, white_texture);
    }
}

fn draw_rounded_rect_border(
    ui: &mut UiEncoder<'_>,
    center: [f32; 2],
    size: [f32; 2],
    radius: f32,
    border_width: f32,
    color: [f32; 4],
    view_w: f32,
    view_h: f32,
    white_texture: &metal::Texture,
) {
    const FULL_UV: [[f32; 2]; 4] = [[0.0, 1.0], [1.0, 1.0], [1.0, 0.0], [0.0, 0.0]];
    const CORNER_SEGMENTS: u32 = 10;
    let [cx, cy] = center;
    let [w, h] = size;
    let r = radius.min(w / 2.0).min(h / 2.0);
    let bw = border_width.min(r).max(0.0);
    let x = cx - w / 2.0;
    let y = cy - h / 2.0;

    for (ex, ey, ew, eh) in [(x + r, y, w - 2.0 * r, bw), (x + r, y + h - bw, w - 2.0 * r, bw), (x, y + r, bw, h - 2.0 * r), (x + w - bw, y + r, bw, h - 2.0 * r)] {
        let quad = ndc_ui_rect(ex, ey, ew.max(0.0), eh.max(0.0), view_w, view_h, FULL_UV);
        draw_ui_quad(ui, &quad, color, white_texture);
    }
    for (corner_center, start_angle) in rounded_rect_corners(x, y, w, h, r) {
        let verts = arc_stroke_verts(corner_center, r, (r - bw).max(0.0), start_angle, std::f32::consts::FRAC_PI_2, view_w, view_h, CORNER_SEGMENTS);
        draw_ui_verts(ui, &verts, MTLPrimitiveType::TriangleStrip, color, white_texture);
    }
}

fn draw_rect_border(
    ui: &mut UiEncoder<'_>,
    center: [f32; 2],
    size: [f32; 2],
    border_width: f32,
    color: [f32; 4],
    view_w: f32,
    view_h: f32,
    white_texture: &metal::Texture,
) {
    const FULL_UV: [[f32; 2]; 4] = [[0.0, 1.0], [1.0, 1.0], [1.0, 0.0], [0.0, 0.0]];
    let [cx, cy] = center;
    let [w, h] = size;
    let x = cx - w / 2.0;
    let y = cy - h / 2.0;
    let bw = border_width.min(w / 2.0).min(h / 2.0);
    for (ex, ey, ew, eh) in [(x, y, w, bw), (x, y + h - bw, w, bw), (x, y, bw, h), (x + w - bw, y, bw, h)] {
        let quad = ndc_ui_rect(ex, ey, ew, eh, view_w, view_h, FULL_UV);
        draw_ui_quad(ui, &quad, color, white_texture);
    }
}

/// `canvas_w`/`canvas_h` must be this pass's actual bound-texture size (e.g.
/// `render_width`/`render_height`), not the device's native resolution — this
/// pass may render into a `render_resolution_scale`-downscaled intermediate.
fn generic_ui_scissor_rect(clip: Option<[f32; 4]>, canvas_w: f32, canvas_h: f32, view_w: f32, view_h: f32) -> MTLScissorRect {
    let phys_w = canvas_w.max(1.0);
    let phys_h = canvas_h.max(1.0);
    let Some([cx, cy, cw, ch]) = clip else {
        return MTLScissorRect { x: 0, y: 0, width: phys_w as u64, height: phys_h as u64 };
    };
    let to_px_x = |v: f32| (v / view_w) * phys_w;
    let to_px_y = |v: f32| (v / view_h) * phys_h;
    let x = to_px_x(cx).clamp(0.0, phys_w - 1.0);
    let y = to_px_y(cy).clamp(0.0, phys_h - 1.0);
    let width = to_px_x(cw).clamp(1.0, phys_w - x);
    let height = to_px_y(ch).clamp(1.0, phys_h - y);
    MTLScissorRect { x: x as u64, y: y as u64, width: width as u64, height: height as u64 }
}

fn draw_generic_ui_nodes(
    ui: &mut UiEncoder<'_>,
    nodes: &[super::snapshot::GenericUiNode],
    canvas_w: f32,
    canvas_h: f32,
    view_w: f32,
    view_h: f32,
    white_texture: &metal::Texture,
    font_atlas_texture: &metal::Texture,
    images: &std::collections::HashMap<AssetId<Image>, metal::Texture>,
) {
    const FULL_UV: [[f32; 2]; 4] = [[0.0, 1.0], [1.0, 1.0], [1.0, 0.0], [0.0, 0.0]];
    const CIRCLE_SEGMENTS: u32 = 40;
    let mut glyphs: Vec<UiVertex> = Vec::new();
    for node in nodes {
        ui.set_scissor(generic_ui_scissor_rect(node.clip_rect, canvas_w, canvas_h, view_w, view_h));

        let [cx, cy] = node.center;
        let [w, h] = node.size;
        // Only a square node with a fully-rounded radius is a circle; a non-square one
        // (Bevy clamps its radius to half the *shorter* side) is a pill, which the
        // rounded-rect path already draws correctly.
        let is_circle = node.corner_radius > 0.0 && node.corner_radius >= w.min(h) / 2.0 - 0.5 && (w - h).abs() <= 0.5;
        let is_rounded = !is_circle && node.corner_radius > 0.5;

        if is_circle {
            let radius = w.min(h) / 2.0;
            if let Some(color) = node.background_color {
                let verts = arc_fill_verts([cx, cy], radius, 0.0, std::f32::consts::TAU, view_w, view_h, CIRCLE_SEGMENTS);
                draw_ui_verts(ui, &verts, MTLPrimitiveType::Triangle, color, white_texture);
            }
            if let Some(border_color) = node.border_color {
                let inner = (radius - node.border_width).max(0.0);
                let verts = arc_stroke_verts([cx, cy], radius, inner, 0.0, std::f32::consts::TAU, view_w, view_h, CIRCLE_SEGMENTS);
                draw_ui_verts(ui, &verts, MTLPrimitiveType::TriangleStrip, border_color, white_texture);
            }
        } else if is_rounded {
            if let Some(color) = node.background_color {
                draw_rounded_rect_fill(ui, [cx, cy], [w, h], node.corner_radius, color, view_w, view_h, white_texture);
            }
            if let Some(border_color) = node.border_color {
                draw_rounded_rect_border(ui, [cx, cy], [w, h], node.corner_radius, node.border_width, border_color, view_w, view_h, white_texture);
            }
        } else {
            if let Some(color) = node.background_color {
                let quad = ndc_ui_rect_rotated(cx, cy, w, h, node.rotation, view_w, view_h, FULL_UV);
                draw_ui_quad(ui, &quad, color, white_texture);
            }
            if let Some(border_color) = node.border_color {
                draw_rect_border(ui, [cx, cy], [w, h], node.border_width, border_color, view_w, view_h, white_texture);
            }
        }

        if let Some(image) = node.image.as_ref().and_then(|image| images.get(&image.id)) {
            let quad = ndc_ui_rect_rotated(cx, cy, w, h, node.rotation, view_w, view_h, FULL_UV);
            draw_ui_quad(ui, &quad, [1.0, 1.0, 1.0, 1.0], image);
        }
        if let Some(text) = &node.text {
            let scale = text.font_size / GLYPH_H as f32;
            let upper = text.content.to_ascii_uppercase();
            let mut cursor_x = cx - w / 2.0;
            let y = cy - h / 2.0;
            glyphs.clear();
            for ch in upper.chars() {
                let Some(index) = glyph_index(ch) else { continue };
                let (u0, u1) = glyph_uv_rect(index);
                let uv = [[u0, 1.0], [u1, 1.0], [u1, 0.0], [u0, 0.0]];
                let glyph_w = GLYPH_W as f32 * scale;
                let glyph_h = GLYPH_H as f32 * scale;
                glyphs.extend_from_slice(&ndc_ui_rect(cursor_x, y, glyph_w, glyph_h, view_w, view_h, uv));
                cursor_x += glyph_w + scale;
            }
            // One draw per line of text rather than per letter, up to what
            // setVertexBytes will carry in a single call.
            for batch in glyphs.chunks(GLYPHS_PER_UI_BATCH * 6) {
                draw_ui_verts(ui, batch, MTLPrimitiveType::Triangle, text.color, font_atlas_texture);
            }
        }
    }

    ui.set_scissor(generic_ui_scissor_rect(None, canvas_w, canvas_h, view_w, view_h));
}

/// Render layers, reduced to the only question a pass has to ask.
/// Blocks on the frame just submitted, which is why it only ever runs once:
/// the readback is worth a stall, a stall every frame is not. Free-standing so
/// it borrows only the slot, not the whole renderer the drawable came from.
fn capture_if_due(
    slot: &mut Option<crate::screen_capture::ScreenCapture>,
    command_buffer: &metal::CommandBufferRef,
    drawable: &metal::TextureRef,
) {
    if !slot.as_ref().is_some_and(|c| c.due()) {
        return;
    }
    command_buffer.wait_until_completed();
    if let Some(capture) = slot.take() {
        capture.write(drawable);
    }
}

/// Layer filter and frustum test together, so no caller can remember one and
/// forget the other.
///
/// The viewmodel is exempt: it is positioned relative to the camera and sits
/// inside the near plane, where a conservative sphere test rejects it.
fn draws_in_view(mesh: &RealGameMeshDraw, planes: &[[f32; 4]; 6]) -> bool {
    mesh.viewmodel || !sphere_outside(planes, mesh.bounds_centre, mesh.bounds_radius)
}


/// Joint matrices for every skinned actor, in one resident buffer per frame
/// in flight rather than a fresh `MTLBuffer` per actor per pass.
///
/// An allocation is not just a malloc here: the driver keeps a set of
/// resident resources and every new buffer is a hash and an insert into it,
/// with the old one left for ARC. At thirty zombies, two passes and 120 Hz
/// that was running seven thousand times a second to move a few kilobytes.
///
/// Three buffers, cycled, because the GPU is still reading the previous
/// frame's matrices while the CPU writes the next.
struct JointRing {
    buffers: Vec<metal::Buffer>,
    capacity: u64,
    frame: std::cell::Cell<usize>,
    cursor: std::cell::Cell<u64>,
}

const JOINT_RING_FRAMES: usize = 3;
/// Offsets handed to setVertexBuffer stay 256-aligned: the device address
/// space only asks for 4, but 256 is what every address space accepts.
const JOINT_RING_ALIGN: u64 = 256;

impl JointRing {
    fn new(device: &metal::DeviceRef, capacity: u64) -> Self {
        let buffers = (0..JOINT_RING_FRAMES)
            .map(|_| device.new_buffer(capacity, MTLResourceOptions::StorageModeShared))
            .collect();
        Self { buffers, capacity, frame: std::cell::Cell::new(0), cursor: std::cell::Cell::new(0) }
    }

    fn begin_frame(&self) {
        self.frame.set((self.frame.get() + 1) % JOINT_RING_FRAMES);
        self.cursor.set(0);
    }

    /// `None` when this frame's buffer is full, which leaves the caller to
    /// allocate: correctness first, and a frame that overflows is one whose
    /// capacity was guessed too low rather than one that should drop actors.
    fn push(&self, matrices: &[[[f32; 4]; 4]]) -> Option<(&metal::BufferRef, u64)> {
        let bytes = std::mem::size_of_val(matrices) as u64;
        let offset = self.cursor.get();
        if offset + bytes > self.capacity {
            return None;
        }
        let buffer = &self.buffers[self.frame.get()];
        unsafe {
            std::ptr::copy_nonoverlapping(
                matrices.as_ptr() as *const u8,
                (buffer.contents() as *mut u8).add(offset as usize),
                bytes as usize,
            );
        }
        self.cursor.set((offset + bytes).div_ceil(JOINT_RING_ALIGN) * JOINT_RING_ALIGN);
        Some((buffer.as_ref(), offset))
    }
}

/// The six planes of a view-projection, each `ax + by + cz + d = 0` with the
/// inside positive.
///
/// Gribb-Hartmann: the rows of the matrix already describe the clip volume, so
/// the planes fall out of adding and subtracting them. No inverse, no corner
/// reconstruction.
fn frustum_planes(view_projection: [[f32; 4]; 4]) -> [[f32; 4]; 6] {
    let m = |row: usize, col: usize| view_projection[col][row];
    let mut planes = [[0.0f32; 4]; 6];
    for (index, (row, sign)) in [(0, 1.0), (0, -1.0), (1, 1.0), (1, -1.0), (2, 1.0), (2, -1.0)]
        .into_iter()
        .enumerate()
    {
        // Metal clips z to 0..w, not -w..w, so the near plane is row 2 alone
        // rather than w + row 2. Getting this wrong culls everything in front
        // of the camera and nothing behind it.
        let near_only = index == 4;
        for component in 0..4 {
            let w = if near_only { 0.0 } else { m(3, component) };
            planes[index][component] = w + sign * m(row, component);
        }
        let normal = [planes[index][0], planes[index][1], planes[index][2]];
        let length = (normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]).sqrt();
        if length > 1e-6 {
            for component in 0..4 {
                planes[index][component] /= length;
            }
        }
    }
    planes
}

/// A zero radius means the host supplied no bounds for this actor, so it draws
/// unconditionally rather than being culled on a sphere nobody filled in.
fn actor_in_view(actor: &super::skinned_actor::SkinnedActorDraw, planes: &[[f32; 4]; 6]) -> bool {
    actor.bounds_radius <= 0.0 || !sphere_outside(planes, actor.bounds_centre, actor.bounds_radius)
}

fn sphere_outside(planes: &[[f32; 4]; 6], centre: [f32; 3], radius: f32) -> bool {
    planes.iter().any(|p| {
        p[0] * centre[0] + p[1] * centre[1] + p[2] * centre[2] + p[3] < -radius
    })
}

/// A sphere around the mesh in its own space, kept with the uploaded geometry
/// so it is computed once rather than every frame.
fn local_bounds(positions: &[[f32; 3]]) -> ([f32; 3], f32) {
    if positions.is_empty() {
        return ([0.0; 3], 0.0);
    }
    let mut lo = [f32::MAX; 3];
    let mut hi = [f32::MIN; 3];
    for p in positions {
        for axis in 0..3 {
            lo[axis] = lo[axis].min(p[axis]);
            hi[axis] = hi[axis].max(p[axis]);
        }
    }
    let centre = [
        (lo[0] + hi[0]) * 0.5,
        (lo[1] + hi[1]) * 0.5,
        (lo[2] + hi[2]) * 0.5,
    ];
    // The corner distance, not the longest half-extent: a sphere that only
    // reaches the face centres leaves the corners of a box outside it.
    let half = [
        (hi[0] - lo[0]) * 0.5,
        (hi[1] - lo[1]) * 0.5,
        (hi[2] - lo[2]) * 0.5,
    ];
    (centre, (half[0] * half[0] + half[1] * half[1] + half[2] * half[2]).sqrt())
}

/// One slot of the persistent draw list. Everything above `model_matrix` is
/// settled when the entity appears and never touched again; the list survives
/// between frames so a scene that does not change costs nothing to prepare.
///
/// The textures are owned handles rather than borrows into
/// `real_game_textures` -- a borrow would make this self-referential and it
/// could not live on the renderer at all. A `metal::Texture` is a refcount,
/// so the clone is a retain, not a copy of any pixels.
#[derive(Clone)]
struct RealGameMeshDraw {
    /// Skipped by the scope pass: the gun is not part of the world the scope
    /// is looking at, and drawn there it shows up inside its own eyepiece.
    viewmodel: bool,
    vertex_buffer: metal::Buffer,
    index_buffer: metal::Buffer,
    index_count: usize,
    /// u16 indices where the mesh's vertex count allows it.
    narrow_index: bool,
    unique_vertices: usize,
    pub base_color_factor: [f32; 4],
    pub emissive: [f32; 4],
    texture: Option<metal::Texture>,
    normal_texture: Option<metal::Texture>,
    /// Which assets this slot holds, so leaving the list can decrement the
    /// right counts.
    mesh_id: AssetId<Mesh>,
    texture_id: Option<AssetId<Image>>,
    normal_texture_id: Option<AssetId<Image>>,

    is_blend: bool,

    /// World-space bounding sphere, for the frustum test. A sphere rather than
    /// a box because a box has to be re-fitted after rotation and a sphere
    /// only needs its centre moved and its radius scaled.
    bounds_centre: [f32; 3],
    bounds_radius: f32,

    pub model_matrix: [[f32; 4]; 4],
}


/// Counts a draw's assets as used by one more slot.
///
/// Always called before the slot it replaces is released, so an id that both
/// the outgoing and the incoming draw name never passes through zero and never
/// loses its GPU buffers.
///
/// Free functions rather than methods: the drawable holds a borrow of the
/// renderer for the length of the frame, so nothing in here may take `&mut
/// self`.
/// GPU assets no live draw names any more. Kept so a respawn rebinds them
/// instead of re-uploading, and evicted oldest-first once they outgrow the
/// budget. Sized for the churning set -- gore, casings, decals -- not for the
/// map, whose slots never leave.
const RETAINED_GPU_BUDGET_BYTES: u64 = 32 << 20;

/// Three switches for the parts of the opaque pass whose cost depends on what
/// is on screen, so a view that stutters can be bisected without a rebuild.
/// Each one restores the behaviour from before this pass was reordered.
fn env_off(name: &str) -> bool {
    std::env::var(name).is_ok()
}

struct PreparedActor {
    model_matrix: [[f32; 4]; 4],
    joint_buffer: metal::Buffer,
    joint_offset: u64,
}

enum RetainedAsset {
    Mesh(AssetId<Mesh>),
    Texture(AssetId<Image>),
}

/// Takes an id back out of the retained queue, because a draw names it again.
fn unretain(
    retained: &mut std::collections::VecDeque<(RetainedAsset, u64)>,
    retained_bytes: &mut u64,
    wanted: &RetainedAsset,
) {
    if retained.is_empty() {
        return;
    }
    let same = |held: &RetainedAsset| match (held, wanted) {
        (RetainedAsset::Mesh(a), RetainedAsset::Mesh(b)) => a == b,
        (RetainedAsset::Texture(a), RetainedAsset::Texture(b)) => a == b,
        _ => false,
    };
    if let Some(at) = retained.iter().position(|(held, _)| same(held)) {
        *retained_bytes -= retained[at].1;
        retained.remove(at);
    }
}

fn retain_draw_assets(
    mesh_refs: &mut std::collections::HashMap<AssetId<Mesh>, u32>,
    texture_refs: &mut std::collections::HashMap<AssetId<Image>, u32>,
    retained: &mut std::collections::VecDeque<(RetainedAsset, u64)>,
    retained_bytes: &mut u64,
    draw: &RealGameMeshDraw,
) {
    *mesh_refs.entry(draw.mesh_id).or_insert(0) += 1;
    unretain(retained, retained_bytes, &RetainedAsset::Mesh(draw.mesh_id));
    for id in [draw.texture_id, draw.normal_texture_id].into_iter().flatten() {
        *texture_refs.entry(id).or_insert(0) += 1;
        unretain(retained, retained_bytes, &RetainedAsset::Texture(id));
    }
}

/// Counts a draw's assets as used by one fewer slot, dropping whatever reaches
/// nobody.
/// Drops the draw's claim on its assets. Nothing is freed here: an unclaimed
/// asset joins the retained queue and only leaves the GPU once that queue is
/// over budget, so a gore piece respawning after its 20s lifetime rebinds what
/// is already uploaded.
fn release_draw_assets(
    mesh_refs: &mut std::collections::HashMap<AssetId<Mesh>, u32>,
    texture_refs: &mut std::collections::HashMap<AssetId<Image>, u32>,
    geometry: &std::collections::HashMap<
        AssetId<Mesh>,
        (metal::Buffer, metal::Buffer, usize, usize, [f32; 3], f32, bool),
    >,
    textures: &std::collections::HashMap<AssetId<Image>, metal::Texture>,
    retained: &mut std::collections::VecDeque<(RetainedAsset, u64)>,
    retained_bytes: &mut u64,
    draw: RealGameMeshDraw,
) {
    if let Some(count) = mesh_refs.get_mut(&draw.mesh_id) {
        *count -= 1;
        if *count == 0 {
            mesh_refs.remove(&draw.mesh_id);
            if let Some((vertices, indices, ..)) = geometry.get(&draw.mesh_id) {
                let bytes = vertices.length() + indices.length();
                *retained_bytes += bytes;
                retained.push_back((RetainedAsset::Mesh(draw.mesh_id), bytes));
            }
        }
    }
    for id in [draw.texture_id, draw.normal_texture_id].into_iter().flatten() {
        let Some(count) = texture_refs.get_mut(&id) else { continue };
        *count -= 1;
        if *count > 0 {
            continue;
        }
        texture_refs.remove(&id);
        if let Some(texture) = textures.get(&id) {
            let bytes = texture.allocated_size();
            *retained_bytes += bytes;
            retained.push_back((RetainedAsset::Texture(id), bytes));
        }
    }
}

impl NativeMetalRenderer {
    /// Hands over the map's baked lightmap. RGBA8 bytes, R ambient occlusion
    /// and G sun visibility, square and `size` on a side.
    pub fn set_lightmap(&mut self, rgba: &[u8], size: u32) {
        let descriptor = metal::TextureDescriptor::new();
        descriptor.set_width(size as u64);
        descriptor.set_height(size as u64);
        descriptor.set_pixel_format(MTLPixelFormat::RGBA8Unorm);
        descriptor.set_usage(metal::MTLTextureUsage::ShaderRead);
        let texture = self.device.new_texture(&descriptor);
        texture.replace_region(
            metal::MTLRegion {
                origin: metal::MTLOrigin { x: 0, y: 0, z: 0 },
                size: metal::MTLSize { width: size as u64, height: size as u64, depth: 1 },
            },
            0,
            rgba.as_ptr() as *const _,
            (size * 4) as u64,
        );
        self.lightmap = Some(texture);
    }

    /// Hands over the baked probe grid. Two RGBA8 volumes, `dims` probes on
    /// each side, the first probe centred on `origin` and `spacing` metres
    /// between them.
    pub fn set_light_probes(
        &mut self,
        positive: &[u8],
        negative: &[u8],
        dims: [u32; 3],
        origin: [f32; 3],
        spacing: f32,
    ) {
        let volume = |bytes: &[u8]| {
            let descriptor = metal::TextureDescriptor::new();
            descriptor.set_texture_type(metal::MTLTextureType::D3);
            descriptor.set_width(dims[0] as u64);
            descriptor.set_height(dims[1] as u64);
            descriptor.set_depth(dims[2] as u64);
            descriptor.set_pixel_format(MTLPixelFormat::RGBA8Unorm);
            descriptor.set_usage(metal::MTLTextureUsage::ShaderRead);
            let texture = self.device.new_texture(&descriptor);
            // replace_region_in_slice, not replace_region: the three-argument
            // form has no image stride and would upload the first depth slice
            // over and over.
            texture.replace_region_in_slice(
                metal::MTLRegion {
                    origin: metal::MTLOrigin { x: 0, y: 0, z: 0 },
                    size: metal::MTLSize {
                        width: dims[0] as u64,
                        height: dims[1] as u64,
                        depth: dims[2] as u64,
                    },
                },
                0,
                0,
                bytes.as_ptr() as *const _,
                (dims[0] * 4) as u64,
                (dims[0] * dims[1] * 4) as u64,
            );
            texture
        };
        self.probes = Some((volume(positive), volume(negative)));
        self.probe_origin_spacing = [origin[0], origin[1], origin[2], spacing];
        self.probe_dims = [dims[0] as f32, dims[1] as f32, dims[2] as f32, 0.0];

        // Clamped, not repeated: a point just outside the grid should read the
        // edge probe, not wrap to the far side of the city.
        let sampler = metal::SamplerDescriptor::new();
        sampler.set_min_filter(metal::MTLSamplerMinMagFilter::Linear);
        sampler.set_mag_filter(metal::MTLSamplerMinMagFilter::Linear);
        sampler.set_address_mode_s(metal::MTLSamplerAddressMode::ClampToEdge);
        sampler.set_address_mode_t(metal::MTLSamplerAddressMode::ClampToEdge);
        sampler.set_address_mode_r(metal::MTLSamplerAddressMode::ClampToEdge);
        self.probe_sampler = Some(self.device.new_sampler(&sampler));
    }

    /// How many joint matrices a pose for `name` has to carry, or 0 if no such
    /// model was preloaded. A caller that guesses this wrong gets a silently
    /// wrong pose rather than an error.
    pub fn skinned_joint_count(&self, name: &str) -> usize {
        self.skinned_models.get(name).map_or(0, |slot| slot.model.joint_count())
    }

    /// `skinned_models` names the skinned glTF files to preload, as
    /// `(model_name, path_relative_to_assets, scale, y_offset, facing_radians)`.
    pub unsafe fn new(view: *mut Object, skinned_model_specs: &[SkinnedModelSpec], features: ShaderFeatures) -> Self {
        let device = Device::system_default().expect("no Metal device");
        let raw_layer: *mut Object = msg_send![view, layer];

        let retained: *mut Object = msg_send![raw_layer, retain];
        let layer = unsafe { MetalLayer::from_ptr(retained as *mut CAMetalLayer) };
        layer.set_device(&device);
        layer.set_pixel_format(MTLPixelFormat::BGRA8Unorm);

        let bounds: CGRect = msg_send![view, bounds];
        let scale = layer.contents_scale();
        layer.set_drawable_size(CGSize::new(bounds.size.width * scale, bounds.size.height * scale));

        let queue = device.new_command_queue();
        let frame_capture = super::gpu_capture::FrameCapture::from_env(&device, &queue);

        let compile_options = CompileOptions::new();

        let depth_descriptor = metal::DepthStencilDescriptor::new();
        depth_descriptor.set_depth_compare_function(metal::MTLCompareFunction::Less);
        depth_descriptor.set_depth_write_enabled(true);
        let depth_state = device.new_depth_stencil_state(&depth_descriptor);

        let blend_depth_descriptor = metal::DepthStencilDescriptor::new();
        blend_depth_descriptor.set_depth_compare_function(metal::MTLCompareFunction::Less);
        blend_depth_descriptor.set_depth_write_enabled(false);
        let blend_depth_state = device.new_depth_stencil_state(&blend_depth_descriptor);

        let ui_depth_descriptor = metal::DepthStencilDescriptor::new();
        ui_depth_descriptor.set_depth_compare_function(metal::MTLCompareFunction::Always);
        ui_depth_descriptor.set_depth_write_enabled(false);
        let ui_depth_state = device.new_depth_stencil_state(&ui_depth_descriptor);

        // Blending is switched on only for the meshes that need it. Leaving it
        // on for opaque geometry costs a read-modify-write of the tile's colour
        // buffer on every fragment of the whole scene.
        fn build_sponza_descriptor(
            vertex_fn: &metal::FunctionRef,
            fragment_fn: &metal::FunctionRef,
            sample_count: u64,
            blending: bool,
        ) -> RenderPipelineDescriptor {
            let descriptor = RenderPipelineDescriptor::new();
            descriptor.set_vertex_function(Some(vertex_fn));
            descriptor.set_fragment_function(Some(fragment_fn));
            let color_attachment = descriptor.color_attachments().object_at(0).unwrap();
            color_attachment.set_pixel_format(MTLPixelFormat::BGRA8Unorm);

            color_attachment.set_blending_enabled(blending);
            color_attachment.set_rgb_blend_operation(metal::MTLBlendOperation::Add);
            color_attachment.set_alpha_blend_operation(metal::MTLBlendOperation::Add);
            color_attachment.set_source_rgb_blend_factor(metal::MTLBlendFactor::SourceAlpha);
            color_attachment.set_source_alpha_blend_factor(metal::MTLBlendFactor::SourceAlpha);
            color_attachment.set_destination_rgb_blend_factor(metal::MTLBlendFactor::OneMinusSourceAlpha);
            color_attachment.set_destination_alpha_blend_factor(metal::MTLBlendFactor::OneMinusSourceAlpha);
            descriptor.set_depth_attachment_pixel_format(MTLPixelFormat::Depth32Float);
            descriptor.set_raster_sample_count(sample_count);
            descriptor
        }

        // Printed so a switch that silently fails to reach the shader cannot be
        // mistaken for a switch that made no difference.
        let build_scene_pipelines = |defines: String, blending: bool| {
            println!("[config] pipeline blending={blending} defines: {}", defines.replace('\n', " "));
            let source = format!("{defines}{SPONZA_SHADER_SOURCE}");
            match device.new_library_with_source(&source, &compile_options) {
                Ok(library) => {
                    let vertex_fn = library.get_function("sponza_vertex", None).unwrap();
                    let fragment_fn = library.get_function("sponza_fragment", None).unwrap();
                    let msaa = build_sponza_descriptor(&vertex_fn, &fragment_fn, MSAA_SAMPLE_COUNT, blending);
                    let no_msaa = build_sponza_descriptor(&vertex_fn, &fragment_fn, 1, blending);
                    (
                        device.new_render_pipeline_state(&msaa).ok(),
                        device.new_render_pipeline_state(&no_msaa).ok(),
                    )
                }
                Err(err) => {
                    eprintln!("[native_metal_renderer] sponza shader compile failed: {err}");
                    (None, None)
                }
            }
        };

        let (sponza_pipeline, sponza_pipeline_no_msaa) = build_scene_pipelines(features.defines(), false);
        // With shadows off the passes that fill the shadow maps are skipped, so
        // sampling them would read stale depth. This variant does not sample
        // them at all, which also drops the per-pixel cost.
        let unlit_by_shadows = ShaderFeatures { shadows: false, ..features };
        let (sponza_pipeline_no_shadows, sponza_pipeline_no_shadows_no_msaa) =
            build_scene_pipelines(unlit_by_shadows.defines(), false);
        // desert_city ships no normal maps at all, so without these every one
        // of its pixels sampled the 1x1 flat fallback and ran the whole
        // tangent-space reconstruction to arrive back at the geometric normal.
        let flat_normal = ShaderFeatures { normal_map: false, ..features };
        let (sponza_pipeline_flat_normal, sponza_pipeline_flat_normal_no_msaa) =
            build_scene_pipelines(flat_normal.defines(), false);
        let flat_normal_unlit = ShaderFeatures { shadows: false, normal_map: false, ..features };
        let (sponza_pipeline_no_shadows_flat_normal, sponza_pipeline_no_shadows_flat_normal_no_msaa) =
            build_scene_pipelines(flat_normal_unlit.defines(), false);
        // Transparent surfaces are drawn last, over geometry that is already
        // shaded, and cannot reject anything behind them. Shadow lookups,
        // normal mapping and specular are the costliest things in the shader
        // and the least visible there, so the blended pass gets a variant
        // without them. Diffuse lighting stays, so the look barely moves.
        let blend_features = ShaderFeatures {
            shadows: false,
            normal_map: false,
            specular: false,
            point_light: features.point_light,
            flat: features.flat,
        };
        let (sponza_blend_pipeline, sponza_blend_pipeline_no_msaa) =
            build_scene_pipelines(blend_features.defines(), true);

        let sampler_descriptor = metal::SamplerDescriptor::new();
        sampler_descriptor.set_min_filter(metal::MTLSamplerMinMagFilter::Linear);
        sampler_descriptor.set_mag_filter(metal::MTLSamplerMinMagFilter::Linear);
        // Without this the sampler ignores the mip chain entirely and every
        // minified fragment reads level 0.
        sampler_descriptor.set_mip_filter(metal::MTLSamplerMipFilter::Linear);
        sampler_descriptor.set_address_mode_s(metal::MTLSamplerAddressMode::Repeat);
        sampler_descriptor.set_address_mode_t(metal::MTLSamplerAddressMode::Repeat);
        let sponza_sampler = device.new_sampler(&sampler_descriptor);

        let white_texture_descriptor = metal::TextureDescriptor::new();
        white_texture_descriptor.set_pixel_format(MTLPixelFormat::RGBA8Unorm);
        white_texture_descriptor.set_width(1);
        white_texture_descriptor.set_height(1);
        white_texture_descriptor.set_storage_mode(metal::MTLStorageMode::Shared);
        let white_texture = device.new_texture(&white_texture_descriptor);
        white_texture.replace_region(metal::MTLRegion::new_2d(0, 0, 1, 1), 0, [255u8, 255, 255, 255].as_ptr() as *const _, 4);


        let flat_normal_descriptor = metal::TextureDescriptor::new();
        flat_normal_descriptor.set_pixel_format(MTLPixelFormat::RGBA8Unorm);
        flat_normal_descriptor.set_width(1);
        flat_normal_descriptor.set_height(1);
        flat_normal_descriptor.set_storage_mode(metal::MTLStorageMode::Shared);
        let flat_normal_texture = device.new_texture(&flat_normal_descriptor);
        flat_normal_texture.replace_region(metal::MTLRegion::new_2d(0, 0, 1, 1), 0, [128u8, 128, 255, 255].as_ptr() as *const _, 4);


        let shadow_map_descriptor = metal::TextureDescriptor::new();
        shadow_map_descriptor.set_pixel_format(MTLPixelFormat::Depth32Float);
        shadow_map_descriptor.set_width(SHADOW_MAP_SIZE);
        shadow_map_descriptor.set_height(SHADOW_MAP_SIZE);
        shadow_map_descriptor.set_storage_mode(metal::MTLStorageMode::Private);
        shadow_map_descriptor.set_usage(metal::MTLTextureUsage::RenderTarget | metal::MTLTextureUsage::ShaderRead);
        let shadow_map = device.new_texture(&shadow_map_descriptor);
        let shadow_map_point = device.new_texture(&shadow_map_descriptor);

        let shadow_depth_descriptor = metal::DepthStencilDescriptor::new();
        shadow_depth_descriptor.set_depth_compare_function(metal::MTLCompareFunction::Less);
        shadow_depth_descriptor.set_depth_write_enabled(true);
        let shadow_depth_state = device.new_depth_stencil_state(&shadow_depth_descriptor);

        let shadow_sampler_descriptor = metal::SamplerDescriptor::new();
        shadow_sampler_descriptor.set_min_filter(metal::MTLSamplerMinMagFilter::Linear);
        shadow_sampler_descriptor.set_mag_filter(metal::MTLSamplerMinMagFilter::Linear);
        shadow_sampler_descriptor.set_address_mode_s(metal::MTLSamplerAddressMode::ClampToEdge);
        shadow_sampler_descriptor.set_address_mode_t(metal::MTLSamplerAddressMode::ClampToEdge);
        shadow_sampler_descriptor.set_compare_function(metal::MTLCompareFunction::LessEqual);
        let shadow_sampler = device.new_sampler(&shadow_sampler_descriptor);

        let (shadow_pipeline_sponza, shadow_pipeline_skinned) =
            match device.new_library_with_source(SHADOW_SHADER_SOURCE, &compile_options) {
                Ok(shadow_library) => {
                    let sponza_shadow_vertex_fn = shadow_library.get_function("shadow_vertex_sponza", None).unwrap();
                    let sponza_descriptor = RenderPipelineDescriptor::new();
                    sponza_descriptor.set_vertex_function(Some(&sponza_shadow_vertex_fn));
                    sponza_descriptor.set_fragment_function(None);
                    sponza_descriptor.set_depth_attachment_pixel_format(MTLPixelFormat::Depth32Float);
                    let skinned_shadow_vertex_fn = shadow_library.get_function("shadow_vertex_skinned", None).unwrap();
                    let skinned_descriptor = RenderPipelineDescriptor::new();
                    skinned_descriptor.set_vertex_function(Some(&skinned_shadow_vertex_fn));
                    skinned_descriptor.set_fragment_function(None);
                    skinned_descriptor.set_depth_attachment_pixel_format(MTLPixelFormat::Depth32Float);
                    (
                        device.new_render_pipeline_state(&sponza_descriptor).ok(),
                        device.new_render_pipeline_state(&skinned_descriptor).ok(),
                    )
                }
                Err(err) => {
                    eprintln!("[native_metal_renderer] shadow shader compile failed: {err}");
                    (None, None)
                }
            };

        fn build_skinned_descriptor(
            vertex_fn: &metal::FunctionRef,
            fragment_fn: &metal::FunctionRef,
            sample_count: u64,
        ) -> RenderPipelineDescriptor {
            let descriptor = RenderPipelineDescriptor::new();
            descriptor.set_vertex_function(Some(vertex_fn));
            descriptor.set_fragment_function(Some(fragment_fn));
            descriptor.color_attachments().object_at(0).unwrap().set_pixel_format(MTLPixelFormat::BGRA8Unorm);
            descriptor.set_depth_attachment_pixel_format(MTLPixelFormat::Depth32Float);
            descriptor.set_raster_sample_count(sample_count);
            descriptor
        }

        let (skinned_pipeline, skinned_pipeline_no_msaa) =
            match device.new_library_with_source(SKINNED_SHADER_SOURCE, &compile_options) {
                Ok(library) => {
                    let vertex_fn = library.get_function("skinned_vertex", None).unwrap();
                    let fragment_fn = library.get_function("skinned_fragment", None).unwrap();
                    let msaa_descriptor = build_skinned_descriptor(&vertex_fn, &fragment_fn, MSAA_SAMPLE_COUNT);
                    let no_msaa_descriptor = build_skinned_descriptor(&vertex_fn, &fragment_fn, 1);
                    (
                        device.new_render_pipeline_state(&msaa_descriptor).ok(),
                        device.new_render_pipeline_state(&no_msaa_descriptor).ok(),
                    )
                }
                Err(err) => {
                    eprintln!("[native_metal_renderer] skinned shader compile failed: {err}");
                    (None, None)
                }
            };

        let resource_path = bundle_resource_path().unwrap_or_default();
        let mut skinned_models = super::skinned_actor::SkinnedModelRegistry::default();
        for spec in skinned_model_specs {
            let path = std::path::Path::new(&resource_path).join("assets").join(spec.asset_path);
            if let Some(model) = super::skinned_gltf::load_skinned_gltf(&device, &path) {
                skinned_models.register(spec.name, model, spec.scale, spec.y_offset, spec.facing_radians);
            }
        }



        fn build_quad_descriptor(vertex_fn: &metal::FunctionRef, fragment_fn: &metal::FunctionRef, sample_count: u64) -> RenderPipelineDescriptor {
            let descriptor = RenderPipelineDescriptor::new();
            descriptor.set_vertex_function(Some(vertex_fn));
            descriptor.set_fragment_function(Some(fragment_fn));
            let color_attachment = descriptor.color_attachments().object_at(0).unwrap();
            color_attachment.set_pixel_format(MTLPixelFormat::BGRA8Unorm);

            color_attachment.set_blending_enabled(true);
            color_attachment.set_rgb_blend_operation(metal::MTLBlendOperation::Add);
            color_attachment.set_alpha_blend_operation(metal::MTLBlendOperation::Add);
            color_attachment.set_source_rgb_blend_factor(metal::MTLBlendFactor::SourceAlpha);
            color_attachment.set_source_alpha_blend_factor(metal::MTLBlendFactor::SourceAlpha);
            color_attachment.set_destination_rgb_blend_factor(metal::MTLBlendFactor::OneMinusSourceAlpha);
            color_attachment.set_destination_alpha_blend_factor(metal::MTLBlendFactor::OneMinusSourceAlpha);

            descriptor.set_depth_attachment_pixel_format(MTLPixelFormat::Depth32Float);
            descriptor.set_raster_sample_count(sample_count);
            descriptor
        }

        let (line_pipeline, line_pipeline_no_msaa) =
            match device.new_library_with_source(LINE_SHADER_SOURCE, &compile_options) {
                Ok(library) => {
                    let vertex_fn = library.get_function("line_vertex", None).unwrap();
                    let fragment_fn = library.get_function("line_fragment", None).unwrap();
                    let build = |sample_count: u64| {
                        let descriptor = build_quad_descriptor(&vertex_fn, &fragment_fn, sample_count);
                        descriptor.set_depth_attachment_pixel_format(MTLPixelFormat::Depth32Float);
                        descriptor
                    };
                    (
                        device.new_render_pipeline_state(&build(MSAA_SAMPLE_COUNT)).ok(),
                        device.new_render_pipeline_state(&build(1)).ok(),
                    )
                }
                Err(err) => {
                    eprintln!("[native_metal_renderer] line shader compile failed: {err}");
                    (None, None)
                }
            };



        let blit_pipeline = match device.new_library_with_source(BLIT_SHADER_SOURCE, &compile_options) {
            Ok(library) => {
                let vertex_fn = library.get_function("blit_vertex", None).unwrap();
                let fragment_fn = library.get_function("blit_fragment", None).unwrap();
                let descriptor = RenderPipelineDescriptor::new();
                descriptor.set_vertex_function(Some(&vertex_fn));
                descriptor.set_fragment_function(Some(&fragment_fn));
                descriptor.color_attachments().object_at(0).unwrap().set_pixel_format(MTLPixelFormat::BGRA8Unorm);
                device.new_render_pipeline_state(&descriptor).ok()
            }
            Err(err) => {
                eprintln!("[native_metal_renderer] blit shader compile failed: {err}");
                None
            }
        };

        let blit_verts = [
            BlitVertex { position: [-1.0, -1.0], uv: [0.0, 1.0] },
            BlitVertex { position: [1.0, -1.0], uv: [1.0, 1.0] },
            BlitVertex { position: [1.0, 1.0], uv: [1.0, 0.0] },
            BlitVertex { position: [-1.0, -1.0], uv: [0.0, 1.0] },
            BlitVertex { position: [1.0, 1.0], uv: [1.0, 0.0] },
            BlitVertex { position: [-1.0, 1.0], uv: [0.0, 0.0] },
        ];
        let blit_vertex_buffer = device.new_buffer_with_data(
            blit_verts.as_ptr() as *const _,
            (blit_verts.len() * std::mem::size_of::<BlitVertex>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let scene_color_placeholder_descriptor = metal::TextureDescriptor::new();
        scene_color_placeholder_descriptor.set_pixel_format(MTLPixelFormat::BGRA8Unorm);
        scene_color_placeholder_descriptor.set_width(1);
        scene_color_placeholder_descriptor.set_height(1);
        scene_color_placeholder_descriptor.set_storage_mode(metal::MTLStorageMode::Private);
        scene_color_placeholder_descriptor.set_usage(metal::MTLTextureUsage::RenderTarget | metal::MTLTextureUsage::ShaderRead);
        let scene_color_texture = device.new_texture(&scene_color_placeholder_descriptor);

        fn build_ui_descriptor(vertex_fn: &metal::FunctionRef, fragment_fn: &metal::FunctionRef, sample_count: u64) -> RenderPipelineDescriptor {
            let descriptor = RenderPipelineDescriptor::new();
            descriptor.set_vertex_function(Some(vertex_fn));
            descriptor.set_fragment_function(Some(fragment_fn));
            let color_attachment = descriptor.color_attachments().object_at(0).unwrap();
            color_attachment.set_pixel_format(MTLPixelFormat::BGRA8Unorm);

            color_attachment.set_blending_enabled(true);
            color_attachment.set_rgb_blend_operation(metal::MTLBlendOperation::Add);
            color_attachment.set_alpha_blend_operation(metal::MTLBlendOperation::Add);
            color_attachment.set_source_rgb_blend_factor(metal::MTLBlendFactor::SourceAlpha);
            color_attachment.set_source_alpha_blend_factor(metal::MTLBlendFactor::SourceAlpha);
            color_attachment.set_destination_rgb_blend_factor(metal::MTLBlendFactor::OneMinusSourceAlpha);
            color_attachment.set_destination_alpha_blend_factor(metal::MTLBlendFactor::OneMinusSourceAlpha);

            descriptor.set_depth_attachment_pixel_format(MTLPixelFormat::Depth32Float);
            descriptor.set_raster_sample_count(sample_count);
            descriptor
        }

        let (ui_pipeline, ui_pipeline_no_msaa) =
            match device.new_library_with_source(UI_SHADER_SOURCE, &compile_options) {
                Ok(library) => {
                    let vertex_fn = library.get_function("ui_vertex", None).unwrap();
                    let fragment_fn = library.get_function("ui_fragment", None).unwrap();
                    let msaa_descriptor = build_ui_descriptor(&vertex_fn, &fragment_fn, MSAA_SAMPLE_COUNT);
                    let no_msaa_descriptor = build_ui_descriptor(&vertex_fn, &fragment_fn, 1);
                    (
                        device.new_render_pipeline_state(&msaa_descriptor).ok(),
                        device.new_render_pipeline_state(&no_msaa_descriptor).ok(),
                    )
                }
                Err(err) => {
                    eprintln!("[native_metal_renderer] ui shader compile failed: {err}");
                    (None, None)
                }
            };

        let ui_sampler_descriptor = metal::SamplerDescriptor::new();
        ui_sampler_descriptor.set_min_filter(metal::MTLSamplerMinMagFilter::Nearest);
        ui_sampler_descriptor.set_mag_filter(metal::MTLSamplerMinMagFilter::Nearest);
        ui_sampler_descriptor.set_address_mode_s(metal::MTLSamplerAddressMode::ClampToEdge);
        ui_sampler_descriptor.set_address_mode_t(metal::MTLSamplerAddressMode::ClampToEdge);
        let ui_sampler = device.new_sampler(&ui_sampler_descriptor);
        let font_atlas_texture = build_font_atlas(&device);
        let joint_ring = JointRing::new(&device, 1 << 20);

        Self {
            device,
            layer,
            queue,
            sponza_pipeline,
            sponza_pipeline_no_msaa,
            sponza_blend_pipeline,
            sponza_blend_pipeline_no_msaa,
            sponza_pipeline_no_shadows,
            sponza_pipeline_no_shadows_no_msaa,
            sponza_pipeline_flat_normal,
            sponza_pipeline_flat_normal_no_msaa,
            sponza_pipeline_no_shadows_flat_normal,
            sponza_pipeline_no_shadows_flat_normal_no_msaa,
            sponza_sampler,
            shadow_map,
            shadow_map_point,
            shadow_pipeline_sponza,
            shadow_pipeline_skinned,
            shadow_depth_state,
            shadow_sampler,
            white_texture,
            flat_normal_texture,
            depth_state,
            blend_depth_state,
            ui_depth_state,
            msaa_targets: super::metal_msaa::MsaaTargets::new(MSAA_SAMPLE_COUNT),
            msaa_enabled: false,
            shadow_quality: 5,
            render_resolution_scale: 1.0,
            scene_color_size: (0, 0),
            scene_color_texture,
            blit_pipeline,
            blit_vertex_buffer,
            camera_position: [0.0, 2.5, 2.0],
            camera_yaw: 0.0,
            camera_pitch: -0.25,
            skinned_pipeline,
            skinned_pipeline_no_msaa,
            skinned_models,
            debug_lines: Vec::new(),
            line_pipeline,
            line_pipeline_no_msaa,
            start_time: std::time::Instant::now(),

            real_game_textures: std::collections::HashMap::new(),
            retained: std::collections::VecDeque::new(),
            retained_bytes: 0,
            gpu_transformed_mesh_cache: std::collections::HashMap::new(),
            real_game_draws: Vec::new(),
            mesh_refs: std::collections::HashMap::new(),
            texture_refs: std::collections::HashMap::new(),
            blend_slots: std::collections::BTreeSet::new(),
            opaque_order: Vec::new(),
            opaque_in_order: Vec::new(),
            opaque_pending: 0,
            real_game_camera_rotation: [0.0, 0.0, 0.0, 1.0],
            real_game_ads_blend: 0.0,
            real_game_wall_raise: 0.0,
            screen_capture: crate::screen_capture::ScreenCapture::from_env(),
            joint_ring,
            last_main_view_projection: std::cell::Cell::new([[0.0; 4]; 4]),
            lightmap: None,
            probes: None,
            probe_origin_spacing: [0.0; 4],
            probe_dims: [1.0; 4],
            probe_sampler: None,
            real_game_fov: std::f32::consts::FRAC_PI_4,
            real_game_light: FALLBACK_LIGHT,
            ui_pipeline,
            ui_pipeline_no_msaa,
            ui_sampler,
            font_atlas_texture,
            ui_generic_nodes: Vec::new(),
            force_offscreen: false,
            // FORCE_OPAQUE=1 draws blended meshes as opaque: same geometry, same
            // draw count, no blending, depth writes on. The difference against a
            // normal run is what transparency itself costs.
            force_opaque: std::env::var("FORCE_OPAQUE").is_ok(),
            stats_enabled: std::env::var("NATIVE_STATS").is_ok(),
            last_command_buffers: Vec::new(),
            shadow_quality_requested: 0,
            shadow_static: true,
            shadow_dynamic: true,
            frame_capture,
        }
    }
    /// Place the camera outright, for a scripted path. `update_camera` integrates input and
    /// cannot express "be exactly here, looking exactly there" -- which a repeatable
    /// measurement needs, since the same path has to be flown every run.
    pub fn set_camera(&mut self, position: [f32; 3], yaw: f32, pitch: f32) {
        self.camera_position = position;
        self.camera_yaw = yaw;
        self.camera_pitch = pitch;
    }

    /// Yaw and pitch that aim the camera from `eye` at `target`, in this renderer's convention:
    /// forward is `(-sin(yaw)cos(pitch), sin(pitch), -cos(yaw)cos(pitch))`, so the two negations
    /// below are the same ones that appear there.
    pub fn aim_angles(eye: [f32; 3], target: [f32; 3]) -> (f32, f32) {
        let d = [target[0] - eye[0], target[1] - eye[1], target[2] - eye[2]];
        let length = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt().max(1e-6);
        let d = [d[0] / length, d[1] / length, d[2] / length];
        (( -d[0]).atan2(-d[2]), d[1].clamp(-1.0, 1.0).asin())
    }


    pub fn update_camera(&mut self, forward: f32, strafe: f32, look_dx: f32, look_dy: f32, dt: f32) {
        self.camera_yaw -= look_dx * 0.005;
        self.camera_pitch = (self.camera_pitch - look_dy * 0.005).clamp(-1.54, 1.54);

        if forward != 0.0 || strafe != 0.0 {
            let (sin_yaw, cos_yaw) = self.camera_yaw.sin_cos();
            let forward_dir = [-sin_yaw, 0.0, -cos_yaw];
            let right_dir = [cos_yaw, 0.0, -sin_yaw];
            let speed = 6.0 * dt;
            self.camera_position[0] += (forward_dir[0] * forward + right_dir[0] * strafe) * speed;
            self.camera_position[1] += (forward_dir[1] * forward + right_dir[1] * strafe) * speed;
            self.camera_position[2] += (forward_dir[2] * forward + right_dir[2] * strafe) * speed;
        }
    }

    fn sync_drawable_size(&self) {
        unsafe {
            let raw_layer = self.layer.as_ptr() as *mut Object;
            let bounds: CGRect = msg_send![raw_layer, bounds];
            let scale = self.layer.contents_scale();
            let target = CGSize::new(bounds.size.width * scale, bounds.size.height * scale);
            let current = self.layer.drawable_size();
            if (current.width - target.width).abs() > 0.5 || (current.height - target.height).abs() > 0.5 {
                self.layer.set_drawable_size(target);
            }
        }
    }

    fn view_size_points(&self) -> (f32, f32) {
        unsafe {
            let raw_layer = self.layer.as_ptr() as *mut Object;
            let bounds: CGRect = msg_send![raw_layer, bounds];
            (bounds.size.width as f32, bounds.size.height as f32)
        }
    }

    fn opaque_draw_order(draws: &[Option<RealGameMeshDraw>]) -> Vec<usize> {
        let mut groups: std::collections::HashMap<
            (Option<AssetId<Image>>, Option<AssetId<Image>>),
            u32,
        > = std::collections::HashMap::new();
        let mut ordered: Vec<((bool, u32), usize)> = Vec::with_capacity(draws.len());
        for (slot, draw) in draws.iter().enumerate() {
            let Some(m) = draw.as_ref() else { continue };
            if m.is_blend {
                continue;
            }
            let next = groups.len() as u32;
            let group = *groups.entry((m.texture_id, m.normal_texture_id)).or_insert(next);
            // Normal-map-ness first: it picks the pipeline, and a pipeline
            // switch costs more than a texture bind.
            ordered.push(((m.normal_texture_id.is_some(), group), slot));
        }
        ordered.sort_unstable();
        ordered.into_iter().map(|(_, slot)| slot).collect()
    }

    /// Computed once per actor per frame: the sun shadow, the point shadow and
    /// the scene pass all draw the same skeletons, and each pushing its own
    /// copy filled the joint ring three times over.
    fn prepare_actor(
        &self,
        actor: &super::skinned_actor::SkinnedActorDraw,
    ) -> Option<PreparedActor> {
        let slot = self.skinned_models.get(actor.model)?;
        let (model, mesh_scale, y_offset) = (&slot.model, slot.scale, slot.y_offset);
        let (model_matrix, joint_matrices): (_, std::borrow::Cow<[[[f32; 4]; 4]]>) = match &actor.pose {
            super::skinned_actor::SkinnedActorPose::JointMatrices(matrices) => {
                ([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]], std::borrow::Cow::Borrowed(matrices.as_slice()))
            }
            super::skinned_actor::SkinnedActorPose::Clip { clip_index, time } => {
                let rotation = if slot.facing_correction != 0.0 {
                    actor.transform.rotation * bevy::math::Quat::from_rotation_y(slot.facing_correction)
                } else {
                    actor.transform.rotation
                };
                let [qx, qy, qz, qw] = rotation.to_array();
                let (x2, y2, z2) = (qx + qx, qy + qy, qz + qz);
                let (xx, xy, xz) = (qx * x2, qx * y2, qx * z2);
                let (yy, yz, zz) = (qy * y2, qy * z2, qz * z2);
                let (wx, wy, wz) = (qw * x2, qw * y2, qw * z2);
                let matrix = [
                    [(1.0 - (yy + zz)) * mesh_scale, (xy + wz) * mesh_scale, (xz - wy) * mesh_scale, 0.0],
                    [(xy - wz) * mesh_scale, (1.0 - (xx + zz)) * mesh_scale, (yz + wx) * mesh_scale, 0.0],
                    [(xz + wy) * mesh_scale, (yz - wx) * mesh_scale, (1.0 - (xx + yy)) * mesh_scale, 0.0],
                    [actor.transform.translation.x, actor.transform.translation.y + y_offset, actor.transform.translation.z, 1.0],
                ];
                (matrix, std::borrow::Cow::Owned(model.compute_joint_matrices(*clip_index, *time)))
            }
        };

        if joint_matrices.is_empty() {
            return None;
        }
        let (joint_buffer, joint_offset) = match self.joint_ring.push(&joint_matrices) {
            Some((buffer, offset)) => (buffer.to_owned(), offset),
            None => (
                self.device.new_buffer_with_data(
                    joint_matrices.as_ptr() as *const _,
                    std::mem::size_of_val(&joint_matrices[..]) as u64,
                    MTLResourceOptions::StorageModeShared,
                ),
                0,
            ),
        };
        Some(PreparedActor { model_matrix, joint_buffer, joint_offset })
    }

    fn draw_scene(
        &self,
        encoder: &metal::RenderCommandEncoderRef,
        view_projection: [[f32; 4]; 4],
        light: LightUniform,

        extracted: &[super::skinned_actor::SkinnedActorDraw],
        prepared: &[Option<PreparedActor>],
        _now: f32,
        real_game_meshes: Option<&[Option<RealGameMeshDraw>]>,

        msaa_enabled: bool,

        light_view_projection: [[f32; 4]; 4],
        point_light_view_projection: [[f32; 4]; 4],
        camera_pos: [f32; 3],
        // The scope pass leaves the weapon out; every other pass leaves the
        // reticle out. Both are render layers the windowed build has and the
        // snapshot had no way to express until now.
    ) {
        let shadows_on = self.shadow_quality_requested > 0;
        let sponza_pipeline = match (msaa_enabled, shadows_on) {
            (true, true) => self.sponza_pipeline.as_ref(),
            (false, true) => self.sponza_pipeline_no_msaa.as_ref(),
            (true, false) => self.sponza_pipeline_no_shadows.as_ref(),
            (false, false) => self.sponza_pipeline_no_shadows_no_msaa.as_ref(),
        };
        let sponza_pipeline_flat_normal = match (msaa_enabled, shadows_on) {
            (true, true) => self.sponza_pipeline_flat_normal.as_ref(),
            (false, true) => self.sponza_pipeline_flat_normal_no_msaa.as_ref(),
            (true, false) => self.sponza_pipeline_no_shadows_flat_normal.as_ref(),
            (false, false) => self.sponza_pipeline_no_shadows_flat_normal_no_msaa.as_ref(),
        }
        .or(sponza_pipeline);
        let sponza_blend_pipeline = if msaa_enabled {
            self.sponza_blend_pipeline.as_ref()
        } else {
            self.sponza_blend_pipeline_no_msaa.as_ref()
        };
        let skinned_pipeline = if msaa_enabled { self.skinned_pipeline.as_ref() } else { self.skinned_pipeline_no_msaa.as_ref() };
        encoder.set_fragment_texture(1, Some(&self.shadow_map));
        encoder.set_fragment_sampler_state(1, Some(&self.shadow_sampler));
        encoder.set_fragment_texture(2, Some(&self.shadow_map_point));

        let pass_uniforms = SponzaPassUniforms {
            view_projection,
            light_pos: light.position,
            light_range: light.range,
            light_color: light.color,
            sun_dir: light.sun_dir,
            _sun_pad: 0.0,
            sun_color: light.sun_color,
            light_view_projection,
            point_light_view_projection,
            camera_pos,
            _camera_pad: 0.0,
        };
        // The skinned pass binds its own buffer(1), texture(3) and texture(4)
        // between the opaque and blended loops, so both sponza loops bind
        // these rather than the pass setting them once.
        let bind_sponza_pass_state = || {
            encoder.set_vertex_bytes(
                1,
                std::mem::size_of::<SponzaPassUniforms>() as u64,
                &pass_uniforms as *const _ as *const _,
            );
            encoder.set_fragment_bytes(
                1,
                std::mem::size_of::<SponzaPassUniforms>() as u64,
                &pass_uniforms as *const _ as *const _,
            );
            encoder.set_fragment_sampler_state(0, Some(&self.sponza_sampler));
            encoder.set_fragment_texture(4, Some(self.lightmap.as_deref().unwrap_or(&self.white_texture)));
        };

        // The most repeated encoder calls in the pass, against a map whose
        // 2363 primitives share 13 images. Compared by pointer so the
        // white/flat fallbacks collapse into the same binding as each other.
        let no_bind_guard = env_off("NO_TEXTURE_BIND_GUARD");
        let bound_texture = std::cell::Cell::new(std::ptr::null_mut());
        let bound_normal_texture = std::cell::Cell::new(std::ptr::null_mut());
        let forget_bound_textures = || {
            bound_texture.set(std::ptr::null_mut());
            bound_normal_texture.set(std::ptr::null_mut());
        };

        let draw_static_mesh = |m: &RealGameMeshDraw| {
            let mesh_uniforms = SponzaMeshUniforms {
                base_color_factor: m.base_color_factor,
                emissive: m.emissive,
                model_matrix: m.model_matrix,
                normal_matrix: normal_matrix_for(m.model_matrix),
            };
            encoder.set_vertex_buffer(0, Some(&m.vertex_buffer), 0);
            encoder.set_vertex_bytes(
                2,
                std::mem::size_of::<SponzaMeshUniforms>() as u64,
                &mesh_uniforms as *const _ as *const _,
            );
            encoder.set_fragment_bytes(
                2,
                std::mem::size_of::<SponzaMeshUniforms>() as u64,
                &mesh_uniforms as *const _ as *const _,
            );
            let texture = m.texture.as_ref().unwrap_or(&self.white_texture);
            if no_bind_guard || bound_texture.get() != texture.as_ptr() {
                encoder.set_fragment_texture(0, Some(texture));
                bound_texture.set(texture.as_ptr());
            }
            let normal_texture = m.normal_texture.as_ref().unwrap_or(&self.flat_normal_texture);
            if no_bind_guard || bound_normal_texture.get() != normal_texture.as_ptr() {
                encoder.set_fragment_texture(3, Some(normal_texture));
                bound_normal_texture.set(normal_texture.as_ptr());
            }
            encoder.draw_indexed_primitives(
                MTLPrimitiveType::Triangle,
                m.index_count as u64,
                if m.narrow_index { MTLIndexType::UInt16 } else { MTLIndexType::UInt32 },
                &m.index_buffer,
                0,
            );
        };

        // Once per pass, not per mesh.
        let planes = frustum_planes(view_projection);
        self.last_main_view_projection.set(view_projection);

        if let Some(meshes) = real_game_meshes {

            let mut bound_flat_normal: Option<bool> = None;
            let no_flat_pipeline = env_off("NO_FLAT_NORMAL_PIPELINE");
            // The lightmap falls back to the white texture so the sample is
            // defined even before the host has handed one over -- the shader
            // skips it on uv2 being zero, but an unbound slot is undefined
            // behaviour rather than a defined white.
            bind_sponza_pass_state();
            // opaque_order, not slot order: it groups the draws by texture so
            // the binds above it can be skipped.
            let slot_order: Vec<usize> = (0..meshes.len()).collect();
            let order: &[usize] =
                if env_off("NO_OPAQUE_SORT") { &slot_order } else { &self.opaque_order };
            for m in order
                .iter()
                .filter_map(|slot| meshes.get(*slot).and_then(|m| m.as_ref()))
                .filter(|m| !m.is_blend && draws_in_view(m, &planes))
            {
                let flat = m.normal_texture_id.is_none();
                let flat = flat && !no_flat_pipeline;
                if bound_flat_normal != Some(flat) {
                    let picked = if flat { sponza_pipeline_flat_normal } else { sponza_pipeline };
                    encoder.set_render_pipeline_state(picked.unwrap());
                    bound_flat_normal = Some(flat);
                }
                draw_static_mesh(m);
            }
        }

        let Some(pipeline) = skinned_pipeline else { return };
        encoder.set_render_pipeline_state(pipeline);
        encoder.set_fragment_sampler_state(0, Some(&self.sponza_sampler));
        // Bound once for the pass. The white texture stands in before a grid
        // has been handed over -- the shader skips it on spacing being zero,
        // but an unbound slot is undefined rather than defined white.
        if let (Some((positive, negative)), Some(sampler)) =
            (self.probes.as_ref(), self.probe_sampler.as_ref())
        {
            encoder.set_fragment_texture(3, Some(positive));
            encoder.set_fragment_texture(4, Some(negative));
            encoder.set_fragment_sampler_state(2, Some(sampler));
        }

        let skinned_pass_uniforms = SkinnedPassUniforms {
            view_projection,
            light_pos: light.position,
            light_range: light.range,
            light_color: light.color,
            sun_dir: light.sun_dir,
            _sun_pad: 0.0,
            sun_color: light.sun_color,
            light_view_projection,
            point_light_view_projection,
            camera_pos,
            _camera_pad: 0.0,
            probe_origin_spacing: self.probe_origin_spacing,
            probe_dims: self.probe_dims,
        };
        encoder.set_vertex_bytes(
            1,
            std::mem::size_of::<SkinnedPassUniforms>() as u64,
            &skinned_pass_uniforms as *const _ as *const _,
        );
        encoder.set_fragment_bytes(
            1,
            std::mem::size_of::<SkinnedPassUniforms>() as u64,
            &skinned_pass_uniforms as *const _ as *const _,
        );

        for (actor, prepared) in extracted.iter().zip(prepared) {
            if !actor_in_view(actor, &planes) {
                continue;
            }
            let (Some(slot), Some(prepared)) =
                (self.skinned_models.get(actor.model), prepared.as_ref())
            else {
                continue;
            };
            let model = &slot.model;
            let model_matrix = prepared.model_matrix;
            let (joint_buffer, joint_offset) =
                (prepared.joint_buffer.as_ref(), prepared.joint_offset);


            for mesh in &model.meshes {
                let skinned_mesh_uniforms = SkinnedMeshUniforms {
                    model_matrix,
                    base_color_factor: mesh.base_color_factor,
                };
                encoder.set_vertex_buffer(0, Some(&mesh.vertex_buffer), 0);
                encoder.set_vertex_buffer(2, Some(joint_buffer), joint_offset);
                encoder.set_vertex_bytes(
                    3,
                    std::mem::size_of::<SkinnedMeshUniforms>() as u64,
                    &skinned_mesh_uniforms as *const _ as *const _,
                );
                encoder.set_fragment_bytes(
                    3,
                    std::mem::size_of::<SkinnedMeshUniforms>() as u64,
                    &skinned_mesh_uniforms as *const _ as *const _,
                );
                let texture = mesh.base_color_texture.as_ref().unwrap_or(&self.white_texture);
                encoder.set_fragment_texture(0, Some(texture));
                encoder.draw_indexed_primitives(
                    MTLPrimitiveType::Triangle,
                    mesh.index_count as u64,
                    MTLIndexType::UInt32,
                    &mesh.index_buffer,
                    0,
                );
            }
        }

        if let Some(meshes) = real_game_meshes {

            // Measured here rather than carried on the draw: the camera moves
            // every frame, so a stored distance would be the one field in the
            // list that had to be rewritten for every mesh every frame --
            // which is the cost this list exists to avoid. Only the handful
            // that survive the blend filter need it.
            let dist_sq = |m: &RealGameMeshDraw| {
                let d = [
                    m.bounds_centre[0] - camera_pos[0],
                    m.bounds_centre[1] - camera_pos[1],
                    m.bounds_centre[2] - camera_pos[2],
                ];
                d[0] * d[0] + d[1] * d[1] + d[2] * d[2]
            };
            let mut blended: Vec<&RealGameMeshDraw> = self
                .blend_slots
                .iter()
                .filter_map(|slot| meshes.get(*slot).and_then(|m| m.as_ref()))
                .filter(|m| draws_in_view(m, &planes))
                .collect();
            blended.sort_by(|a, b| dist_sq(b).total_cmp(&dist_sq(a)));
            if !blended.is_empty() {
                encoder.set_depth_stencil_state(&self.blend_depth_state);
                encoder.set_render_pipeline_state(sponza_blend_pipeline.unwrap_or(sponza_pipeline.unwrap()));
                bind_sponza_pass_state();
                forget_bound_textures();
                for m in blended {
                    draw_static_mesh(m);
                }

                encoder.set_depth_stencil_state(&self.depth_state);
            }
        }

        self.draw_debug_lines(encoder, view_projection, msaa_enabled);
    }

    fn draw_debug_lines(
        &self,
        encoder: &metal::RenderCommandEncoderRef,
        view_projection: [[f32; 4]; 4],
        msaa_enabled: bool,
    ) {
        if self.debug_lines.is_empty() {
            return;
        }
        let pipeline = if msaa_enabled { &self.line_pipeline } else { &self.line_pipeline_no_msaa };
        let Some(pipeline) = pipeline else { return };

        let verts: Vec<LineVertex> = self
            .debug_lines
            .iter()
            .flat_map(|line| {
                [
                    LineVertex { position: line.a, color: line.color },
                    LineVertex { position: line.b, color: line.color },
                ]
            })
            .collect();
        let buffer = self.device.new_buffer_with_data(
            verts.as_ptr() as *const _,
            (verts.len() * std::mem::size_of::<LineVertex>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let uniforms = LineUniforms { view_projection };

        encoder.set_render_pipeline_state(pipeline);
        encoder.set_depth_stencil_state(&self.depth_state);
        encoder.set_vertex_buffer(0, Some(&buffer), 0);
        encoder.set_vertex_bytes(1, std::mem::size_of::<LineUniforms>() as u64, &uniforms as *const _ as *const _);
        encoder.draw_primitives(MTLPrimitiveType::Line, 0, verts.len() as u64);
    }

    fn make_shadow_texture(device: &metal::Device, size: u64) -> metal::Texture {
        let descriptor = metal::TextureDescriptor::new();
        descriptor.set_pixel_format(MTLPixelFormat::Depth32Float);
        descriptor.set_width(size);
        descriptor.set_height(size);
        descriptor.set_storage_mode(metal::MTLStorageMode::Private);
        descriptor.set_usage(metal::MTLTextureUsage::RenderTarget | metal::MTLTextureUsage::ShaderRead);
        device.new_texture(&descriptor)
    }

    fn ensure_shadow_map_size(&mut self, quality: u8) {
        let quality = quality.min(5);
        if quality == self.shadow_quality {
            return;
        }
        // The fragment shader samples these twice per pixel whatever the
        // quality setting says, so their size is a real per-pixel cost and a
        // host measuring that cost needs to be able to pin it.
        let size = SHADOW_MAP_SIZES[quality as usize];
        println!("[config] shadow map {size}px (quality {quality})");
        self.shadow_map = Self::make_shadow_texture(&self.device, size);
        self.shadow_map_point = Self::make_shadow_texture(&self.device, size);
        self.shadow_quality = quality;
    }

    fn ensure_scene_color_texture(&mut self, width: u64, height: u64) -> metal::Texture {
        if (width, height) == self.scene_color_size {
            return self.scene_color_texture.clone();
        }
        let descriptor = metal::TextureDescriptor::new();
        descriptor.set_pixel_format(MTLPixelFormat::BGRA8Unorm);
        descriptor.set_width(width);
        descriptor.set_height(height);
        descriptor.set_storage_mode(metal::MTLStorageMode::Private);
        descriptor.set_usage(metal::MTLTextureUsage::RenderTarget | metal::MTLTextureUsage::ShaderRead);
        let tex = self.device.new_texture(&descriptor);
        self.scene_color_texture = tex.clone();
        self.scene_color_size = (width, height);
        tex
    }


    fn draw_shadow_casters(
        &self,
        encoder: &metal::RenderCommandEncoderRef,
        light_view_projection: [[f32; 4]; 4],
        extracted: &[super::skinned_actor::SkinnedActorDraw],
        prepared: &[Option<PreparedActor>],
        real_game_meshes: Option<&[Option<RealGameMeshDraw>]>,
    ) {
        if let Some(meshes) = real_game_meshes.filter(|_| self.shadow_static) {
            if let Some(pipeline) = self.shadow_pipeline_sponza.as_ref() {
                encoder.set_render_pipeline_state(pipeline);
                // The light's frustum, not the camera's: a wall behind the
                // player still casts into the map in front of them.
                let planes = frustum_planes(light_view_projection);
                for m in meshes.iter().flatten().filter(|m| {
                    !m.viewmodel
                        && !sphere_outside(&planes, m.bounds_centre, m.bounds_radius)
                }) {
                    let uniforms = SponzaShadowUniforms { light_view_projection, model_matrix: m.model_matrix };
                    encoder.set_vertex_buffer(0, Some(&m.vertex_buffer), 0);
                    encoder.set_vertex_bytes(1, std::mem::size_of::<SponzaShadowUniforms>() as u64, &uniforms as *const _ as *const _);
                    encoder.draw_indexed_primitives(
                MTLPrimitiveType::Triangle,
                m.index_count as u64,
                if m.narrow_index { MTLIndexType::UInt16 } else { MTLIndexType::UInt32 },
                &m.index_buffer,
                0,
            );
                }
            }
        }

        if !self.shadow_dynamic {
            return;
        }
        let Some(pipeline) = self.shadow_pipeline_skinned.as_ref() else { return };
        encoder.set_render_pipeline_state(pipeline);
        // The light's frustum, for the same reason the static casters use it.
        let planes = frustum_planes(light_view_projection);
        for (actor, prepared) in extracted.iter().zip(prepared) {
            if !actor_in_view(actor, &planes) {
                continue;
            }
            let (Some(slot), Some(prepared)) =
                (self.skinned_models.get(actor.model), prepared.as_ref())
            else {
                continue;
            };
            let model = &slot.model;
            let model_matrix = prepared.model_matrix;
            let (joint_buffer, joint_offset) =
                (prepared.joint_buffer.as_ref(), prepared.joint_offset);
            for mesh in &model.meshes {
                let uniforms = SkinnedShadowUniforms { light_view_projection, model_matrix };
                encoder.set_vertex_buffer(0, Some(&mesh.vertex_buffer), 0);
                encoder.set_vertex_buffer(2, Some(joint_buffer), joint_offset);
                encoder.set_vertex_bytes(1, std::mem::size_of::<SkinnedShadowUniforms>() as u64, &uniforms as *const _ as *const _);
                encoder.draw_indexed_primitives(
                    MTLPrimitiveType::Triangle,
                    mesh.index_count as u64,
                    MTLIndexType::UInt32,
                    &mesh.index_buffer,
                    0,
                );
            }
        }
    }

    /// True once this mesh's vertices live in a GPU buffer, so the host can
    /// skip re-extracting geometry it already uploaded.
    pub fn has_uploaded_mesh(&self, id: AssetId<Mesh>) -> bool {
        self.gpu_transformed_mesh_cache.contains_key(&id)
    }

    /// Lets the host skip copying bytes out of an image this already holds.
    pub fn has_uploaded_texture(&self, id: AssetId<Image>) -> bool {
        self.real_game_textures.contains_key(&id)
    }

    /// Points, not pixels — what a host needs to size its UI camera viewport.
    pub fn view_size(&self) -> (f32, f32) {
        self.view_size_points()
    }

    pub fn contents_scale(&self) -> f32 {
        self.layer.contents_scale() as f32
    }

    pub fn render_frame(&mut self, snapshot: RenderSnapshot) -> FrameTimings {
        self.joint_ring.begin_frame();
        let mut timings = FrameTimings::default();
        self.sync_drawable_size();
        let drawable_size = self.layer.drawable_size();
        let (width, height) = (drawable_size.width as u64, drawable_size.height as u64);

        let is_real_game = true;

        let now = self.start_time.elapsed().as_secs_f32();

        let mesh_updates: Vec<(usize, super::snapshot::ExtractedMesh)>;
        let mesh_slots: usize;

        let mut real_game_extracted: Vec<super::skinned_actor::SkinnedActorDraw> = Vec::new();
        {
            self.ui_generic_nodes = snapshot.generic_ui_nodes;
            self.debug_lines = snapshot.debug_lines;
            self.msaa_enabled = snapshot.map_msaa_on;
            self.ensure_shadow_map_size(snapshot.shadow_quality);
            self.shadow_quality_requested = snapshot.shadow_quality;
            self.shadow_static = snapshot.shadow_static;
            self.shadow_dynamic = snapshot.shadow_dynamic;
            self.render_resolution_scale = snapshot.render_resolution_scale.clamp(0.1, 1.0);
            if let Some(cam) = &snapshot.camera {
                self.camera_position = cam.eye;
                self.real_game_fov = cam.fov;
                self.camera_yaw = cam.yaw;
                self.camera_pitch = cam.pitch;
                self.real_game_camera_rotation = cam.rotation;
                self.real_game_ads_blend = cam.ads_blend;
                self.real_game_wall_raise = cam.wall_raise;
            }
            self.real_game_light = LightUniform::from_snapshots(
                snapshot.light.as_ref(),
                snapshot.sun.as_ref(),
                // On or off, not a dimmer. The windowed build scales
                // illuminance by the same fraction, but that is lux fed to a
                // PBR pipeline with exposure, which normalises most of it
                // away. Here the fraction lands directly on the final colour,
                // so quality 3 of 5 rendered the whole scene at 74% of its
                // albedo -- visibly darker than the same settings in the
                // windowed build.
                if snapshot.light_quality > 0 { 1.0 } else { 0.0 },
                snapshot.point_light_quality.min(5) as f32 / 5.0,
            );
            mesh_updates = snapshot.static_mesh_updates;
            mesh_slots = snapshot.static_mesh_slots;

            real_game_extracted.extend(snapshot.skinned_actors.into_iter().map(|actor| {
                super::skinned_actor::SkinnedActorDraw {
                    transform: actor.transform,
                    model: actor.model,
                    pose: super::skinned_actor::SkinnedActorPose::JointMatrices(actor.joint_matrices),
                    bounds_centre: actor.bounds_centre,
                    bounds_radius: actor.bounds_radius,
                }
            }));
        }

        let msaa_enabled = self.msaa_enabled;
        let device = self.device.clone();

        let render_width = ((width as f64 * self.render_resolution_scale as f64) as u64).max(RENDER_SCALE_MIN_PX);
        let render_height = ((height as f64 * self.render_resolution_scale as f64) as u64).max(RENDER_SCALE_MIN_PX);
        let depth_texture = if msaa_enabled {
            self.msaa_targets.ensure_depth(&device, render_width, render_height)
        } else {
            self.msaa_targets.ensure_depth_no_msaa(&device, render_width, render_height)
        };
        let msaa_color_texture = msaa_enabled.then(|| self.msaa_targets.ensure_color(&device, render_width, render_height));

        // At native resolution with no multisampling there is nothing for the
        // offscreen target to buy, and skipping it saves a full-screen
        // read-write of the colour buffer.
        let skip_blit = !self.force_offscreen && !msaa_enabled && render_width == width && render_height == height;
        timings.render_size = (render_width, render_height);
        timings.skipped_blit = skip_blit;
        let offscreen = (!skip_blit).then(|| self.ensure_scene_color_texture(render_width, render_height));
        // next_drawable blocks until the display hands one back, so this is
        // where "the GPU or the refresh rate is the limit" shows up — keep it
        // out of the encode numbers.
        let wait_start = std::time::Instant::now();
        let Some(drawable) = self.layer.next_drawable() else { return timings };
        timings.drawable_wait_ms = wait_start.elapsed().as_secs_f32() * 1000.0;
        // Read once the display has handed a drawable back, which means the GPU
        // has retired an earlier frame — before that the timestamps are zero.
        // This is real GPU execution time, unlike `drawable_wait_ms`, which is
        // only the CPU blocked on it.
        // Every buffer, not just the scene one: the shadow and blit passes are
        // separate submissions, and measuring only the scene pass hid their
        // cost entirely — switching the blit off looked free because its time
        // was never in the number being compared.
        if let Some(capture) = &mut self.frame_capture {
            if capture.frame_done() {
                self.frame_capture = None;
            }
        }
        timings.command_buffers = self.last_command_buffers.len();
        for (label, previous) in self.last_command_buffers.drain(..) {
            // metal 0.33 binds neither of these.
            let raw = previous.as_ptr() as *mut Object;
            let start: f64 = unsafe { msg_send![raw, GPUStartTime] };
            let end: f64 = unsafe { msg_send![raw, GPUEndTime] };
            let elapsed = ((end - start) * 1000.0) as f32;
            // Timestamps that are not yet valid report a nonsense span — one
            // sample came back as 6_547_575 ms — and a single one of those
            // ruins the average for the whole window it lands in.
            if !(0.0..1000.0).contains(&elapsed) {
                continue;
            }
            timings.gpu_total_ms += elapsed;
            if label == "scene" {
                timings.gpu_ms = elapsed;
                timings.gpu_status = previous.status() as u64;
            }
        }
        let bookkeeping_start = std::time::Instant::now();
        let drawable_texture = drawable.texture();
        let texture = offscreen.unwrap_or_else(|| drawable_texture.to_owned());

        // UI textures first and unconditionally: a HUD image appears and goes
        // without any mesh slot moving, and gating this on meshes would leave
        // it unuploaded until something else happened to change.
        for image in self.ui_generic_nodes.iter().filter_map(|node| node.image.as_ref()) {
            // An empty payload means the host skipped the copy because this was
            // already uploaded. Drawing untextured for a frame beats uploading
            // nothing as if it were the image.
            if !image.rgba.is_empty() && !self.real_game_textures.contains_key(&image.id) {
                let uploaded = match image.astc {
                    Some(block) => super::metal_texture::upload_astc(
                        &self.device, image.width, image.height, &image.rgba, block,
                        image.mip_level_count,
                    ),
                    None => super::metal_texture::upload_texture(
                        &self.device, image.width, image.height, &image.rgba,
                    ),
                };
                self.real_game_textures.insert(image.id, uploaded);
            }
        }

        // Rebuild only the slots the host reports as changed. For a map that
        // is standing still that is none of them, and the list below is the
        // one prepared last frame -- which is the whole point. The old code
        // rebuilt all 1544 draws every frame: a hash lookup for the geometry,
        // two more for the textures, a matrix and a bounding sphere, into a
        // freshly allocated Vec. desert_swift_metal builds its list once at
        // load and the per-frame cost of preparing it is zero.
        if self.real_game_draws.len() < mesh_slots {
            self.real_game_draws.resize(mesh_slots, None);
        }
        for (slot, mesh) in &mesh_updates {
            if !mesh.visible {
                self.blend_slots.remove(slot);
                if let Some(gone) = self.real_game_draws[*slot].take() {
                    release_draw_assets(
                        &mut self.mesh_refs, &mut self.texture_refs,
                        &self.gpu_transformed_mesh_cache, &self.real_game_textures,
                        &mut self.retained, &mut self.retained_bytes, gone,
                    );
                }
                continue;
            }
            let geometry = match self.gpu_transformed_mesh_cache.get(&mesh.mesh_id) {
                Some(cached) => cached.clone(),
                None => {
                    let (verts, indices) = bake_local_mesh_vertices(mesh);
                    if verts.is_empty() || indices.is_empty() {
                        bevy::log::warn!(
                            "empty mesh skipped: positions={} normals={} uvs={} indices={}",
                            mesh.positions.len(),
                            mesh.normals.len(),
                            mesh.uvs.len(),
                            mesh.indices.len()
                        );
                        self.blend_slots.remove(slot);
                        if let Some(gone) = self.real_game_draws[*slot].take() {
                            release_draw_assets(
                                &mut self.mesh_refs, &mut self.texture_refs,
                                &self.gpu_transformed_mesh_cache, &self.real_game_textures,
                                &mut self.retained, &mut self.retained_bytes, gone,
                            );
                        }
                        continue;
                    }
                    let buffer = self.device.new_buffer_with_data(
                        verts.as_ptr() as *const _,
                        (verts.len() * std::mem::size_of::<super::gltf_scene::GltfVertex>()) as u64,
                        MTLResourceOptions::StorageModeShared,
                    );
                    // Half the index bandwidth for anything that fits, which
                    // on this map is every mesh -- the largest is 8233 verts.
                    let narrow = verts.len() <= u16::MAX as usize + 1;
                    let index_buffer = if narrow {
                        let narrowed: Vec<u16> = indices.iter().map(|i| *i as u16).collect();
                        self.device.new_buffer_with_data(
                            narrowed.as_ptr() as *const _,
                            std::mem::size_of_val(narrowed.as_slice()) as u64,
                            MTLResourceOptions::StorageModeShared,
                        )
                    } else {
                        self.device.new_buffer_with_data(
                            indices.as_ptr() as *const _,
                            std::mem::size_of_val(indices.as_slice()) as u64,
                            MTLResourceOptions::StorageModeShared,
                        )
                    };
                    let (centre, radius) = local_bounds(&mesh.positions);
                    let entry =
                        (buffer, index_buffer, indices.len(), verts.len(), centre, radius, narrow);
                    self.gpu_transformed_mesh_cache.insert(mesh.mesh_id, entry.clone());
                    entry
                }
            };
            let (vertex_buffer, index_buffer, index_count, unique_vertices, local_centre, local_radius, narrow_index) =
                geometry;
            // After the bake, not before: a mesh that fails to bake builds no
            // draw, so nothing would ever stand behind textures uploaded for it
            // and nothing would ever count them back down.
            for t in [mesh.texture.as_ref(), mesh.normal_texture.as_ref()].into_iter().flatten() {
                if !t.rgba.is_empty() && !self.real_game_textures.contains_key(&t.id) {
                    let uploaded = match t.astc {
                        Some(block) => super::metal_texture::upload_astc(
                            &self.device, t.width, t.height, &t.rgba, block,
                            t.mip_level_count,
                        ),
                        None => super::metal_texture::upload_texture(
                            &self.device, t.width, t.height, &t.rgba,
                        ),
                    };
                    self.real_game_textures.insert(t.id, uploaded);
                    timings.texture_uploads += 1;
                }
            }
            let draw = RealGameMeshDraw {
                viewmodel: mesh.viewmodel,
                vertex_buffer,
                index_buffer,
                index_count,
                narrow_index,
                unique_vertices,
                base_color_factor: mesh.base_color,
                emissive: mesh.emissive,
                texture: mesh.texture.as_ref().and_then(|t| self.real_game_textures.get(&t.id).cloned()),
                normal_texture: mesh
                    .normal_texture
                    .as_ref()
                    .and_then(|t| self.real_game_textures.get(&t.id).cloned()),
                mesh_id: mesh.mesh_id,
                texture_id: mesh.texture.as_ref().map(|t| t.id),
                normal_texture_id: mesh.normal_texture.as_ref().map(|t| t.id),
                is_blend: mesh.is_blend && !self.force_opaque,
                bounds_centre: mesh.transform.transform_point(bevy::math::Vec3::from_array(local_centre)).to_array(),
                // The largest scale on any axis. A non-uniform scale makes
                // the true bound an ellipsoid; taking the largest keeps the
                // sphere conservative, which is the only direction a cull
                // is allowed to be wrong in.
                bounds_radius: local_radius * mesh.transform.scale.abs().max_element(),
                model_matrix: mesh.transform.to_matrix().to_cols_array_2d(),
            };
            // Retain before release: a slot usually keeps the same assets
            // across an update, and releasing first would take their count to
            // zero and throw away buffers the new draw is about to ask for.
            if draw.is_blend {
                self.blend_slots.insert(*slot);
            } else {
                self.blend_slots.remove(slot);
            }
            retain_draw_assets(
                &mut self.mesh_refs, &mut self.texture_refs,
                &mut self.retained, &mut self.retained_bytes, &draw,
            );
            if let Some(gone) = self.real_game_draws[*slot].replace(draw) {
                release_draw_assets(
                    &mut self.mesh_refs, &mut self.texture_refs,
                    &self.gpu_transformed_mesh_cache, &self.real_game_textures,
                    &mut self.retained, &mut self.retained_bytes, gone,
                );
            }
        }

        for (slot, _) in &mesh_updates {
            let slot = *slot;
            if self.opaque_in_order.len() <= slot {
                self.opaque_in_order.resize(slot + 1, false);
            }
            let opaque = self
                .real_game_draws
                .get(slot)
                .and_then(|m| m.as_ref())
                .is_some_and(|m| !m.is_blend);
            if opaque == self.opaque_in_order[slot] {
                continue;
            }
            self.opaque_in_order[slot] = opaque;
            self.opaque_pending += 1;
            if opaque {
                self.opaque_order.push(slot);
            } else if let Some(at) = self.opaque_order.iter().position(|s| *s == slot) {
                self.opaque_order.swap_remove(at);
            }
        }
        while self.retained_bytes > RETAINED_GPU_BUDGET_BYTES {
            let Some((asset, bytes)) = self.retained.pop_front() else { break };
            self.retained_bytes -= bytes;
            match asset {
                RetainedAsset::Mesh(id) => {
                    self.gpu_transformed_mesh_cache.remove(&id);
                }
                RetainedAsset::Texture(id) => {
                    // A HUD image lives in the same map without any mesh naming
                    // it, so no count ever stands behind it. Checked here, at
                    // the one point the texture would actually be freed.
                    let on_screen = self.ui_generic_nodes.iter().any(|node| {
                        node.image.as_ref().is_some_and(|image| image.id == id)
                    });
                    if !on_screen {
                        self.real_game_textures.remove(&id);
                    }
                }
            }
        }

        // Regrouping the whole list costs ~126us at this map's slot count, so
        // it waits until enough of the tail is out of order to be worth that,
        // rather than running on every spawned casing.
        if self.opaque_pending >= 64 {
            self.opaque_order = Self::opaque_draw_order(&self.real_game_draws);
            self.opaque_in_order.clear();
            self.opaque_in_order.resize(self.real_game_draws.len(), false);
            for slot in &self.opaque_order {
                self.opaque_in_order[*slot] = true;
            }
            self.opaque_pending = 0;
        }
        let real_game_meshes_opt =
            if self.real_game_draws.is_empty() { None } else { Some(self.real_game_draws.as_slice()) };
        timings.bookkeeping_ms = bookkeeping_start.elapsed().as_secs_f32() * 1000.0;
        timings.cached_textures = self.real_game_textures.len();
        timings.cached_meshes = self.gpu_transformed_mesh_cache.len();
        // Five walks over the whole list for numbers nobody reads unless
        // NATIVE_STATS is set. Counting the draws cost more than some of the
        // passes being counted.
        if self.stats_enabled {
            let planes = frustum_planes(self.last_main_view_projection.get());
            for m in self.real_game_draws.iter().flatten() {
                timings.draws += 1;
                if !draws_in_view(m, &planes) {
                    timings.culled += 1;
                }
                timings.vertices_drawn += m.index_count;
                timings.unique_vertices += m.unique_vertices;
                if m.is_blend {
                    timings.blend_draws += 1;
                }
            }
            // The index the blend pass now trusts, against the count this
            // walk arrives at the slow way. A slot that enters or leaves
            // blend without telling the index draws nothing, or draws twice.
            debug_assert_eq!(timings.blend_draws, self.blend_slots.len());
        }
        let passes_start = std::time::Instant::now();

        let aspect = width as f32 / height.max(1) as f32;
        let far = if is_real_game { 500.0 } else { 100.0 };

        let (look_dir, up) = if is_real_game {
            let rotation = bevy::math::Quat::from_array(self.real_game_camera_rotation);
            (
                (rotation * bevy::math::Vec3::NEG_Z).to_array(),
                (rotation * bevy::math::Vec3::Y).to_array(),
            )
        } else {
            let (sin_pitch, cos_pitch) = self.camera_pitch.sin_cos();
            let (sin_yaw, cos_yaw) = self.camera_yaw.sin_cos();
            ([-sin_yaw * cos_pitch, sin_pitch, -cos_yaw * cos_pitch], [0.0, 1.0, 0.0])
        };
        let center = [
            self.camera_position[0] + look_dir[0],
            self.camera_position[1] + look_dir[1],
            self.camera_position[2] + look_dir[2],
        ];
        let view = if is_real_game {
            look_at(self.camera_position, center, up)
        } else {
            look_at([0.0, 3.0, 10.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0])
        };
        let proj = if is_real_game {
            perspective(self.real_game_fov, aspect, 0.05, far)
        } else {
            perspective(45.0_f32.to_radians(), aspect, 0.05, far)
        };

        let light = self.real_game_light;
        let view_projection = mat_mul(proj, view);


        let extracted: Vec<super::skinned_actor::SkinnedActorDraw> = real_game_extracted;
        let prepared: Vec<Option<PreparedActor>> =
            extracted.iter().map(|actor| self.prepare_actor(actor)).collect();

        let sun_dir = {
            let d = bevy::math::Vec3::from_array(light.sun_dir);
            if d.length_squared() > 1e-6 { d.normalize() } else { bevy::math::Vec3::new(-0.33, -0.67, -0.33).normalize() }
        };

        let shadow_up = if sun_dir.y.abs() > 0.99 { [0.0, 0.0, 1.0] } else { [0.0, 1.0, 0.0] };
        let shadow_center = self.camera_position;
        let shadow_eye = [
            shadow_center[0] - sun_dir.x * SHADOW_DISTANCE,
            shadow_center[1] - sun_dir.y * SHADOW_DISTANCE,
            shadow_center[2] - sun_dir.z * SHADOW_DISTANCE,
        ];
        let shadow_view = look_at(shadow_eye, shadow_center, shadow_up);
        let shadow_proj = orthographic(
            -SHADOW_HALF_EXTENT,
            SHADOW_HALF_EXTENT,
            -SHADOW_HALF_EXTENT,
            SHADOW_HALF_EXTENT,
            0.1,
            SHADOW_DISTANCE * 2.0,
        );
        let light_view_projection = mat_mul(shadow_proj, shadow_view);

        // Quality 0 means the player turned shadows off, so the pass has to go
        // entirely. Shrinking the shadow map, which is all the setting used to
        // do, left both passes still drawing every mesh in the scene — the
        // texture was smaller but the geometry was not.
        let draw_sun_shadows = self.shadow_quality_requested > 0;
        // Nothing to cast from if there is no point light in range.
        let draw_point_shadows = draw_sun_shadows && light.range > 0.0;

        if draw_sun_shadows {
            let shadow_pass_descriptor = RenderPassDescriptor::new();
            let shadow_depth = shadow_pass_descriptor.depth_attachment().unwrap();
            shadow_depth.set_texture(Some(&self.shadow_map));
            shadow_depth.set_load_action(MTLLoadAction::Clear);
            shadow_depth.set_store_action(MTLStoreAction::Store);
            shadow_depth.set_clear_depth(1.0);

            let shadow_command_buffer = self.queue.new_command_buffer();
            let shadow_encoder = shadow_command_buffer.new_render_command_encoder(shadow_pass_descriptor);
            shadow_encoder.set_depth_stencil_state(&self.shadow_depth_state);

            if self.shadow_quality > 0 {
                self.draw_shadow_casters(shadow_encoder, light_view_projection, &extracted, &prepared, real_game_meshes_opt);
            }
            shadow_encoder.end_encoding();
            shadow_command_buffer.commit();
            self.last_command_buffers.push(("sun shadow", shadow_command_buffer.to_owned()));
        }

        let point_light_pos = light.position;
        let point_to_center = (bevy::math::Vec3::from_array(shadow_center) - bevy::math::Vec3::from_array(point_light_pos)).normalize_or_zero();
        let point_up = if point_to_center.y.abs() > 0.99 { [0.0, 0.0, 1.0] } else { [0.0, 1.0, 0.0] };
        let point_light_view = look_at(point_light_pos, shadow_center, point_up);
        let point_light_proj = perspective(90.0_f32.to_radians(), 1.0, 0.1, light.range.max(SHADOW_DISTANCE * 2.0));
        let point_light_view_projection = mat_mul(point_light_proj, point_light_view);

        if draw_point_shadows {
            let point_shadow_pass_descriptor = RenderPassDescriptor::new();
            let point_shadow_depth = point_shadow_pass_descriptor.depth_attachment().unwrap();
            point_shadow_depth.set_texture(Some(&self.shadow_map_point));
            point_shadow_depth.set_load_action(MTLLoadAction::Clear);
            point_shadow_depth.set_store_action(MTLStoreAction::Store);
            point_shadow_depth.set_clear_depth(1.0);

            let point_shadow_command_buffer = self.queue.new_command_buffer();
            let point_shadow_encoder = point_shadow_command_buffer.new_render_command_encoder(point_shadow_pass_descriptor);
            point_shadow_encoder.set_depth_stencil_state(&self.shadow_depth_state);
            if self.shadow_quality > 0 {
                self.draw_shadow_casters(point_shadow_encoder, point_light_view_projection, &extracted, &prepared, real_game_meshes_opt);
            }
            point_shadow_encoder.end_encoding();
            point_shadow_command_buffer.commit();
            self.last_command_buffers.push(("point shadow", point_shadow_command_buffer.to_owned()));
        }


        let pass_descriptor = RenderPassDescriptor::new();
        let color_attachment = pass_descriptor.color_attachments().object_at(0).unwrap();
        if let Some(msaa_color_texture) = &msaa_color_texture {

            color_attachment.set_texture(Some(msaa_color_texture));
            color_attachment.set_resolve_texture(Some(&texture));
            color_attachment.set_store_action(MTLStoreAction::MultisampleResolve);
        } else {
            color_attachment.set_texture(Some(&texture));
            color_attachment.set_store_action(MTLStoreAction::Store);
        }
        color_attachment.set_load_action(MTLLoadAction::Clear);

        color_attachment.set_clear_color(MTLClearColor::new(0.05, 0.055, 0.07, 1.0));

        let depth_attachment = pass_descriptor.depth_attachment().unwrap();
        depth_attachment.set_texture(Some(&depth_texture));
        depth_attachment.set_load_action(MTLLoadAction::Clear);
        depth_attachment.set_store_action(MTLStoreAction::DontCare);
        depth_attachment.set_clear_depth(1.0);

        timings.prepasses_ms = passes_start.elapsed().as_secs_f32() * 1000.0;
        let scene_start = std::time::Instant::now();
        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_render_command_encoder(pass_descriptor);
        encoder.set_depth_stencil_state(&self.depth_state);
        self.draw_scene(
            encoder,
            view_projection,
            light,
            &extracted,
            &prepared,
            now,
            real_game_meshes_opt,
            msaa_enabled,
            light_view_projection,
            point_light_view_projection,
            self.camera_position,
        );


        let ui_pipeline_ref = if msaa_enabled { &self.ui_pipeline } else { &self.ui_pipeline_no_msaa };
        if let Some(ui_pipeline) = ui_pipeline_ref {

            encoder.set_depth_stencil_state(&self.ui_depth_state);
            if !self.ui_generic_nodes.is_empty() {
                let (view_w, view_h) = self.view_size_points();
                encoder.set_render_pipeline_state(ui_pipeline);
                encoder.set_fragment_sampler_state(0, Some(&self.ui_sampler));

                let mut ui = UiEncoder { encoder, bound_texture: 0, scissor: None };
                draw_generic_ui_nodes(
                    &mut ui,
                    &self.ui_generic_nodes,
                    render_width as f32,
                    render_height as f32,
                    view_w,
                    view_h,
                    &self.white_texture,
                    &self.font_atlas_texture,
                    &self.real_game_textures,
                );
            }
        }

        encoder.end_encoding();
        // With nothing to blit, the scene buffer presents its own drawable.
        // Handing that to a second submission costs an extra command buffer and
        // makes the present wait on this one through the scheduler.
        if skip_blit {
            command_buffer.present_drawable(drawable);
        }
        command_buffer.commit();
        if skip_blit {
            capture_if_due(&mut self.screen_capture, &command_buffer, drawable_texture);
        }
        timings.scene_ms = scene_start.elapsed().as_secs_f32() * 1000.0;
        self.last_command_buffers.push(("scene", command_buffer.to_owned()));
        if skip_blit {
            return timings;
        }

        let blit_start = std::time::Instant::now();

        let blit_command_buffer = self.queue.new_command_buffer();
        if let Some(blit_pipeline) = self.blit_pipeline.as_ref() {
            let blit_pass_descriptor = RenderPassDescriptor::new();
            let blit_color_attachment = blit_pass_descriptor.color_attachments().object_at(0).unwrap();
            blit_color_attachment.set_texture(Some(drawable_texture));
            blit_color_attachment.set_load_action(MTLLoadAction::DontCare);
            blit_color_attachment.set_store_action(MTLStoreAction::Store);
            let blit_encoder = blit_command_buffer.new_render_command_encoder(blit_pass_descriptor);
            blit_encoder.set_render_pipeline_state(blit_pipeline);
            blit_encoder.set_vertex_buffer(0, Some(&self.blit_vertex_buffer), 0);
            blit_encoder.set_fragment_texture(0, Some(&texture));
            blit_encoder.set_fragment_sampler_state(0, Some(&self.sponza_sampler));
            blit_encoder.draw_primitives(MTLPrimitiveType::Triangle, 0, 6);
            blit_encoder.end_encoding();
        }
        blit_command_buffer.present_drawable(drawable);
        blit_command_buffer.commit();
        capture_if_due(&mut self.screen_capture, &blit_command_buffer, drawable_texture);
        self.last_command_buffers.push(("blit", blit_command_buffer.to_owned()));
        timings.blit_ms = blit_start.elapsed().as_secs_f32() * 1000.0;
        timings
    }
}

/// Where one frame's CPU time went inside the renderer. `drawable_wait_ms` is
/// the one to read first: it is time blocked on the display handing back a
/// drawable, so a large value means the frame is limited by the GPU or by the
/// refresh rate, not by anything the CPU is doing.
#[derive(Default, Copy, Clone)]
pub struct FrameTimings {
    pub drawable_wait_ms: f32,
    /// Meshes the main camera's frustum rejected this frame.
    pub culled: usize,
    /// Texture and mesh cache upkeep plus building the draw list.
    pub bookkeeping_ms: f32,
    /// Shadow maps and the scope render target.
    pub prepasses_ms: f32,
    pub scene_ms: f32,
    pub blit_ms: f32,
    pub draws: usize,
    /// GPU-side cache sizes. A frame rate that decays while the scene stands
    /// still is usually one of these climbing.
    pub cached_textures: usize,
    pub cached_meshes: usize,
    pub texture_uploads: usize,
    /// Vertex shader invocations per frame. The ECS path de-indexes its meshes,
    /// so this is the index count, not the unique-vertex count.
    pub vertices_drawn: usize,
    /// Meshes taking the blended path: no depth write, so nothing behind them
    /// can be rejected early.
    pub blend_draws: usize,
    /// Vertices the meshes actually define, against `vertices_drawn` which
    /// counts index fetches. The ratio is how much the loader split them.
    pub unique_vertices: usize,
    /// Real GPU execution time for the scene pass, from the previous frame.
    pub gpu_ms: f32,
    /// Every command buffer the frame submitted, not just the scene pass.
    pub gpu_total_ms: f32,
    /// How many command buffers the frame submitted. Counted on submission, not
    /// from the timestamps — a buffer with no GPU work has none, so it is
    /// invisible in the timings while still costing a submission.
    pub command_buffers: usize,
    pub gpu_status: u64,
    pub gpu_start: f64,
    pub gpu_end: f64,
    /// The size actually rendered at, so GFX_RES_SCALE can be seen to apply.
    pub render_size: (u64, u64),
    pub skipped_blit: bool,
}
