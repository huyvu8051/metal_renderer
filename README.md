# metal_renderer

A hand-written Metal renderer for iOS, driven by plain Bevy data.

It knows nothing about any particular game. A host builds a `RenderSnapshot`
from its own ECS world each frame and hands it over:

```rust
let snapshot = my_extractor(app.world_mut(), |id| renderer.has_uploaded_mesh(id));
renderer.render_frame(snapshot);
```

Splitting update, extraction and encoding across three host-visible calls is
deliberate: it is what lets a host time each phase separately.

## Used by

- `bevy_fps_demo` — the game, via `crates/client/src/cadisplaylink/`
- `desert_rust_bevy` — a minimal Bevy app over this renderer
- `desert_rust_metal` — the same map with no engine at all, the control case

Those two are measured against `desert_swift_metal` (Swift + Metal, no engine),
which is the floor. Naming is language + what sits above Metal.

## Decisions that exist for performance reasons

These all look like they could be simplified. They cannot — each one was
measured, and the write-up in `desert_rust_bevy/README.md` has the numbers.

**Textures carry a mip chain and the sampler uses it.** `upload_texture` builds
the chain on the CPU at load; `MTLTextureDescriptor` defaults to one level and
`MTLSamplerDescriptor` defaults to `NotMipmapped`, so both halves have to be
set or minified fragments read level 0 and miss the texture cache. Leaving
either out cost roughly five times the frame rate on the Desert City map.

**Blending is enabled per pipeline, not globally.** The opaque pipeline has it
off. Switching it on for everything costs a read-modify-write of the tile's
colour buffer on every fragment in the frame, and it is invisible because
opaque textures have alpha = 1.

**There are four scene pipelines, not one.** Opaque with and without shadow
sampling, a blended variant without shadows, normal mapping or specular, and
the MSAA pairs of each. A quality setting that only scales a uniform or shrinks
a texture still pays for the fetches and the ALU; removing the work needs a
different compilation.

**The shadow passes are skipped at quality zero,** and the point-light pass
whenever no point light is in range. They draw the whole scene into depth maps,
so shrinking the maps saves nothing — the cost is the draw calls.

**The offscreen colour target is opt-in.** It is only worth a full-screen store
plus a full-screen load when the scene renders at a size other than the
drawable's, or when MSAA needs resolving.

**Meshes are drawn indexed.** Expanding the index list into a flat vertex array
runs the vertex shader once per index instead of once per vertex, with nothing
for the post-transform cache to reuse.

## Measurement

`render_frame` returns `FrameTimings`. Two fields matter more than they look:

`drawable_wait_ms` is time blocked in `next_drawable`, so it measures the GPU
and the refresh rate. Folded into an "encode" total it reads exactly like CPU
work, which sent several rounds of investigation after the wrong suspect.

`gpu_total_ms` covers every command buffer the frame submitted, not just the
scene pass. Measuring one buffer makes moving work into another buffer look
like removing it.

