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
