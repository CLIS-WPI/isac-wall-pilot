# API notes — executed evidence only

Rule (AGENTS.md §2): before any Sionna / Mitsuba / Dr.Jit call is used in package
code, its real signature, argument meanings, return shapes/dtypes, and defaults
must be recorded here together with a snippet that was actually executed and its
actual printed output.

Environment for every snippet below: image `isac-wall-pilot:2.2.0`
(torch 2.9.1+cu126, sionna 2.2.0, sionna-rt 2.2.0, mitsuba 3.9.1, drjit 1.5.0),
NVIDIA H100 NVL.

## Environment / imports

```python
import sionna, sionna.rt as rt, mitsuba as mi
print(sionna.__version__, mi.variant())
```

```
2.2.0 cuda_ad_mono_polarized
```

`import sionna.rt` triggers the installed backend selection. Do not call
`mi.set_variant()` after scene objects exist. `mi.variant()` must start with
`cuda`; preflight hard-fails otherwise.

## `load_scene`

```python
print(inspect.signature(load_scene))
s = load_scene(rt.scene.simple_reflector)
print(type(s), list(s.objects), list(s.radio_materials))
print(s.tx_array, s.rx_array, s.frequency, s.bandwidth)
```

```
(filename: 'str | None' = None, merge_shapes: 'bool' = True, merge_shapes_exclude_regex: 'str | None' = None, remove_duplicate_vertices: 'bool' = False) -> 'Scene'
<class 'sionna.rt.scene.Scene'> ['merged-shapes'] ['reflector-mat']
None None [3.5e+09] [1e+06]
```

- `filename=None` loads an empty/default scene.
- `merge_shapes=True` (default) merges shapes sharing a radio material, so the
  built-in `simple_reflector` appears as one object named `merged-shapes`.
- `tx_array` and `rx_array` are `None` until assigned on the scene.
- Scene defaults here: frequency 3.5 GHz, bandwidth 1 MHz (actual values are
  Dr.Jit scalars printed as `[3.5e+09]`, `[1e+06]`).

## Custom scene construction from geometry

`load_scene_from_string` signature (same preprocess/merge semantics as
`load_scene`):

```python
import inspect
print(inspect.signature(load_scene_from_string))
```

```
(xml_string: 'str', merge_shapes: 'bool' = True, merge_shapes_exclude_regex: 'str | None' = None, remove_duplicate_vertices: 'bool' = False) -> 'Scene'
```

The preflight scene is two finite `rectangle` walls (20 m x 5 m at y = -2 and
5 m x 10 m at x = 8) in one XML string with a placeholder ITU concrete material.
Executed:

```python
scene = load_scene_from_string(XML, merge_shapes=False)
print(type(scene), list(scene.objects), list(scene.radio_materials))
```

```
<class 'sionna.rt.scene.Scene'> ['wall-a', 'wall-b'] ['wall-mat']
```

`merge_shapes=False` is required to keep the two walls as separate
`SceneObject`s (`wall-a`, `wall-b`). `scene.objects` is a
`dict[str, SceneObject]`.

Python-side `RadioMaterial` creation and assignment (executed):

```python
mat = RadioMaterial(name="wall-material", thickness=0.1,
                    relative_permittivity=5.0, conductivity=0.02,
                    scattering_coefficient=0.0)
print(mat.relative_permittivity, mat.conductivity, mat.scattering_coefficient)
scene.add(mat)
for name, obj in scene.objects.items():
    print(name, obj.radio_material.name)
    obj.radio_material = mat.name
    print(name, obj.radio_material.name)
```

```
[5] [0.02] [0]
wall-a wall-mat
wall-a wall-material
wall-b wall-mat
wall-b wall-material
```

- `RadioMaterial` signature: `(name=None, thickness=0.1,
  relative_permittivity=1.0, conductivity=0.0, scattering_coefficient=0.0,
  xpd_coefficient=0.0, scattering_pattern='lambertian',
  frequency_update_callback=None, color=None, props=None, **kwargs)`.
- Material parameters are Dr.Jit scalars (printed with brackets).
- `SceneObject.radio_material` accepts a material object or its name string.
- `Scene.add(item)` accepts `SceneObject`, `RadioDevice`, `RadioMaterialBase`,
  or a list of these; adding scene objects rebuilds the Mitsuba scene, so add
  in batches. `scene.objects`, `scene.sensing_targets`,
  `scene.radio_materials`, `scene.transmitters`, and `scene.receivers` are
  dicts; `Scene.targets(...)` is a method used internally by solvers, not the
  target dictionary.

## `PlanarArray`

```python
a = PlanarArray(num_rows=1, num_cols=1, vertical_spacing=0.5,
                horizontal_spacing=0.5, pattern="iso", polarization="V")
print(type(a), a.num_ant, a.array_size, type(a.positions),
      np.array(a.normalized_positions).shape)
print(a.antenna_pattern)
```

```
<class 'sionna.rt.antenna_array.PlanarArray'> 1 1 <class 'method'> (3, 1)
<sionna.rt.antenna_pattern.PolarizedAntennaPattern object at 0x...>
```

- Signature (keyword-only): `(*, num_rows, num_cols, vertical_spacing=0.5,
  horizontal_spacing=0.5, pattern, **kwargs)`.
- `polarization="V"` is accepted via `**kwargs` and produces a
  `PolarizedAntennaPattern`.
- `positions` is a method, not an attribute; `normalized_positions` is a
  `mitsuba.Point3f` of shape `(3, num_ant)`.
- Assigning `scene.tx_array = planar_array` stores the `PlanarArray` descriptor;
  `scene.tx_array` then returns a `PlanarArray` (not a separate
  `AntennaArray`). For one element, `num_ant=1`, `array_size=1`.

## `Transmitter` / `Receiver`

```python
print(inspect.signature(Transmitter))
print(inspect.signature(Receiver))
scene.add(Transmitter("tx", position=[-4, 2, 2]))
print(type(scene.get("tx").position), scene.get("tx").position,
      scene.get("tx").power_dbm)
```

```
(name: str, position: mitsuba.Point3f, orientation: mitsuba.Point3f | None = None, look_at: Union[mitsuba.Point3f, Self, NoneType] = None, velocity: mitsuba.Vector3f | None = None, power_dbm=44, color: Tuple[float, float, float] = (1.0, 0.0, 0.0), display_radius: float | None = None)
(name: str, position: mitsuba.Point3f, orientation: mitsuba.Point3f | None = None, look_at: Union[mitsuba.Point3f, Self, NoneType] = None, velocity: mitsuba.Vector3f | None = None, color: Tuple[float, float, float] = (0.4, 0.8, 0.4), display_radius: float | None = None)
<mitsuba.Point3f> [[-4, 2, 2]] [44]
```

- List positions are converted to `mitsuba.Point3f`.
- Transmitter default power is 44 dBm. Receiver has no power argument.
- `scene.get(name)` returns the device.

## `PathSolver`

```python
print(inspect.signature(PathSolver))
print(inspect.signature(PathSolver.__call__))
paths = PathSolver()(scene, max_depth=3, seed=42)
print(type(paths.tau), paths.tau.shape, paths.tau)
print(type(paths.valid), paths.valid.shape, paths.valid)
print(type(paths.a), paths.a.shape)  # paths.a is a tuple
print(type(paths.interactions), paths.interactions.shape)
```

```
(deterministic: bool = False)
(self, scene, max_depth=3, max_num_paths_per_src=1000000, samples_per_src=1000000, synthetic_array=True, los=True, specular_reflection=True, diffuse_reflection=False, refraction=True, diffraction=False, edge_diffraction=False, diffraction_lit_region=True, seed=42) -> Paths
<class 'drjit.cuda.ad.TensorXf'> (1, 1, 4) [[[2.66851e-08, 3.77385e-08, 5.33703e-08, 5.96698e-08]]]
<class 'drjit.cuda.ad.TensorXb'> (1, 1, 4) [[[True, True, True, True]]]
<class 'tuple'> (1, 1, 1, 1, 4)
<class 'drjit.cuda.ad.TensorXu'> (3, 1, 1, 4)
```

- `Paths.tau`: `(num_rx, num_tx, num_paths)` absolute delays (measured before
  calling `cir`). `Paths.cir()` returns its own delay tensor, normalized by
  default; pass `normalize_delays=False` to keep absolute delays.
- `Paths.valid`: `(num_rx, num_tx, num_paths)` boolean mask.
- `Paths.a`: **tuple** `(real, imag)`, each
  `(num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths)`.
- `Paths.interactions`: `(max_depth, num_rx, num_tx, num_paths)`; value `0` is
  LoS and `1` is specular reflection for this scene. The measured matrix was
  `[[[0, 1, 1, 1]], [[0, 0, 0, 1]], [[0, 0, 0, 0]]]`, i.e. 3 of 4 paths are
  specular.
- `seed` defaults to 42; pass an explicit seed for reproducibility.
- Scene: tx `(-4,2,2)`, rx `(4,2,2)`, two walls. Four valid paths: LoS plus
  three wall reflections.

## `Paths.cir`

```python
print(inspect.signature(Paths.cir))
a, tau = paths.cir(out_type="torch", normalize_delays=False)
print(type(a), tuple(a.shape), a.device, a.dtype,
      type(tau), tuple(tau.shape), tau.device, tau.dtype)
print(float(a.abs().max()), float(tau.min()), float(tau.max()))
a_def, tau_def = paths.cir(out_type="torch")
print(tau_def)
```

```
(self, *, sampling_frequency=1.0, num_time_steps=1, normalize_delays=True, reverse_direction=False, out_type='drjit')
<class 'torch.Tensor'> (1, 1, 1, 1, 4, 1) cuda:0 torch.complex64 <class 'torch.Tensor'> (1, 1, 4) cuda:0 torch.float32
0.0008520259289070964 2.6685126996994768e-08 5.966975891169568e-08
tensor([[[0.0000e+00, 1.1053e-08, 2.6685e-08, 3.2985e-08]]], device='cuda:0')
```

- Defaults: `sampling_frequency=1.0`, `num_time_steps=1`,
  `normalize_delays=True`, `reverse_direction=False`, `out_type='drjit'`.
- `normalize_delays=True` subtracts the minimum path delay per tx/rx pair:
  the LoS path becomes exactly 0 s and the reflections become
  `1.1053e-8, 2.6685e-8, 3.2985e-8` s.
- `normalize_delays=False` keeps absolute delays (LoS `2.6685127e-8` s for
  8 m), which is required for absolute-delay processing.
- **`out_type="torch"` returns `torch.complex64`/`torch.float32` tensors
  directly on `cuda:0`** in this CUDA RT backend (no CPU round trip).
- Channel tensor shape with 1 Tx and 1 Rx antenna:
  `(num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths, num_time_steps)`.

## `RCSSolver` and target object

```python
print(inspect.signature(RCSSolver))
print(inspect.signature(RCSSolver.__call__))
print(inspect.signature(ConstantRCSSensingTarget))
target = ConstantRCSSensingTarget(name="target", sigma=1.0, length=0.2,
                                  width=0.2, height=0.2,
                                  position=[0, 5, 2])
print(type(target), target.name, target.sigma, target.position)
scene.add(target)
print(list(scene.sensing_targets))
```

```
(deterministic: bool = False)
(self, scene, max_depth=3, buffer_size_per_sp=1000000, samples_per_sp=1000000, synthetic_array=True, los=True, specular_reflection=True, refraction=True, seed=None) -> Paths
(name: 'str', sigma: 'float | mi.Float' = 1.0, xpr_db: 'float | mi.Float | None' = None, mi_mesh: 'mi.Mesh | None' = None, fname: 'str | None' = None, length: 'float | None' = None, width: 'float | None' = None, height: 'float | None' = None, color: 'Tuple[float, float, float]' = (0.2, 0.4, 1.0), display_opacity: 'float' = 0.5, position: 'mi.Point3f | None' = None, orientation: 'mi.Point3f | None' = None, look_at: 'mi.Point3f | SceneObject | RadioDevice | None' = None, velocity: 'mi.Vector3f | None' = None)
<class 'sionna.rt.rcs.constant_rcs_sensing_target.ConstantRCSSensingTarget'> target [1] <class 'mitsuba.Point3f'> [[0, 5, 2]]
['target']
```

Executed solve (`max_depth=1`, both legs unobstructed):

```python
rpaths = RCSSolver()(scene, max_depth=1, seed=42)
print(rpaths.tau.shape, rpaths.tau, rpaths.valid)
print(rpaths.a.shape, rpaths.a)
ra, rtau = rpaths.cir(out_type="torch", normalize_delays=False)
print(tuple(ra.shape), ra.device, ra.dtype, float(ra.abs().max()),
      tuple(rtau.shape), float(rtau.min()))
```

```
(1, 1, 1) [[[3.33564e-08]]] [[[True]]]
(1, 1, 1, 1, 1) ([[[[[7.69127e-05]]]]], [[[[[0]]]]])
(1, 1, 1, 1, 1, 1) cuda:0 torch.complex64 7.691267092013732e-05 (1, 1, 1) 3.335640741397583e-08
```

- `RCSSolver` computes only paths that interact with a sensing target; the
  scattering event counts as one interaction, so `max_depth=1` restricts both
  legs to unobstructed LoS.
- Target used: `ConstantRCSSensingTarget` (default cube geometry, RCS
  `sigma=1.0 m^2`, no cross-polarization, scattering point at the target
  centre). Targets are added with `scene.add(...)` and appear in
  `scene.sensing_targets`.
- Executed target at `(0,5,2)`: two legs `5 m + 5 m = 10 m`, expected delay
  `10/c = 3.33564095e-8 s`; measured `3.33564074e-8 s` (relative error
  `6.31e-8`).
- Target return is nonzero: `|a|max = 7.6913e-5`, complex64 on `cuda:0`.

## Preflight scene: executed output

Two-wall custom XML scene, material `eps_r=5.0`, `sigma=0.02`,
`scattering_coefficient=0.0`, single isotropic V-polarized element at both ends:

```
PathSolver: valid=4/4 specular_valid=3 cir=(1, 1, 1, 1, 4, 1) device=cuda:0 |a|max=8.520259e-04 LoS delay rel.err=2.319e-08
RCSSolver: valid=1 cir=(1, 1, 1, 1, 1, 1) device=cuda:0 |a|max=7.691267e-05 target delay rel.err=6.313e-08
```

These are the actual values written to `results/preflight/environment.json` in
Phase 0.
