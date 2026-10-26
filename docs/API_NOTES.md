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
- Target used: `ConstantRCSSensingTarget` (`length=width=height=0.2 m`, RCS
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

## Phase 1 probe run

Command (inside the repository, live source mount):

```bash
ISAC_DEV=1 scripts/drun.sh python scripts/api_probe.py
```

Final output:

```
status: PASS
wrote results/phase1/api_probe.json
PASS paths_and_angles
PASS cir
PASS los_amplitude_carrier_phase
PASS cfr_and_normalization
PASS reflection_and_material
PASS rcs
PASS synthetic_array
PASS conversions
PASS isac
PASS array
PASS frequency_callback
PASS interop
PASS determinism
```

The probe hard-fails on any mismatch; `results/phase1/api_probe.json` is the
executed evidence file.

## `Paths` angle, vertex, and interaction conventions

Executed with the two-wall scene, `PathSolver()(scene, max_depth=1, seed=42)`:

```python
paths = PathSolver()(scene, max_depth=1, seed=42)
print(paths.tau.shape, paths.theta_t.shape, paths.theta_r.shape,
      paths.vertices.shape, paths.interactions.shape)
print(paths.theta_t, paths.phi_t, paths.theta_r, paths.phi_r)
```

```
(1, 1, 3) (1, 1, 3) (1, 1, 3) (1, 1, 1, 3, 3) (1, 1, 1, 3)
[[[1.5708, 1.5708, 1.5708]]] [[[0.0000, 0.0000, 0.0000]]]
[[[1.5708, 1.5708, 1.5708]]] [[[3.1416, 3.1416, 3.1416]]]
```

- `theta_t`, `phi_t`, `theta_r`, `phi_r`: `(num_rx, num_tx, num_paths)`,
  float32 on `cuda:0`. Zenith `theta` is measured from `+z`; azimuth `phi` is
  measured from `+x` toward `+y`.
- Unit direction: `r_hat(theta, phi) = [sin(theta)cos(phi),
  sin(theta)sin(phi), cos(theta)]`.
- LoS TX `(-4,2,2)` to RX `(4,2,2)` along `+x`: TX departure
  `theta_t=pi/2, phi_t=0`; RX arrival `theta_r=pi/2, phi_r=pi`. The RX angle
  vector points from RX back toward TX.
- `Paths.vertices`: `(max_depth, num_rx, num_tx, num_paths, 3)`. The LoS entry
  is `[0,0,0]`; a reflection entry is the specular point. Measured wall points:
  `[-2.18e-5, -2.0, 2.0000002]` and `[8.0, 1.9999998, 2.0]`.
- `Paths.interactions`: `(max_depth, num_rx, num_tx, num_paths)`, uint32.
  `InteractionType.NONE=0`, `SPECULAR=1`, `DIFFUSE=2`, `REFRACTION=4`,
  `DIFFRACTION=8`, `SENSING=16`. LoS is all-zero in the first row; unused depth
  slots are also zero.

## `Paths.cir` and `Paths.cfr`

`Paths.cir` signature (no amplitude-normalize argument):

```
(*, sampling_frequency: float = 1.0, num_time_steps: int = 1,
 normalize_delays: bool = True, reverse_direction: bool = False,
 out_type: Literal['drjit', 'jax', 'numpy', 'tf', 'torch'] = 'drjit')
```

`Paths.cir(normalize_delays=False)` measured for the two-wall scene:

```
a shape=(1, 1, 1, 1, 3, 1) dtype=torch.complex64 device=cuda:0
tau shape=(1, 1, 3) dtype=torch.float32 device=cuda:0
tau min=2.6685126996994768e-08 s max=5.3370253993989536e-08 s
|a|max=0.0008520259289070964
```

`Paths.cfr` signature:

```
(frequencies: drjit.cuda.ad.Float, sampling_frequency: float = 1.0,
 num_time_steps: int = 1, normalize_delays: bool = True,
 normalize: bool = False, reverse_direction: bool = False,
 out_type: Literal['drjit', 'jax', 'numpy', 'tf', 'torch'] = 'drjit')
```

- `Paths.a` is the passband coefficient without the propagation phase. For
  LoS, `a` is real positive and equals `lambda/(4*pi*d)`.
- `Paths.cir` multiplies `a` by `exp(-j*2*pi*f_c*tau)`, so it includes the
  carrier phase. Measured LoS `a=8.5202593e-4`, `tau=2.6685127e-8 s`,
  `cir` phase matches the carrier factor.
- `Paths.cfr` expects baseband subcarrier frequencies; passing absolute
  frequencies adds the carrier a second time. The probe measured an absolute-
  versus-baseband CFR difference of `1.6172385e-3`, which is the expected
  double-counting effect.
- `Paths.cfr` output omits the singleton `num_paths` axis; shape is
  `(1,1,1,1,1,2048)`. `cir_to_ofdm_channel` shape is
  `(1,1,1,1,1,1,2048)`.

## `subcarrier_frequencies` and `cir_to_ofdm_channel`

```
subcarrier_frequencies(num_subcarriers: int, subcarrier_spacing: float,
                       precision: Optional[str] = None,
                       device: Optional[str] = None) -> torch.Tensor

cir_to_ofdm_channel(frequencies: torch.Tensor, a: torch.Tensor,
                    tau: torch.Tensor, normalize: bool = False) -> torch.Tensor
```

Executed at 3.5 GHz, FFT 2048, SCS 60 kHz:

```
freqs shape=(2048,) dtype=torch.float32 device=cuda:0
first_three=[-61440000.0, -61380000.0, -61320000.0] Hz
last_three=[61260000.0, 61320000.0, 61380000.0] Hz
center_pair=[-60000.0, 0.0] Hz
```

- Frequencies are baseband, centered around 0, with the expected FFT ordering.
- `cir_to_ofdm_channel(freqs, a, tau, normalize=False)` equals
  `Paths.cfr(freqs, normalize_delays=False, normalize=False)` to
  `3.958549e-10` max absolute difference.
- The subcarrier phase residual against `exp(-j*2*pi*f*tau)` is
  `9.266216e-8` rad max.
- `normalize=True` sets mean power to `1.0`; raw mean power was
  `7.259482e-7`.

## OFDM conversions

Signatures:

```
ofdm_to_time_channel(h_f: torch.Tensor, l_min: int = 0,
                     l_max: Optional[int] = None) -> torch.Tensor
time_to_doppler_channel(h_t: torch.Tensor) -> torch.Tensor
ofdm_to_delay_doppler_channel(h_f: torch.Tensor, l_min: int = 0,
                              l_max: Optional[int] = None) -> torch.Tensor
```

Executed with FFT 2048, 64 symbols, SCS 60 kHz, integer delay `l0=10`:

```
ofdm_to_time_channel: input (1,1,1,1,1,64,2048)
                      output (1,1,1,1,1,64,64)
                      peak lag=10, peak value=1.0
ofdm_to_time_channel(l_min=-10): output (1,1,1,1,1,64,74)
                                  peak index=20
time_to_doppler_channel: input (1,1,1,1,1,64,8)
                         output (1,1,1,1,1,64,8)
                         q0=2 peak index=34, center=32
ofdm_to_delay_doppler_channel: input (1,1,1,1,1,64,2048)
                               output (1,1,1,1,1,64,64)
                               peak (q=34, l=10)
```

- `ofdm_to_time_channel` uses a centered subcarrier ordering and inverse-FFT
  taps; `l_min` shifts the lag index.
- `time_to_doppler_channel` centers Doppler bins; for 64 symbols, q=+2 maps to
  index 34.
- `ofdm_to_delay_doppler_channel` requires the complete time-frequency grid
  `[..., num_time_steps, fft_size]`; it applies
  `exp(+j*2*pi*n*l/N) * exp(-j*2*pi*b*q/S)` as measured.

## `steering_vectors` and `angular_delay_doppler_spectrum`

Signatures:

```
steering_vectors(positions: torch.Tensor, theta: float | torch.Tensor,
                 phi: float | torch.Tensor, wavelength: float | torch.Tensor,
                 *, mode: str = 'cartesian',
                 precision: Optional[Literal['single', 'double']] = None)
                 -> torch.Tensor

angular_delay_doppler_spectrum(h_dd: torch.Tensor,
                               rx_steering_vectors: torch.Tensor,
                               tx_steering_vectors: torch.Tensor,
                               *, mode: str = 'paired') -> torch.Tensor
```

Executed with `M=4`, `lambda=0.0856549814 m`, ULA along `y`:

```
paired scalar shape=(4,)
cartesian scalar shape=(1, 1, 4)
cartesian phi-array shape=(1, 2, 4)
unit norm=1.0
```

- Formula: `a_m = exp(+j*2*pi/lambda * d_m . r_hat(theta, phi)) / sqrt(M)`.
- Paired scalar returns `[M]`, not `[1,M]`; Cartesian scalar returns
  `[1,1,M]`; Cartesian phi arrays add the corresponding axis.
- RT synthetic-array phases match this convention for both TX departure and
  RX arrival, with measured errors `5.761645e-6` (TX) and `4.270060e-6` (RX).
- `angular_delay_doppler_spectrum` with `h_dd` shape `(1,1,4,1,4,1,1)` and
  5x5 Cartesian scans returned `(1,1,5,1,5,1,1)`. Matched peak `1.0` at
  `[0,0,3,0,1,0,0]` for RX `phi=0.6`, TX `phi=-0.4`; global max was `1.0`.
- For a transmit beam, `a_TX` is the physical TX response; matched transmit
  weights are `conj(a_TX)`. RX steering is `a_RX`.

## `PlanarArray` 16-element ULA

```
PlanarArray(*, num_rows: int, num_cols: int, vertical_spacing: float = 0.5,
            horizontal_spacing: float = 0.5, pattern: str, **kwargs)
```

Executed with `num_rows=1, num_cols=16, vertical_spacing=0.5,
horizontal_spacing=0.5, pattern="iso", polarization="V"`:

```
num_ant=16 array_size=16
normalized_positions shape=(3,16)
normalized y (wavelengths)=[-3.75,-3.25,...,3.25,3.75]
positions shape=(3,16), physical y=[-0.321206,...,0.321206] m
x=z=0
```

- The ULA is along `y`, lies in the `y-z` plane at `x=0`, and has broadside
  `+x`.
- Element order increases with `y`: row 0, then columns 0..15.
- `positions(wavelength)` returns physical positions in meters; normalized
  positions are in wavelengths.

## `Scene.frequency` and material callback

```
Scene.frequency setter: (self, f)
callback: Callable[[mi.Float], Tuple[mi.Float, mi.Float]]
```

Executed callback `lambda f: (f/1e9, 0.01)`:

```
epsilon at 3.5 GHz=3.5, conductivity=0.01
epsilon at 7 GHz=7.0, conductivity=0.01
material.thickness=0.10000000149 m
```

- The frequency setter requires a positive frequency and calls
  `frequency_update()` on every radio material.
- `wavelength = speed_of_light/f`; LoS amplitude scales with wavelength, delay
  does not. Measured `a_3.5GHz=8.520259e-4`, `a_7GHz=4.2601296e-4`, ratio
  `2.0`; delay unchanged at `2.6685127e-8 s`.

## RT to torch interop

```
mi.TensorXf.torch()                 # works
torch.as_tensor(mi.TensorXf)        # also works
```

Measured `PathSolver` tensors:

```
a_real/a_imag: torch.float32 cuda:0 requires_grad=False
tau: torch.float32 cuda:0 requires_grad=False
valid: torch.bool cuda:0 requires_grad=False
interactions: torch.uint32 cuda:0 requires_grad=False
Paths.cir(out_type="torch"): complex64 cuda:0
```

No CPU round trip is needed. `interactions` is uint32; convert to int64 before
using it as an index in torch (`index_cuda` is not implemented for uint32).

## Determinism

Executed two consecutive solves with the same seed, specular-only scene:

```
PathSolver deterministic=False: tau exact=True, a exact=True
PathSolver deterministic=True:  tau exact=True, a exact=True
RCSSolver deterministic=False:  tau exact=True, a exact=True
RCSSolver deterministic=True:   tau exact=True, a exact=True
```

Sionna documents that a fixed seed alone is not a guarantee; use
`deterministic=True` when bitwise reproducibility is required.

## RCS depth, target occlusion, and sigma

`RCSSolver` and `ConstantRCSSensingTarget` are imported from
`sionna.rt.rcs`, not from `sionna.rt`.

Executed with target `(0,5,2)`, `max_depth` varied:

```
depth 1: valid=1  interactions=[[16]]
depth 2: valid=5  interactions include [[16,0], [1,16], ...]
depth 3: valid=11 interactions include [[16,0,0], [1,16,0], [1,1,16], ...]
target two-leg distance=10.0 m, tau=3.335640741e-8 s
```

- `max_depth` includes the sensing-target scattering event. Depth 1 is
  target-only LoS; depth 2 adds one wall reflection on either leg; depth 3
  adds two-wall combinations.
- `ConstantRCSSensingTarget` uses `AbsorberRadioMaterial`; adding one at the
  LoS midpoint blocked ordinary `PathSolver`: valid went from `1` to `0`.
- `RCSSolver` with no sensing target raises
  `ValueError: Scene has no sensing targets`; it does not return an empty path
  set.
- `ConstantRCSSensingTarget(name="target", sigma=1.0, ...)`: measured
  `sigma=1.0 m^2`, position `[0,5,2]`. With `sigma=4.0`, `|a|` doubled:
  `a_sigma_1=7.691268e-5`, `a_sigma_4=1.5382536e-4`, ratio `2.0`.

## Reflection and material check

Executed `PathSolver(max_depth=1)` with `scene.frequency=3.5 GHz`, material
`eps_r=5.0`, `sigma=0.02 S/m`, thickness `0.1 m`, `lambda=0.0856549814 m`:

```
wall y=-2: L=11.313708499 m, cos(theta)=0.7071087094
  a_RT / (lambda/(4*pi*L)) = -0.2030512689 + 0.1115346686j
  slab r_TE                 = -0.2030507326 + 0.1115333885j
  abs error=1.387875e-6

wall x=8: L=16.000000000 m, cos(theta)=1.0
  a_RT / (lambda/(4*pi*L)) = -0.3660115766 - 0.2306077063j
  slab r_TE                 = -0.3660115600 - 0.2306076884j
  abs error=2.441216e-8
```

- RT uses the finite-thickness slab:
  `itu_coefficients_single_layer_slab(cos_theta, eta, d, wavelength)`, which
  returns `(r_te, r_tm, t_te, t_tm)`.
- `fresnel_reflection_coefficients_simplified` is a single-interface function
  and is not what RT uses for these walls. Executed example at cos=0.70710678:
  `r_te=-0.500061+0.0042787j`, `r_tm=0.250043-0.0042794j`.
- `complex_relative_permittivity(eta_r, sigma, omega)` returns a Dr.Jit
  `Complex2f`; `eta=[5-0.102715j]` at 3.5 GHz.
- The measured V-polarized reflection coefficients match slab `r_TE` for both
  geometries; the corresponding slab `r_TM` values are the negative at normal
  incidence. Do not substitute the simplified single-interface Fresnel
  coefficients for the RT slab coefficients.

## Phase 2 — wall material default and slab coefficients

Executed:

```python
print(inspect.signature(RadioMaterial))
material = RadioMaterial(name="phase2-default-thickness")
print(material.thickness, float(material.thickness[0]))
omega = 2.0 * math.pi * 3.5e9
eta = complex_relative_permittivity(mi.Float(5.0), mi.Float(0.02), mi.Float(omega))
print(eta)
r_te, r_tm, t_te, t_tm = itu_coefficients_single_layer_slab(
    mi.Float(math.cos(math.radians(45.0))),
    eta,
    mi.Float(float(material.thickness[0])),
    mi.Float(0.0856549814),
)
print(r_te, r_tm)
```

```
(name: str | None = None, thickness: float | drjit.cuda.ad.Float | None = 0.1,
 relative_permittivity: float | drjit.cuda.ad.Float = 1.0,
 conductivity: float | drjit.cuda.ad.Float = 0.0,
 scattering_coefficient: float | drjit.cuda.ad.Float = 0.0,
 xpd_coefficient: float | drjit.cuda.ad.Float = 0.0,
 scattering_pattern: str = 'lambertian',
 frequency_update_callback: Optional[Callable[[drjit.cuda.ad.Float], Tuple[drjit.cuda.ad.Float, drjit.cuda.ad.Float]]] = None,
 color: Optional[Tuple[float, float, float]] = None,
 props: mitsuba.Properties | None = None, **kwargs)
thickness [0.1] 0.10000000149011612
eta [5-0.102715j]
slab45 [-0.203053+0.111538j] [0.0855331-0.053198j]
```

- The configured D2 thickness is the realized default
  `0.10000000149011612 m`, not the decimal `0.1`; this is recorded exactly in
  `configs/pilot.yaml` and `results/phase2/material_reference.json`.
- `itu_coefficients_single_layer_slab` returns
  `(r_te, r_tm, t_te, t_tm)` as a tuple of Dr.Jit complex scalars. The printed
  values above are rounded by Dr.Jit's display; the full JSON reference values
  are generated from `.real[0]` and `.imag[0]`.

## Phase 2 — synthetic-array RT adapter for D5

Executed with a 16-element y-axis ULA at 3.5 GHz, two receivers at 0 and 45
degrees, `PathSolver(deterministic=True)`, `max_depth=0`, one LoS path:

```python
paths = PathSolver(deterministic=True)(
    scene, max_depth=0, synthetic_array=synthetic, los=True,
    specular_reflection=False, diffuse_reflection=False,
    refraction=False, diffraction=False, seed=3101,
)
a = torch.as_tensor(paths.a[0]) + 1j * torch.as_tensor(paths.a[1])
print(tuple(a.shape))
print(torch.angle(a[1, 0, 0, :, 0]).mul(180 / math.pi).tolist())
```

```
synthetic False a_shape (2, 1, 1, 16, 1)
rx 1 phase0deg [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
synthetic True  a_shape (2, 1, 1, 16, 1)
rx 1 phase0deg [125.40576934814453, -107.31503295898438, 19.96424102783203, 147.24351501464844, -85.477294921875, 41.80194854736328, 169.0811767578125, -63.639610290527344, 63.639610290527344, -169.0811767578125, -41.80194854736328, 85.477294921875, -147.24351501464844, -19.96424102783203, 107.31503295898438, -125.40576934814453]
```

- `Paths.a` is `(2, num_rx, num_tx, num_ant, num_paths)` for real/imag when
  `synthetic_array=True`; reshape the 16-element axis before applying beam
  weights.
- `synthetic_array=False` applies the array response per element in this
  configuration and the tested `Paths.a` phases were all zero. Use
  `synthetic_array=True` for the D5 validation adapter, then evaluate
  `h @ w` with `v(phi)=conj(a_TX(phi))`; do not add a second conjugate.
- The y-axis ULA has front-back ambiguity. The Phase 2 validation ring uses
  the forward visible arc `[-90, 90]` degrees; a full 360-degree ring produces
  aliased peaks at both `phi` and `180-phi`.

## Phase 3 — absolute CIR/CFR shapes and Experiment A paths

Executed in the live container with the nominal Experiment A geometry
(`gnb=(0,0,2)`, relay `(30,0,2)`, candidate `(60,0,2)`, 16-element y-axis gNB
TX and RX ULAs, a single-antenna relay, no walls, `RCSSolver` for the target and
`PathSolver` for the two relay links):

```python
a_t, tau_t = paths_t.cir(out_type="torch", normalize_delays=False)
a_d, tau_d = paths_d.cir(out_type="torch", normalize_delays=False)
a_u, tau_u = paths_u.cir(out_type="torch", normalize_delays=False)
print(tuple(a_t.shape), tuple(tau_t.shape), tuple(a_d.shape), tuple(a_u.shape))
h_t = cir_to_ofdm_channel(freqs, a_t.unsqueeze(0), tau_t.unsqueeze(0), normalize=False)
h_d = cir_to_ofdm_channel(freqs, a_d.unsqueeze(0), tau_d.unsqueeze(0), normalize=False)
h_u = cir_to_ofdm_channel(freqs, a_u.unsqueeze(0), tau_u.unsqueeze(0), normalize=False)
print(tuple(h_t.shape), tuple(h_d.shape), tuple(h_u.shape))
```

```
target a (1, 16, 1, 16, 1, 1) tau (1, 1, 1) cfr (1, 1, 16, 1, 16, 1, 2048)
downlink a (1, 1, 1, 16, 1, 1) tau (1, 1, 1) cfr (1, 1, 1, 1, 16, 1, 2048)
uplink a (1, 16, 1, 1, 1, 1) tau (1, 1, 1) cfr (1, 1, 16, 1, 1, 1, 2048)
target coefficient (16, 16) delay 4.002769173894194e-07
downlink coefficient (1, 16) delay 1.0006922934735485e-07
uplink coefficient (16, 1) delay 1.0006922934735485e-07
```

- `Paths.cir()` returns 6-D `a` with layout
  `[num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths, num_time_steps]`.
  `cir_to_ofdm_channel` requires the extra leading batch axis, so add
  `a.unsqueeze(0)` and `tau.unsqueeze(0)`. The output layout is
  `[batch, num_rx, num_rx_ant, num_tx, num_tx_ant, num_time_steps, fft_size]`.
- The gNB TX and RX arrays are both 16-element horizontal ULAs at the same
  phase center. The target is receive-by-transmit: its CIR coefficient matrix
  is `(16, 16)`. The downlink gNB-to-relay coefficient is `(1, 16)`; the
  uplink relay-to-gNB coefficient is `(16, 1)`. The relay is single-antenna in
  both directions. Sionna arrays are scene-global, so each link is solved in
  its own scene with its own TX/RX array setting; these settings are recorded
  under `arrays.links` in `configs/pilot.yaml`.
- The TX-antenna axis is axis 4 of the 7-D CFR. The RX-antenna axis is axis 2.
  The package helper `effective_response` extracts
  `cfr[0, 0, :, 0, :, 0, :]` and applies the beam weights, returning a
  `(num_rx_antennas, fft_size)` RX vector per subcarrier. It never collapses the
  RX array before the equality comparison.
- `Paths.cir()` already applies the carrier phase
  `exp(-j 2*pi*f_c*tau)` to `a`; `cir_to_ofdm_channel(..., baseband_freqs,
  normalize=False)` applies only the additional baseband phase. Do not add a
  separate carrier factor for the relay added delay. A relay calibration built
  directly from the `Paths.cir` coefficients matched every RX element and
  subcarrier of the target to `8.310603e-6` maximum relative error over 70
  beams.
- The target and relay receive steering vectors had relative L2 difference
  `1.201e-07` after phase alignment in the collinear Experiment A geometry.
- The nominal Experiment A values measured in Phase 3 were
  `tau_target=400.2769173894194 ns`, `tau_down=tau_up=100.06922934735485 ns`,
  and `delta=200.1384586947097 ns`. The analytical delay was
  `200.1384571188912 ns`; the difference was `1.575818e-6 ns`.

## Phase 3 — why `synthetic_array=False` has zero `Paths.a` phases

Executed with a 16-element y-axis ULA, two receivers at 0 and 45 degrees, one
LoS path, and `PathSolver(deterministic=True, max_depth=0)`:

```python
for synthetic in [False, True]:
    paths = PathSolver(deterministic=True)(scene, max_depth=0,
        synthetic_array=synthetic, los=True, specular_reflection=False,
        diffuse_reflection=False, refraction=False, diffraction=False, seed=7101)
    a = torch.as_tensor(paths.a[0]) + 1j * torch.as_tensor(paths.a[1])
    tau = torch.as_tensor(paths.tau)
    print(synthetic, tuple(a.shape), tuple(tau.shape),
          torch.angle(a[1, 0, 0, :, 0]).mul(180/math.pi).tolist(),
          float(tau[1, 0, :].min()), float(tau[1, 0, :].max()))
```

```
False (2, 1, 1, 16, 1) (2, 1, 1, 16, 1) [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] 9.931449795885783e-08 1.008296877103021e-07
True  (2, 1, 1, 16, 1) (2, 1, 1)       [125.4058837890625, -107.31492614746094, 19.9643497467041, 147.24351501464844, -85.47723388671875, 41.80195236206055, 169.08119201660156, -63.639610290527344, 63.639610290527344, -169.08119201660156, -41.80195236206055, 85.47723388671875, -147.24351501464844, -19.9643497467041, 107.31492614746094, -125.4058837890625] 1.0006922934735485e-07 1.0006922934735485e-07
```

- Installed `Paths.cir` source applies
  `phase = -2*pi*self._frequency*tau_` and then `a = cpx_mul(a, exp)`.
- With `synthetic_array=False`, `Paths.tau` has the full per-element shape
  `(2, 1, 1, 16, 1)` in the executed test, matching the
  `[num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths]` layout of `Paths.a`,
  and already carries the per-element propagation delay. `Paths.a` remains the
  common passband coefficient, so the tested element phases are zero and the
  array phase is in `tau`.
- With `synthetic_array=True`, `Paths.tau` collapses to `(2, 1, 1)`
  `[num_rx, num_tx, num_paths]` and the per-element array phase is applied to
  `Paths.a`. This is the mode used for the D5 RT adapter and all Phase 3
  Experiment A solves. `synthetic_array=False` is explained but not used.

## Phase 3 fix-up — Sionna residual with float64 delays

Executed with the nominal 16/1/16 Experiment A links and the baseline 2048
subcarriers. The CFR was rebuilt after casting `Paths.tau` to `float64`; the
complex64 path coefficients and calibration gain were left unchanged.

```python
target32 = experiment.target_effective(weights)
relay32 = experiment.relay_effective(weights)
target64 = experiment.target_effective(weights, tau_float64=True)
relay64 = experiment.relay_effective(weights, tau_float64=True)
print(target32.dtype, target64.dtype,
      ((target32 - relay32).abs() / target32.abs()).max().item(),
      ((target64 - relay64).abs() / target64.abs()).max().item())
```

```
torch.complex64 torch.complex128 8.310603334393818e-06 8.66097007216447e-06
```

- `cir_to_ofdm_channel` derives its real dtype from `tau.dtype`. Promoting
  `tau` to `float64` therefore promotes the exponential and CFR to
  `complex128`, but cannot recover precision already lost in the complex64
  `Paths.a` coefficients and the complex64 relay calibration gain.
- The measured maximum error increased from `8.310603e-6` to `8.661e-6`, so
  float32 storage of `tau` is not the limiting source. The residual is
  dominated by roundoff in the per-element complex64 path coefficients, their
  array phases, and the calibrated gain product. It remains well inside the
  declared `1e-3` Sionna tolerance; no tolerance was changed.
