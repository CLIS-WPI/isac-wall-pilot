# Developer implementation brief: GPU-only Sionna pilot for active wall-assisted sensing verification

**Project:** ISAC active verification under digital-twin mismatch  
**Target execution:** NVIDIA H100 inside Docker on a Linux host  
**Document date:** October 26, 2026  
**Deliverable:** A complete runnable research repository, measured results, and an honest physical feasibility decision.

## 1. Assignment and scope

Implement and run two small, reproducible experiments that test the physical information available for distinguishing a real scatterer from a causal replay source:

1. **Experiment A — LoS identifiability control:** A real point target and a nearer, collinear replay source produce the same apparent direct-path target. Changing an ordinary far-field transmit beam should not distinguish their matched ideal responses.
2. **Experiment B — wall-assisted verification:** A beam selected using an imperfect map illuminates the candidate position through a sidewall. Determine whether the resulting return supplies useful evidence for a physical scatterer at the claimed position, under finite aperture, physical path loss, noise, and map error.

Use Sionna RT 2.2.0 for the actual propagation environments and Sionna PHY 2.2.0/PyTorch for received-signal processing. Perform ray tracing and the main signal computations on the H100. Docker is mandatory.

The purpose is to establish or reject a **conditional physical mechanism**. A successful implementation may produce a negative or inconclusive scientific result. Do not tune the simulator to force separation.

Do not implement an LLM agent, multi-agent arbitration, reinforcement learning, an OAI deployment, USRP integration, a communications scheduler, continuous precoder optimization, or a complete manuscript in this phase. Do not invent throughput results: this pilot contains a known OFDM probe, not an implemented multiuser communications system.

### Required claim boundaries

- A digital twin supplies prior information about useful reflection paths; it does not create an information advantage when the observations are intrinsically identical.
- A method without a map can search for a reflector or learn environmental structure. Do not claim that all map-free adaptive methods necessarily fail in a multipath environment.
- A finite beam does not deposit zero energy everywhere outside its intended ray. Compute the relay's actual received signal through all enabled propagation paths.
- An absent predicted echo is not, by itself, proof of an attack. Genuine-target blockage and map error can also remove that echo.
- Fresh random probe symbols do not eliminate a relay that causally forwards those same symbols. Process the correct absolute delay and the complete transmitted waveform.
- Any positive result must state the relay model, geometry, bandwidth, power, map uncertainty, and measurement budget for which it holds.

## 2. Execution feasibility and hardware contract

H100 execution is technically supported by the Sionna/Mitsuba/Dr.Jit CUDA stack. NVIDIA's Sionna large-radio-map project explicitly discusses running on H100-class non-RTX GPUs [R1]. H100 does not provide RTX ray-tracing cores; do not promise RTX-equivalent ray-tracing speed.

Use one full H100 by default. A second H100, if available, may run independent experiment shards. Multi-GPU training and distributed tensor parallelism are unnecessary.

The following are mandatory:

1. Docker GPU visibility through NVIDIA Container Toolkit.
2. A functional CUDA-backed Mitsuba/Dr.Jit ray-tracing workload.
3. PyTorch tensors and signal-processing kernels on the same exposed H100.
4. A recorded, actually measured device identity.
5. A hard failure if scientific execution falls back to CPU.

CPU scene bookkeeping, small independent analytical checks, filesystem access, plotting, and report generation are permitted. Those activities must not be described as H100 numerical execution.

This document is an implementation specification. It does not certify that a container has already run successfully on the researcher's server.

## 3. Container and dependency requirements

### 3.1 Baseline stack

Use this baseline unless an actual compatibility failure requires a documented change:

| Component | Baseline |
| --- | --- |
| Host | Linux with a working NVIDIA driver and NVIDIA Container Toolkit |
| Container OS | Ubuntu 24.04 |
| CUDA base | nvidia/cuda:12.6.3-devel-ubuntu24.04 |
| Python | 3.12 |
| PyTorch | 2.9.1, official CUDA 12.6 wheel |
| Sionna | 2.2.0 |
| Sionna RT | 2.2.0 |
| Mitsuba and Dr.Jit | Resolve according to the pinned Sionna packages; freeze the exact working versions |
| Main numerical precision | complex64/float32 |
| Analytical/reference precision | complex128/float64 where supported and appropriate |

Sionna's documented Python/PyTorch requirements and Docker recommendation are in [R2]. The PyTorch CUDA 12.6 wheel for version 2.9.1 is documented in [R3].

Do not install an NVIDIA kernel driver inside the container. The host driver and its OptiX-related libraries must be exposed by the NVIDIA runtime. Do not disable compatibility checks to hide a failed environment.

### 3.2 Baseline Dockerfile

Implement this Dockerfile, then freeze the resolved dependencies after a successful preflight. If the baseline tag or any dependency is unavailable, record the precise error and choose the nearest compatible, explicitly pinned alternative. Do not silently switch software generations.

~~~dockerfile
FROM nvidia/cuda:12.6.3-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-venv python3-dev \
    build-essential git ca-certificates \
    libgl1 libegl1 libopengl0 \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH
ENV PYTHONUNBUFFERED=1
ENV MPLBACKEND=Agg
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics

RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel
RUN python -m pip install --no-cache-dir \
    torch==2.9.1 --index-url https://download.pytorch.org/whl/cu126
RUN python -m pip install --no-cache-dir \
    sionna==2.2.0 sionna-rt==2.2.0 \
    numpy scipy pandas pyarrow matplotlib pyyaml \
    pytest nvidia-ml-py

WORKDIR /workspace
COPY . /workspace
RUN python -m pip install --no-cache-dir --no-deps -e .
CMD ["python", "-m", "isac_verify", "--help"]
~~~

The final repository must also include a dependency lock or fully resolved freeze file and the base-image digest. The Dockerfile above is the initial bootstrap recipe, not a claim that transitive dependencies are already locked.

Use the installed Sionna RT backend-selection behavior, and verify the selected variant. Do not call Mitsuba variant selection after creating/importing incompatible scene objects. If explicit variant selection is needed, make it part of one tested startup sequence.

### 3.3 Required invocation interface

The completed repository must support these commands from its root:

~~~bash
docker build -t isac-wall-pilot:2.2.0 .

docker run --rm --gpus device=0 \
  --shm-size=8g \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
  -e MPLBACKEND=Agg \
  --mount type=bind,src="$PWD/results",dst=/workspace/results \
  isac-wall-pilot:2.2.0 \
  python -m isac_verify preflight --require-h100 --output results/preflight

docker run --rm --gpus device=0 \
  --shm-size=8g \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
  --mount type=bind,src="$PWD/results",dst=/workspace/results \
  isac-wall-pilot:2.2.0 \
  python -m isac_verify validate --config configs/pilot.yaml --require-h100

docker run --rm --gpus device=0 \
  --shm-size=8g \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
  --mount type=bind,src="$PWD/results",dst=/workspace/results \
  isac-wall-pilot:2.2.0 \
  python -m isac_verify run --config configs/pilot.yaml --stage smoke --require-h100

docker run --rm --gpus device=0 \
  --shm-size=8g \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
  --mount type=bind,src="$PWD/results",dst=/workspace/results \
  isac-wall-pilot:2.2.0 \
  python -m isac_verify run --config configs/pilot.yaml --stage pilot --require-h100

docker run --rm --gpus device=0 \
  --shm-size=8g \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
  --mount type=bind,src="$PWD/results",dst=/workspace/results \
  isac-wall-pilot:2.2.0 \
  python -m isac_verify report --input results --output results/report
~~~

Create the host results directory before these commands. On a shared server, choose the allocated H100 by UUID rather than assuming that host GPU 0 is available. The exposed GPU will normally be device 0 inside a single-GPU container. Do not change host MIG mode or GPU configuration as part of this project.

NVIDIA's documented GPU-selection and container capability mechanisms are in [R4].

### 3.4 Mandatory preflight evidence

Write environment.json and a text log containing:

- GPU name, UUID, total memory, compute capability, and driver version.
- Docker image ID/digest, Python version, platform, and package versions.
- torch.cuda.is_available(), the selected CUDA device, and torch.version.cuda.
- Mitsuba variant; its name must indicate the CUDA backend.
- An actual nonempty Sionna PathSolver result and an actual RCSSolver target return on CUDA.
- One actual CUDA tensor contraction and FFT, followed by synchronization.
- PyTorch peak allocated/reserved memory for a representative signal batch.
- NVML process/device memory observations, clearly distinguished from PyTorch's allocator statistics.

Check that the GPU name contains H100 when --require-h100 is set. Do not infer H100 identity merely from compute capability. Check that CUDA is active in both frameworks; torch.cuda.is_available() alone is insufficient.

A CUDA/OptiX initialization failure is an environment failure. Report it and stop the scientific run. Do not automatically continue with an LLVM/CPU backend. Headless execution must not depend on an interactive renderer or a display server.

## 4. Repository deliverables

Implement a package with these responsibilities. Names below are the required interface; internal decomposition may be extended.

| Path | Responsibility |
| --- | --- |
| Dockerfile | Reproducible container build |
| pyproject.toml | Package, dependency metadata, CLI entry point |
| requirements.lock.txt | Resolved working environment |
| configs/pilot.yaml | All physical, waveform, statistical, and device settings |
| src/isac_verify/__main__.py | preflight, validate, run, and report CLI |
| src/isac_verify/environment.py | GPU checks and environment manifest |
| src/isac_verify/geometry.py | Image-method calculations and candidate geometry |
| src/isac_verify/scenes.py | Separate true and twin Sionna scenes |
| src/isac_verify/channels.py | RCSSolver/PathSolver adapters and absolute-delay handling |
| src/isac_verify/replay.py | Causal multipath relay and forwarded noise |
| src/isac_verify/waveform.py | OFDM generation and continuous-waveform reference |
| src/isac_verify/beams.py | Direct, wall, and map-free scan beams |
| src/isac_verify/receiver.py | Measured-signal processing and candidate estimation |
| src/isac_verify/verification.py | Twin-only prediction, gate construction, and evidence score |
| src/isac_verify/experiments.py | Experiment A/B orchestration and deterministic seeding |
| src/isac_verify/metrics.py | Metrics, paired comparisons, and uncertainty intervals |
| src/isac_verify/report.py | Figures, tables, and scientific decision |
| tests/ | Meaningful physics, causality, normalization, and information-access tests |
| README.md | Exact host/container commands and limitations |

Ship complete implementations. Do not leave pseudocode, TODO computation branches, fabricated arrays, or placeholder result figures. A physically unsupported case must return an explicit status and measured diagnostics.

## 5. Coordinate system and nominal geometry

Use a right-handed Cartesian frame:

- x: down the street.
- y: lateral direction.
- z: height.
- Azimuth 0 degrees points along +x.
- The horizontal ULA extends along y, with its broadside toward +x.
- Keep the Sionna azimuth/zenith conventions separate from application-level azimuth/elevation, and test their conversion.

### 5.1 Nominal physical scene

| Object | Position or dimensions |
| --- | --- |
| Co-located gNB transmitter/receiver phase centers | (0, 0, 2) m |
| Candidate real point target | (60, 0, 2) m |
| Replay source, attack world only | (30, 0, 2) m |
| Positive-y wall | Inner reflecting face at y = +15 m |
| Negative-y wall | Inner reflecting face at y = -15 m |
| Wall extent | x from -10 to 130 m; z from 0 to 12 m |
| Target velocity | Zero in the core experiments |
| Relay velocity | Zero in the core experiments |
| gNB TX/RX arrays | 16-element horizontal ULAs, half-wavelength spacing |
| Antenna element model | Isotropic, one consistent polarization |

Use finite walls with correct inward-facing surfaces. Start with the walls only: no ground reflection, roof, diffuse scattering, diffraction, RIS, or unrelated moving objects. Add such features only in explicitly labeled follow-up configurations.

Use a physically lossy homogeneous wall material. Initial parameters: relative permittivity 5.0, conductivity 0.02 S/m, and zero diffuse-scattering coefficient. Record the exact realized radio material and reflection coefficients at the chosen frequency. These are controlled simulation parameters, not a calibrated claim about a specific real wall.

Use a constant-RCS, nondepolarizing point target with nominal RCS 1 m². This isolates geometry; it does not establish robustness to unknown angle-dependent RCS. Add an RCS sweep later within this pilot's stress suite.

### 5.2 Two hypothesis worlds

For each common street geometry, create:

- **Target world:** A real scatterer is present at the candidate location. No active relay is present.
- **Replay world:** No real scatterer is present at that claimed location. A nearer collinear radio receives and causally retransmits the probe.

The core relay is an idealized point radio. Passive structural scattering from its housing and distinctive transmitter hardware impairments are excluded. State that exclusion; do not accidentally allow hardware artifacts or an artificial body mesh to supply the label.

Use common blank-scene clutter for the primary additive point-target model, and declare this approximation. A separate full-scene check may include target-induced occlusion; do not attribute its differences to the wall-verification mechanism without isolating them.

The **true environment** generates every received sample. The **twin environment** generates only predictions and beam/gate decisions.

### 5.3 Analytical image-method reference

For a wall at y = d, a gNB at (0, 0), and a candidate at (R, 0):

\[
\phi_{\rm wall}(R,d)=\tan^{-1}(2d/R),
\qquad
q_{\rm reflection}=(R/2,d).
\]

For the path with an outgoing wall reflection and a direct return:

\[
L_{\rm wall}(R,d)=\sqrt{R^2+4d^2}+R,
\qquad
\tau_{\rm wall}=L_{\rm wall}/c.
\]

The corresponding causal relay, with a fixed added delay chosen to match the direct-path candidate, has

\[
\delta_{\rm replay}=2(R_1-R_0)/c,
\]

\[
\tau_{{\rm relay,wall}}
=\left[\sqrt{R_0^2+4d^2}+R_0\right]/c+\delta_{\rm replay}.
\]

For the nominal geometry, the following values were computed analytically while preparing this brief:

| Quantity | Analytical value |
| --- | --- |
| Wall departure angle toward the real candidate | 26.565051 degrees |
| Wall departure angle toward the nearer relay | 45.000000 degrees |
| Real candidate's direct round-trip delay | 400.276914 ns |
| Relay's added delay for direct-path matching | 200.138457 ns |
| Real target's outgoing-wall/direct-return delay | 423.900055 ns |
| Near relay's corresponding wall path plus fixed replay delay | 441.726946 ns |
| Difference between the last two delays | 17.826891 ns |

These numbers are not ray-tracing results and are not evidence of successful detection. Verify them independently, then compare with actual Sionna paths.

At 3.5 GHz the 16-element array has an approximate Fraunhofer distance of 9.64 m using its physical linear aperture. The nominal objects are farther away. Experiment A must explicitly retain the common far-field model; do not let near-field curvature create unintended separation.

## 6. Waveform, clocks, power, and noise

### 6.1 Baseline OFDM parameters

| Parameter | Baseline |
| --- | --- |
| Carrier frequency | 3.5 GHz |
| Number of occupied subcarriers/FFT size | 2048, full contiguous research grid |
| Subcarrier spacing | 60 kHz |
| Complex sample rate/occupied bandwidth | 122.88 MHz |
| Cyclic-prefix length | 256 samples |
| OFDM symbol duration including CP | 18.75 microseconds |
| Symbols per beam action | 64 |
| Duration per beam action | 1.2 ms |
| Candidate observation followed by one verification action | 2.4 ms total, excluding recorded computation/turnaround time |
| Probe symbols | Fresh, known, unit-modulus QPSK |
| Nominal total transmitted power | 10 dBm across the entire array |
| BS receiver noise figure | 7 dB |
| Relay receiver noise figure | 7 dB |
| Relay maximum average transmit power | 10 dBm |
| Temperature | 290 K |
| Baseline Doppler | Zero |
| Phase continuity | Explicit and consistent over the complete transmitted record |

This is a research OFDM probe, not a claim of exact NR resource-grid compliance.

Also implement bandwidth profiles 61.44 MHz and 245.76 MHz using FFT sizes 1024 and 4096 at the same subcarrier spacing. Scale CP lengths to 128 and 512, respectively, preserving symbol duration. Compare them at equal transmitted power and observation duration, and report the bandwidth difference.

The raw baseline delay-bin spacing is 8.138021 ns. The nominal wall-delay difference is only about 2.19 raw bins. Window broadening, leakage, and estimation uncertainty may prevent reliable separation. Zero padding does not improve physical resolution.

### 6.2 Absolute clock requirements

Use one absolute transmit-time reference shared by the channel, replay operator, receiver, and evaluator.

For sensing and replay channels:

- Set normalize_delays=False whenever obtaining CIR/CFR/taps.
- Set normalize=False on channel/tap APIs that expose amplitude normalization.
- Do not independently zero the first arrival of each link.
- Do not independently normalize target, clutter, downlink-to-relay, or relay-uplink energy.
- Preserve path loss, RCS scaling, reflection loss, and cascaded propagation delays.
- Convert wall-path delay to total path length, not automatically to a monostatic radial range.

Sionna Paths APIs document delay normalization defaults that would otherwise remove the physical time reference [R5].

### 6.3 Physical power convention

Use beam vectors with a declared total-power convention:

\[
\|\mathbf w\|_2^2=P_{\rm tx}.
\]

With unit-modulus subcarrier symbols and a unitary OFDM implementation, verify average total time-domain transmit power numerically. Log CP energy, finite-record energy, and total illumination energy for every method.

Antenna gains must come from the array/element model. Do not add an extra array-gain factor after applying beam vectors.

### 6.4 Noise convention

Generate proper complex Gaussian receiver noise with a variance derived from kT, the complex sample rate, and the receiver noise figure. State whether each reported variance is before or after FFT, combining, and coherent integration.

Propagate noise scaling through the exact transform convention. Numerically verify the receiver noise floor using noise-only records.

Include relay receiver noise before relay amplification and propagation through the return channel in the physical replay model. Include a separate ideal/noiseless-relay setting for Experiment A's information-equivalence control.

Sweep total probe power over {-20, -10, 0, 10, 20, 30} dBm. Keep the physical channel fixed. An additional controlled-noise sweep is permitted if labeled separately; never obtain an SNR sweep by normalizing each hypothesis channel to unit norm.

## 7. Propagation and replay implementation

### 7.1 Actual Sionna components

Use:

- RCSSolver for target-interacting sensing paths.
- PathSolver for background paths and the gNB-to-relay/relay-to-gNB radio links.
- Paths.cir or the documented equivalent with absolute delays.
- cir_to_ofdm_channel for the validated frequency-domain path model.
- ofdm_to_delay_doppler_channel and angular_delay_doppler_spectrum only after constructing the measured effective response.

The RCS solver supports specular reflection/refraction on the legs around a target interaction [R6]. It does not model a DRFM replay operator for you; implement the relay explicitly.

For Experiment A, restrict both legs to LoS. For Experiment B, retain the relevant first-order wall paths and test convergence with a larger path-depth setting. In RCSSolver's documented convention, the target scattering event contributes to depth. Inspect and export the returned interactions rather than assuming that an integer depth selects only the desired mixed path.

Use explicit solver seeds and deterministic solver construction where supported. Fixed seeds alone do not guarantee identical results for every GPU solver configuration.

No automatic differentiation is needed. The fast default solver mode is acceptable. Do not advertise differentiable ray tracing or gradient optimization as a contribution of this pilot.

### 7.2 Received-signal model

For the frequency-domain approximation, use the shape convention

\[
\mathbf y[n,m]
=\left(\mathbf H_{\rm target}[n,m]+\mathbf H_{\rm background}[n,m]\right)
\mathbf w_b s[n,m]
+\mathbf y_{\rm relay}[n,m]+\mathbf z[n,m].
\]

For a single-antenna relay, the approximate replay contribution is

\[
\mathbf y_{\rm relay}[n,m]
=\mathbf h_{\rm up}[n,m]\,
g\,e^{-j2\pi f_n\delta}\,e^{j2\pi\nu_{\rm add}t_m}
\left(\mathbf h_{\rm down}[n,m]\mathbf w_b s[n,m]+z_{\rm A}[n,m]\right).
\]

All downlink and uplink channels include their enabled multipath and actual absolute delays. Keep their matrix/vector dimensions explicit.

This diagonal OFDM expression is an approximation with CP/Doppler conditions. It must be checked against a continuous time-domain implementation for selected cases.

Do not double-count carrier phase when moving between Sionna CIR coefficients and subcarrier-frequency phase factors. Verify the convention with an analytical single path.

### 7.3 Continuous-waveform reference

Implement a GPU time-domain reference that:

1. Generates CP-extended OFDM symbols as one continuous record.
2. Propagates the record through finite-bandwidth fractional-delay channels.
3. Applies a genuinely causal relay delay, gain, optional frequency shift, and forwarded noise.
4. Propagates the relay output through the physical return channel.
5. Retains symbol-boundary tails and sufficient capture guard time.
6. Processes the complete known waveform with its correct delay alignment.

Linear convolution must not become circular wraparound. FFT-based convolution is allowed with sufficient padding.

Within the valid CP/small-Doppler regime, confirm agreement with the faster frequency-domain model. Outside that regime, flag the diagonal model as invalid and use the time-domain reference.

### 7.4 Attacker capabilities and constraints

The primary threat model is a **single-antenna, causal, scalar, fixed-filter replay source**:

- The relay captures the actual beam-dependent waveform at its physical position.
- It uses one complex gain and one added delay during the two-action verification episode.
- It cannot transmit before receiving the relevant fresh waveform.
- It has a declared maximum average transmit power.
- It has no independently generated perfect target waveform.
- It is allowed to hear all actual propagation paths, including wall paths and sidelobes.

Nominal relay processing latency floor: 20 ns as an idealized model parameter, not a statement about existing hardware. The nominal total added delay of about 200 ns is feasible under that assumption. Reject geometries where direct-path matching would require a delay below the latency floor.

Calibrate the relay's complex gain once against the direct-path candidate return to construct a difficult matched case. This is an evaluator-side attack construction; it must not be visible to the planner. Freeze this gain for the wall action. Log the gain and actual relay transmit power. If its power cap is exceeded, mark the requested matching attack infeasible rather than rescaling the target.

Required stress diagnostics:

- Sweep fixed relay gain by {-10, 0, +10, +20} dB relative to the feasible direct-matched setting, subject to the power cap.
- Compare ideal relay noise and physical forwarded noise.
- Calculate whether a relay that can change its added delay between actions could align a received wall path with the verifier gate while satisfying causality.
- Calculate the gain change required for such a stronger relay to match wall-return energy.

These last two are vulnerability diagnostics, not claims that the primary fixed-filter relay can adapt instantly. If stronger adaptive replay is not simulated, mark it untested and do not claim general replay resilience. Do not obtain a positive result by assuming that a relay receiving small nonzero power simply cannot retransmit.

Random-symbol coding is allowed to prevent substitution with an unrelated stale recording. It must not falsely suppress an honestly delayed copy of the current waveform just because it crossed an OFDM-symbol boundary.

## 8. Information separation and scene uncertainty

### 8.1 Planner inputs

The planner may receive only:

- The imperfect twin scene.
- The noisy direct-observation candidate estimate and its estimated uncertainty.
- Known transmitted symbols and beam settings.
- Fixed calibration parameters and the declared map-uncertainty model.
- Measurements acquired before the current decision.

The planner must not receive:

- True target or relay coordinates.
- Ground-truth path identities or true path delays.
- The true wall offset relative to the twin.
- Hypothesis labels, evaluator seeds, or a true-scene object reference.
- A replay-only gain/delay supplied by the evaluator.

Simulation/evaluation modules may use truth to generate and score trials. Do not pass that truth through planner DTOs, filenames, cache keys used by the decision algorithm, or global variables.

Add a regression test: holding the observed data and twin fixed while changing hidden truth metadata must not change the planned beam or receive gate.

### 8.2 Imperfect map construction

For each fixed true geometry, generate independent twin wall-normal offsets:

\[
\widehat d_{\pm}=d_{\pm}+\epsilon_{\pm},
\qquad
\epsilon_{\pm}\sim\mathcal N(0,\sigma_{\rm map}^2).
\]

Sweep sigma_map over {0, 0.25, 0.5, 1.0, 2.0} m.

Translate whole wall surfaces coherently. Do not jitter mesh vertices independently or create invalid/nonplanar walls. Preserve wall dimensions and reject physically invalid corridor layouts through a declared rule.

Also include a material-error sweep using multiplicative relative-permittivity/conductivity changes of ±20%. Keep it separate from geometry error before combining them.

The planner knows sigma_map, not its realized error. Build beam and delay uncertainty from the distribution of the wall offset and measured candidate uncertainty. Do not widen a gate after inspecting true test-set paths.

Primary uncertainty predictions must use magnitudes, power ratios, angles, and delay distributions. Meter-scale map error generally invalidates a coherent carrier-phase match; do not compare predicted and observed absolute complex phase as if the map were exact.

### 8.3 Unknown blockers

Include a small labeled stress set where a physical absorber blocks the planned wall-to-candidate or candidate-to-gNB segment and is absent from the twin.

Place an actual obstacle and recompute propagation. Arbitrary deletion of selected target paths is permitted only as an explicitly labeled synthetic ablation.

For a blocked genuine target, lack of a wall return should produce absent support or an unresolved case, not an asserted attack label.

## 9. Beam design and candidate acquisition

### 9.1 Direct observation

Transmit a normalized direct beam, observe the received signal, and estimate the candidate delay and arrival angle from measurements.

For the nominal controlled test, a fixed surveillance sector is allowed. Its bounds must be chosen before looking at labels and must be the same for both worlds.

Compute the candidate position hypothesis from the **estimated direct-path** delay/angle:

\[
\widehat{\mathbf p}
=\mathbf p_{\rm gNB}+(c\widehat\tau_{\rm D}/2)\,\widehat{\mathbf u}.
\]

Keep the nominal scenario horizontal. Additional elevation ambiguity is outside the first pilot.

Do not substitute the true candidate coordinates for this estimate in measured-performance runs. An exact-coordinate diagnostic may be reported separately as an oracle upper bound.

If no candidate is detected, record acquisition failure. Do not remove such trials silently from end-to-end verification success rates.

### 9.2 Twin-guided wall prediction

For each candidate:

1. Insert a hypothetical constant-RCS scatterer at the estimated candidate position in a copy of the twin.
2. Use the image method to identify finite, visible first-order reflection candidates.
3. Confirm those candidate paths with the twin's RCSSolver.
4. Select the predicted strongest eligible outgoing-wall/direct-return path using only the twin.
5. Record predicted AoD, AoA, total delay, loss, uncertainty, and path availability.

Reject a reflection if its specular point lies outside the finite wall or its geometry is blocked in the twin. Require a configurable minimum departure-angle separation from the direct direction; default 10 degrees.

If no usable path exists, return NOT_OBSERVABLE. Report its rate. Do not search the true scene for an alternative.

### 9.3 Wall beam and finite-aperture leakage

Define a transmit steering convention v(phi) such that physical directional gain is |v(phi)^H w|². Reconcile this with Sionna's array-response convention in a tested adapter.

For the exact-map diagnostic, form the nominal wall beam by projecting the wall steering vector away from the direct steering vector:

\[
\mathbf P_\perp
=\mathbf I-\frac{\mathbf v_0\mathbf v_0^{\rm H}}
{\mathbf v_0^{\rm H}\mathbf v_0},
\qquad
\mathbf w_{\rm W}
=\sqrt{P_{\rm tx}}\,
\frac{\mathbf P_\perp\mathbf v(\widehat\phi_{\rm W})}
{\|\mathbf P_\perp\mathbf v(\widehat\phi_{\rm W})\|_2}.
\]

For the robust beam:

1. Generate 257 deterministic uncertainty samples from the declared twin/candidate uncertainty model, using fixed quantile or Sobol sampling and recording the construction.
2. Compute each predicted wall departure direction.
3. Align steering-vector phases at a common reference element.
4. Average the steering vectors using equal nonnegative weights.
5. Project the average with P_perp and normalize to the same total power.

This is a fully digital amplitude-and-phase beam. No constant-modulus analog hardware claim is made.

If the projected norm is too small, return an unusable-beam status. Do not use hidden true geometry to repair it.

Measure and save the complete beampattern, predicted gain over uncertainty samples, direct-direction suppression, actual relay illumination, sidelobe leakage, and target illumination. The averaged beam is not guaranteed to be uniformly broad or optimal; verify its realized pattern.

Do not add an exact null toward a hidden true relay angle. The only null direction available to the planner is the measured direct candidate direction.

## 10. Receiver and verification statistic

### 10.1 Effective-channel processing

The first pilot uses one known transmitted stream per beam action:

\[
\mathbf x[n,m]=\mathbf w_b s[n,m].
\]

Within the validated diagonal OFDM regime, form

\[
\widehat{\mathbf h}_{\rm eff}[n,m]
=\frac{\mathbf y[n,m]s^*[n,m]}{|s[n,m]|^2+\varepsilon}.
\]

For unit-modulus probes, this estimates the beamformed effective response, not the complete transmit-array MIMO channel.

Pass that response through the delay-Doppler transform, with the declared window and scaling. For angular processing, use the actual receive-array steering vectors. If using angular_delay_doppler_spectrum, represent the collapsed transmit dimension as one virtual transmit antenna with a unit steering vector. Do not claim to recover a full AoD spectrum from one beamformed stream.

The detector must receive target plus clutter plus replay, when present, plus noise. Ground-truth path masks are allowed for diagnostic plots only.

Static clutter cancellation, if implemented, must use a recorded blank-scene calibration observation available to all methods. Do not subtract a noiseless ground-truth clutter tensor from one method only.

### 10.2 Gates

Use the measured direct candidate to construct the direct reference gate.

Use only the twin uncertainty ensemble to construct the wall delay/angle gate. Use its approximate central 99% model prediction interval, expanded by the **measured processing point-spread width** and candidate-estimation uncertainty. Check its actual coverage on calibration geometries; any coverage correction must be fixed before held-out evaluation. The ensemble quantile is a model-based gate, not a demonstrated 99% coverage guarantee.

For a one-wall outgoing path, gate around the predicted total delay and the candidate's predicted direct-return arrival direction. Do not center the wall gate using a true path from the evaluator.

Keep the gate fixed during the verification observation. Record the number of angle/delay/Doppler cells searched. Wider gates require their own calibrated noise/false-support threshold.

### 10.3 Evidence score

Measure noise-corrected direct and wall gate energies E_D and E_W. Estimate noise from receiver-side calibration or guard cells, not from truth labels.

Use the twin to predict the wall-to-direct power ratio r_DT under the same beams and processing. Form the observable normalized ratio

\[
S
=\frac{E_{\rm W}-\widehat N_{\rm W}}
{\max(E_{\rm D}-\widehat N_{\rm D},\varepsilon)}
\frac{1}{\widehat r_{\rm DT}}.
\]

Do not interpret S as a Bayesian posterior or a proof of authenticity.

If the direct denominator is below its calibrated reliability floor, or r_DT predicts an undetectable reference path, return unresolved status. Do not let arbitrary clipping create confident support.

On calibration data, set the admissible energy/ratio band and an absolute wall-evidence floor. Freeze thresholds before held-out evaluation. Report full score distributions and an operating-characteristic curve, not only one chosen threshold.

Unknown bistatic RCS can change r_DT. Begin with constant RCS, then perturb the wall/direct scattering ratio independently by ±6 dB. This stress test is required before generalizing beyond the controlled point-scatterer model.

### 10.4 Allowed decisions

Use only:

- SUPPORTED: the measured additional return supports the physical candidate hypothesis within the specified model.
- UNRESOLVED: insufficient, unavailable, or inconsistent support.
- ACQUISITION_FAILED: no direct candidate was obtained.

Do not label UNRESOLVED as a detected attacker.

For the measured target-support task, report how often an attack world is incorrectly SUPPORTED and how often a true target is SUPPORTED by the deadline. If a future controller defers action, count failure to support a genuine target before the deadline as lost sensing service.

## 11. Experiment A — matched LoS negative control

### 11.1 Physical assumptions

Remove both walls and all background scatterers. Use the same far-field array model for both hypotheses.

A real scatterer at R1 = 60 m is compared with a relay at R0 = 30 m. Choose the relay's fixed added delay and complex gain so that its apparent direct-path delay, Doppler, and noiseless receive response match the target.

The equality is conditional on the idealized relay/point-target model. It is not a claim that every physical relay has identical hardware noise or distortion.

### 11.2 Required checks

1. Construct the analytical matched rank-one channels.
2. Apply the direct beam, angular dithers at {-5, -2, 0, +2, +5} degrees, and 64 seeded random unit-power beam vectors.
3. Verify equal effective channels for every beam, not just equal peak locations.
4. Reproduce the comparison using actual Sionna LoS channels with the declared far-field treatment.
5. Add matched receiver noise for a paired numerical check, then independent identically distributed noise for statistical evaluation.
6. Show that the measured ratio of responses across beam actions does not provide label information.

Analytical equality tolerances: relative error no greater than 1e-10 in complex128 and 1e-5 in complex64 when amplitudes exceed the numerical floor.

For the Sionna comparison, declare a justified tolerance after checking conventions and precision; begin with 1e-3 relative error. A larger discrepancy must be investigated. Do not explain away a mismatch without checking near-field effects, carrier phase, normalization, fractional delays, and element patterns.

Optionally train a small diagnostic classifier on the beam-response features. If used, split by random realization and report its held-out AUC interval. Chance-level behavior is expected for the exactly matched control; an arbitrary empirical AUC cutoff is not itself a proof.

A reliable discriminator in this matched control is a warning of an implementation difference or leakage, unless an additional physical observable is explicitly included.

## 12. Experiment B — actual wall-assisted evidence

### 12.1 Exact-map path verification first

Before Monte Carlo:

1. Build the nominal finite-wall true scene.
2. Confirm the outgoing-wall/direct-return target path exists.
3. Compare its reflection point, AoD, AoA, and delay with Section 5's image-method values.
4. Generate the wall beam using the twin, initially with zero map error.
5. Obtain the actual target-world receive waveform.
6. Obtain the actual relay-world waveform through the relay's downlink and uplink multipath.
7. Apply the same receiver, gate rule, and noise/power conventions.

Export a path-level diagnostic table for both worlds. Keep this table inaccessible to the planner.

Do not retain only the desired target path in the measured observation. Include all enabled target multipath, relay multipath, and background clutter. A path-isolated run is a labeled diagnostic, not the primary result.

Check actual finite-beam excitation at the relay and include its amplified output. Never implement the attack world as a manually deleted wall echo.

### 12.2 Imperfect-map evaluation

Run the sigma_map sweep using the true geometry unchanged within each paired trial. Both the predicted beam and gate must come from the perturbed twin and measured candidate.

Measure whether true target support remains separable from the fixed-filter relay response when:

- The beam is narrow versus uncertainty-broadened.
- Probe power and bandwidth change.
- Relay gain increases within its cap.
- The wall/direct RCS ratio changes.
- Material properties are wrong.
- A genuine target's expected wall path is blocked.

No specific robustness at 0.5–1 m is assumed. Those values are stress settings to be measured.

### 12.3 Minimum comparison set

Implement these pilot comparisons:

| Method | Information and action |
| --- | --- |
| Direct observation only | No extra wall action; reports acquisition and direct features |
| Exact-map wall probe | Oracle-map diagnostic, not the deployable primary method |
| Imperfect-map narrow wall probe | One extra action using nominal twin geometry |
| Imperfect-map robust wall probe | One extra action using the uncertainty ensemble |
| Map-free equal-budget scan | No map; searches a declared departure-angle codebook |

For the map-free scan, use projected beams on {-60, -50, -40, -30, -20, -10, +10, +20, +30, +40, +50, +60} degrees relative to the candidate direction, subject to the same array visibility limits.

Provide:

- A two-beam map-free control that chooses one angle from this set using a fixed seeded rule.
- A coarse-to-fine map-free search that uses at most the same 64 extra OFDM symbols: four 8-symbol exploration looks at {-50, -30, +30, +50} degrees, then two 16-symbol refinement looks. Choose refinement directions as the nearest ±10-degree neighbors of the best exploration direction, within bounds. The best direction is selected using measured excess return energy in a common, predeclared post-direct-delay search interval.
- A total-energy/time comparison that includes every exploration symbol.

The map-free search is an initial information-cost baseline, not a universal best adaptive algorithm. Its wider delay search needs separately calibrated thresholds. If no common, reliable authenticity decision rule can be justified for this baseline, compare discovery cost and measured evidence and clearly limit the conclusion.

A later publication-level claim that the twin outperforms adaptive probing requires a stronger tuned map-free method and a passive twin-consistency method. The two physical experiments alone do not establish that novelty claim.

## 13. Statistical protocol and reproducibility

### 13.1 Staged runs

**Validation stage:** deterministic analytical/RT checks, environment checks, and selected time-domain comparisons.

**Smoke stage:** nominal geometry, sigma_map in {0, 0.5, 1.0}, both worlds, all core beam settings, 16 noise records per case. Verify data integrity and produce initial figures. Do not infer statistical significance from this stage.

**Pilot stage:**

- Use 12 calibration geometries and 24 distinct held-out geometries.
- Draw candidate distance uniformly from 50–70 m, relay distance from 20–40 m while requiring a feasible latency margin, and true corridor half-width from 12–18 m.
- Keep target and relay collinear for the core evaluation.
- Use all five map-error settings.
- Use four independent map-error realizations per geometry/setting.
- Use 32 independent noise/probe realizations per world and case.
- Cache compact path data once per physical geometry; do not ray trace again for every noise record.
- Run the primary statistical comparison at 10 dBm, baseline bandwidth, and feasible direct-matched fixed relay gain.
- Perform the power/bandwidth/gain/material/RCS/blockage stress sweeps one factor at a time before a selected combined stress set.

Write explicit seed lists to the config/manifest. Use separate random streams for physical geometry, twin error, probe symbols, receiver noise, relay noise, and diagnostic beam selection.

### 13.2 Equal-budget comparisons

Report both:

1. Conditional verification given successful direct acquisition.
2. End-to-end physical target support, including acquisition failure and unavailable wall paths.

Match total transmit energy and observation duration. Additional calibration, map computation, and search cost must be reported separately. Do not hide them by placing them outside the measured loop.

Use the same physical geometry and suitable common random numbers for paired method comparisons. Do not reuse identical random values across training/test partitions or expose evaluator seeds to decision logic.

### 13.3 Uncertainty

Report 95% uncertainty intervals. For held-out aggregate comparisons, resample at the geometry level so repeated noise records from one geometry are not treated as independent environments. With only 24 test geometries, describe the result as a pilot rather than a tight population guarantee.

Choose operating thresholds on calibration geometries only. As a pilot target, aim for at most 5% attack-world support on calibration, then report the actual held-out rate and interval. Do not claim a validated 1% security guarantee from this dataset.

Report robustness per geometry and per map-error setting; an aggregate mean must not hide a subset of consistently failed geometries.

## 14. Metrics and required outputs

### 14.1 Core scientific metrics

- Direct acquisition success.
- Target SUPPORT rate by the 2.4 ms observation deadline, with computation/turnaround latency reported separately.
- Attack-world false SUPPORT rate.
- UNRESOLVED/NOT_OBSERVABLE rates.
- Conditional and end-to-end performance.
- Direct and wall echo power, measured pre/post-integration SNR or SCNR, and exact definitions.
- Wall/direct energy ratio and normalized evidence score.
- Beam gain at relevant true directions, direct leakage, and relay receive/output power.
- True versus predicted wall delay/angle error.
- Gate width, number of searched cells, and noise contribution.
- Total observation time, transmitted energy, and relay power-cap feasibility.
- Stronger-relay delay/gain rematching feasibility.

These are support metrics, not an automatic attack-detection probability.

### 14.2 Runtime and H100 metrics

Separate:

- Initial JIT/compilation time.
- True-scene ray-tracing time.
- Twin prediction time.
- Beam/gate planning time.
- Waveform/channel propagation time.
- Receiver processing time.
- Data transfer and serialization time.
- Warm steady-state time and full end-to-end time.

Synchronize the appropriate framework/device before reading timers. CUDA events can time PyTorch work on the measured stream; they do not automatically time Dr.Jit work on another stream. Use synchronized wall-clock measurements for cross-framework stages.

Record batch sizes, PyTorch allocator memory, NVML memory, and GPU utilization samples. Do not claim that NVML memory equals PyTorch tensor memory.

Keep path/channel tensors on the H100 through the main calculations. Transfer compact metrics for CPU statistics/plotting. If memory pressure occurs, reduce batch size and log it; do not switch to CPU.

Avoid repeated scalar extraction, per-subcarrier Python loops, and a fully materialized batch × RX × TX × symbols × subcarriers tensor when path factorization/effective-channel contraction is sufficient.

Do not require FP16/BF16 or tensor-core use. Phase-sensitive beam nulls and complex channel processing should use the validated complex64 pipeline.

### 14.3 Files

Write machine-readable files:

- environment.json
- resolved_config.yaml
- seed_manifest.json
- dependency_versions.txt
- geometry_reference.json
- paths_diagnostic.parquet
- trials.parquet
- aggregate_metrics.csv
- thresholds.json
- timing.json
- scientific_decision.json
- report.md

Store compact representative waveforms/path tensors in NPZ or a documented PyTorch format. Keep sufficient information to regenerate each figure. Do not store only rendered plots.

Every trial row must include geometry ID, split, hypothesis, method, map sigma, hidden realized map offset in evaluator-only output, power, bandwidth, relay setting, acquisition status, path availability, beam/gate parameters, scores, final support status, energy/time, and measured runtime.

### 14.4 Figures

Generate labeled PNG and vector PDF/SVG versions where appropriate:

1. Nominal top-down geometry with true and twin walls, candidate, relay, and relevant reflection paths.
2. Experiment A effective-response differences across all beams.
3. Direct and wall transmit beampatterns, including direct-direction suppression and relay leakage.
4. Predicted versus true mixed-path delays for target and relay.
5. Representative measured delay/angle spectra for both worlds.
6. Score distributions and operating-characteristic curves.
7. Target/attack SUPPORT and unresolved rates versus map error.
8. Power/bandwidth/gain stress results.
9. Warm/cold runtime breakdown and H100 memory usage.

Use physical units and explicit uncertainty intervals. Plot negative results as clearly as positive results.

## 15. Required validation tests

These tests protect scientific conclusions and are part of the deliverable:

1. Image-method geometry matches the analytical nominal values.
2. Steering convention places the main beam at the requested azimuth.
3. The direct-direction projected null is measured numerically.
4. Beam normalization preserves total transmitted power across all methods.
5. A single target's free-space power follows the radar equation with the configured array gains.
6. A radio link's power follows the expected one-way free-space scaling.
7. Absolute sensing and relay delays survive CIR/CFR conversion.
8. The relay's cascaded delay is downlink + processing + uplink.
9. Fractional-delay propagation does not wrap energy before its causal arrival.
10. The OFDM diagonal model agrees with the time-domain reference in its valid regime.
11. Outside-CP and non-negligible-Doppler cases do not silently use an invalid diagonal model.
12. A correctly delayed current probe remains correlated at its actual delay despite random symbols.
13. Experiment A matched channels remain indistinguishable under every tested beam.
14. A finite-wall reflection path exists where the image method predicts it.
15. Missing/blocked genuine wall paths become unresolved evidence.
16. Changing hidden truth metadata cannot change a fixed planner input/output.
17. Peak/gate/threshold selection never uses held-out labels or true path centers.
18. The scientific GPU execution refuses a CPU backend.
19. The generated report can be rebuilt entirely from saved results.
20. Repeating a seeded run agrees within documented GPU numerical tolerances.

A failed physics invariant blocks scientific conclusions. A valid negative scientific result does not count as a software-test failure.

## 16. Decision rules

Separate **implementation validity** from **scientific feasibility**.

### INVALID

Use when a physics invariant, GPU requirement, information boundary, normalization rule, or timing convention is violated. Repair before interpreting separation.

### INCONCLUSIVE

Use when the pilot has insufficient observable target returns, excessive unresolved cases, insufficient geometry coverage, or uncertainty too wide to support a conclusion.

### CONDITIONAL GO

Use only when:

- Experiment A passes its matched-information control.
- Experiment B's additional evidence is produced by actual propagated signals.
- A frozen operating point provides useful held-out target support while keeping attack-world support low.
- A measured advantage remains at sigma_map = 0.5 m or 1.0 m in the predeclared evaluation population. Favorable geometry subsets may be reported as exploratory findings, but cannot replace the original population after results are inspected.
- The advantage is not explained by hidden truth, an unmatched measurement budget, amplitude normalization, impossible relay behavior, or omitted finite-beam leakage.

Before the held-out run, use these provisional engineering targets: target SUPPORT at least 0.80, attack SUPPORT at most 0.10, and NOT_OBSERVABLE no more than 0.20 in the declared unblocked geometry subset. These are project decision thresholds, not guaranteed scientific results or universal security requirements. Report confidence intervals and also report the full population including unavailable paths. If estimates meet the thresholds but uncertainty is too broad, return INCONCLUSIVE.

A conditional GO supports continued research on that stated threat/environment subset. It does not establish publication novelty, general spoofing immunity, or superiority over every map-free method.

### NO-GO FOR THE PRESENT MECHANISM

Use when valid held-out results show inadequate target/relay separation at usable measurement budgets, or when the proposed robustness vanishes under the specified uncertainty/leakage/noise settings.

Report which limitation caused the failure: weak wall return, insufficient angular separation, unresolved delay signatures, map uncertainty, unknown RCS, relay gain, blocked genuine paths, or adaptive-delay vulnerability.

Do not enlarge power, narrow the tested geometry distribution, or change thresholds after inspecting held-out labels and present the result as the original pilot.

## 17. Implementation order

1. Build the container and obtain verified H100 execution in both numerical frameworks.
2. Implement the analytical geometry and establish axis/steering/delay/power conventions.
3. Implement the matched LoS control and pass it before wall experiments.
4. Build the nominal finite-wall Sionna scene and verify actual path geometry.
5. Implement the physical fixed-filter multipath replay.
6. Validate selected channel cases against the continuous-waveform reference.
7. Implement measured candidate acquisition and a planner restricted to the twin.
8. Implement nominal and uncertainty-averaged wall beams and fixed prediction gates.
9. Run the smoke stage and diagnose validity failures.
10. Freeze the pilot configuration, calibration procedure, and held-out seeds.
11. Run the held-out pilot and one-factor stress suite on H100.
12. Export all measured results and issue INVALID, INCONCLUSIVE, CONDITIONAL GO, or NO-GO.

Stop expansion into agents, arbitration, or a full paper until these physical results are available.

## 18. Definition of done

Return the complete repository and:

- The exact successful build/run commands.
- A verified environment manifest showing actual H100/CUDA use.
- A full resolved dependency record and image digest.
- Passing scientific validation checks, or specific unresolved validity failures.
- Experiment A's numerical equivalence evidence.
- Experiment B's measured evidence, error rates, availability, and resource costs.
- Raw metrics, representative signals, figures, and regenerable reports.
- A plain-English assessment of what worked, what failed, and which attacker assumptions matter.
- An explicit distinction between analytical predictions, completed measurements, and untested extensions.

Do not report an H100 execution, a passing test, or a physical advantage unless it actually occurred.

## 19. Official reference links

These sources support environment/API choices. Experimental geometry, operating settings, and decision thresholds in this document are proposed design choices.

- **[R1] H100/non-RTX Sionna execution:** [NVlabs/sionna-large-radio-maps](https://github.com/NVlabs/sionna-large-radio-maps), computing-radio-maps guidance.
- **[R2] Sionna installation and dependency requirements:** [Sionna installation](https://nvlabs.github.io/sionna/installation.html).
- **[R3] PyTorch 2.9.1 CUDA wheel availability:** [Previous PyTorch versions](https://pytorch.org/get-started/previous-versions/), version 2.9.1.
- **[R4] GPU access inside Docker:** [NVIDIA Container Toolkit sample workload](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/sample-workload.html) and [GPU selection/driver capabilities](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/docker-specialized.html).
- **[R5] Absolute-delay and normalization APIs:** [Sionna RT Paths](https://nvlabs.github.io/sionna/rt/api/paths.html).
- **[R6] Target scattering, multipath, and solver settings:** [Sionna RCS API](https://nvlabs.github.io/sionna/rt/api/rcs.html) and [RCS tutorial](https://nvlabs.github.io/sionna/rt/tutorials/RCS.html).
- **[R7] Effective-channel spectrum processing:** [angular_delay_doppler_spectrum](https://nvlabs.github.io/sionna/phy/api/isac/sionna.phy.isac.angular_delay_doppler_spectrum.html) and [ofdm_to_delay_doppler_channel](https://nvlabs.github.io/sionna/phy/api/channel/wireless/sionna.phy.channel.ofdm_to_delay_doppler_channel.html).
- **[R8] Framework interoperability:** [Sionna RT compatibility with other frameworks](https://nvlabs.github.io/sionna/rt/developer/dev_compat_frameworks.html).

---

**First milestone:** Demonstrate a valid LoS negative control and measure whether a physically generated wall return supplies useful additional evidence under the stated fixed-filter replay model. Accept a negative answer if that is what the measurements show.

