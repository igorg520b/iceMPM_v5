# Per-point vertical temperature tracking — implementation specification

Scope: solve the 1D vertical heat-conduction problem carried by each material
point, using the 3-sine-mode reduction (see heat_note.pdf for the derivation).
This spec covers ONLY the temperature machinery: state, spin-up, runtime
update, and profile reconstruction. The mapping from temperature to strength
(brine volume, q_max modification) is a separate later task; this module's
deliverable is the ability to evaluate T(z) at any depth of any point at any
time, cheaply.

Target code: `preparer` (initialization + spin-up), `gplate`/`cplate`
(main MPM solver).

---

## 1. Model

Each material point owns an independent vertical conduction problem over its
ice thickness H (the existing per-point thickness field), z downward from the
surface, 0 ≤ z ≤ H:

    ∂T/∂t = κ ∂²T/∂z²
    T(0,t) = Ts(t)      surface temperature, sampled from the CARRA-derived
                        forcing field at the point's current position
    T(H,t) = Tb         constant, −1.8 °C

New global constants in `SimParams`:

    kappa_ice = 1.1e-6      // m²/s, ice thermal diffusivity
    T_bottom  = -1.8        // °C

### What is actually tracked

The simulation state per point is FOUR NUMBERS: `Ts_old`, `A1`, `A2`, `A3`.
Nothing else is stored or integrated. Everything below explains what those
four numbers mean and how they're advanced.

### (R) — the reconstruction formula (a read-out, not a state update)

Given the CURRENT values of Ts, A1, A2, A3 at some instant, the full vertical
temperature profile at that same instant is

    T(z) = Ts + (Tb − Ts)·z/H + Σ_{n=1..3} A_n·sin(nπz/H)              (R)

This is pure algebra — plug in the four current numbers and any depth z,
get a temperature. It does **not** advance anything in time. It is only
evaluated when a profile is actually needed (validation plots, or later, the
strength module sampling specific depths). The main spin-up / runtime loop
below never needs to call it.

### r — the local, per-point, per-hour warming rate

Temperature forcing arrives as two rasterized frames (frame0, frame1) per
CARRA interval. Each point samples the SURFACE temperature at its own
position from those frames (same spatial interpolation as wind). Two
neighboring points can see different r if the field isn't spatially uniform.

Across one thermal step of duration dt_th (nominally 1 hour):

    Ts_new = frame-blended surface temperature sampled at the point's
             current position, at the end of this step
    r      = (Ts_new − Ts_old) / dt_th

r is treated as constant for the duration of this one step — this is what
makes the closed-form update below exact (not approximate) within the step.

### (U) — the update formula (advances A_n by one thermal step)

The mode-n amplitude obeys dA_n/dt = −A_n/tau_n − b_n·r, with

    tau_n = H² / (kappa_ice · π² · n²)      // mode's own decay time
    b_n   = 2 / (n·π)                        // same for all n

Holding r fixed over the step, this ODE has an exact closed-form solution.
(U) is that solution evaluated at the end of the step — i.e. the new value to
store, given the old value and this step's r:

    A_n ← (A_n + r·b_n·tau_n) · exp(−dt_th/tau_n) − r·b_n·tau_n         (U)

Apply (U) three times per thermal step — once per mode (n=1,2,3), each with
its own tau_n and b_n, all three using the SAME r (one warming event drives
all three modes). No sub-stepping is ever needed, regardless of how large or
small dt_th is, because (U) is exact, not an Euler approximation.

**NOTE on parenthesization**: the typesetting of Eq. (12) in heat_note.pdf is
ambiguous; the grouping above — `(A_n + r·b_n·tau_n) · exp(...)  −  r·b_n·tau_n`
— is the correct one. Implementers should not re-derive this from the PDF
rendering; use the formula as written here.

**Optional, for consumers wanting a mid-step snapshot** (not needed by the
spin-up or runtime loop, provided for completeness/validation only): within a
single step, holding r fixed, A_n varies continuously as

    A_n(t) = A_eq + (A_n(t_start) − A_eq)·exp(−(t−t_start)/tau_n),
    A_eq = −r·b_n·tau_n

(U) is this formula evaluated at t = t_start + dt_th. Across steps, r changes
at each step boundary, so the full trajectory of A_n over time is a sequence
of such exponential segments glued end-to-end (continuous, but with a slope
kink at each step boundary where r changes) — this shape is never
constructed explicitly; only the value at the current step's end is ever
stored, as one register per mode.

## 2. New per-point state (add to `PtArrIdx`, particle SOA)

    idx_A1, idx_A2, idx_A3   // sine-mode amplitudes, °C
    idx_Ts_old               // surface temperature at the previous thermal
                             // step, °C

Four scalars. Nothing else. H is the existing thickness field; kappa_ice and
T_bottom are global. These fields travel with the point through the existing
SOA sort/transfer automatically, provided transfers copy the full pitch.

## 3. New grid forcing data

Scalar surface-temperature field, rasterized to the grid exactly like the
wind components (same CARRA source, same projection/rotation-free scalar
path, same double-buffer frame mechanism in `bgrid_forcing` /
`GridForcingFramesIndex`):

    grid_idx_temp_frame0
    grid_idx_temp_frame1

One scalar per frame (not a 2-vector). Blending factor `current_alpha_temp`
passed to the kernel alongside `current_alpha_wind`; if the temperature
frames share the wind frames' cadence (both CARRA), the two alphas will be
equal — verify and unify only if the frame boundaries genuinely coincide.
Halo handling: identical to the existing forcing planes; no new
communication.

## 4. Runtime thermal step (`gplate`/`cplate`)

Cadence: every `N_thermal` mechanical steps with N_thermal·dt ≈ 1 hour
(≈ 63,000 steps at dt = 0.057 s). New `SimParams` integer, gated like the
existing Glen-flow stride:

    if (step % gprms.N_thermal == 0) { ... }

Placement: inside `partition_kernel_g2p` (the stencil weights `ww` and
`cell_i` are already computed there) or as a separate rarely-launched kernel
— implementer's choice; the gather must use the same 3×3 weighted-stencil
interpolation as the wind sampling.

Per point, at each thermal step (this is one iteration of the loop in §6,
just triggered by mechanical-step count instead of a spin-up date range):

    1. Ts_new = frame-blended temperature gathered at the point's CURRENT
       position (weights ww, blend with current_alpha_temp). Sampling at
       the moving position is what makes the material derivative automatic
       — do NOT add any advection term.
    2. dt_th = N_thermal * dt          // seconds, ≈ 3600
       r = (Ts_new − Ts_old) / dt_th
    3. for n = 1,2,3: apply (U) with tau_n, b_n computed from H (§1)
    4. Ts_old ← Ts_new

Skip disabled points (`status_disabled`), same as the mechanical kernels.
Between thermal steps the four-number state is simply carried; nothing is
evaluated.

## 5. Profile reconstruction (the module's output interface)

Provide a device function

    __device__ double reconstruct_T(double z, double H, double Ts,
                                    double A1, double A2, double A3)

implementing (R) from §1. Downstream consumers (the future strength module)
call it at quadrature depths of their choosing. Notes for consumers, to be
kept as comments: the sine terms vanish at z=0 and z=H (values there are
exactly Ts and Tb); interior evaluation should use midpoints of equal
sub-layers.

This spec deliberately does NOT define what is computed from the
reconstructed profile. No brine volume, no strength formula, no multiplier —
that is a separate, later task, and its formula choice is undecided.

## 6. Spin-up algorithm (`preparer`)

**Purpose.** Initialize A1, A2, A3, Ts_old per point so the mechanical run
starts with the interior thermal lag actually accumulated over the pre-event
warming, instead of starting from the equilibrium (straight-line) profile.

**Why points can be held fixed.** Motion enters the thermal problem only
through where Ts is sampled each step. The ice cover is observed stationary
before breakup onset, so freezing particle positions at their seeding
location for the whole spin-up window is a valid approximation — no
mechanical time-stepping, no stress, no advection of the thermal state.

**Per-point pseudocode:**

    // Initialization at the start of the lead-in window (e.g. May 1)
    A1 = A2 = A3 = 0
    Ts_old = sample_surface_temperature(x_p, t = window_start)
        // sampled AT THE POINT'S POSITION from the CARRA raster — do not
        // hardcode a single value; the field varies across the domain

    // Hourly loop from window_start to simulation_start (e.g. May 1 -> Jun 9)
    for each hourly frame boundary t_i in [window_start, simulation_start):
        Ts_new = sample_surface_temperature(x_p, t_i + 1 hour)
        dt_th  = 3600.0                       // seconds
        r      = (Ts_new − Ts_old) / dt_th

        for n in {1, 2, 3}:
            tau_n = H*H / (kappa_ice * pi*pi * n*n)
            b_n   = 2.0 / (n * pi)
            A_n   = (A_n + r*b_n*tau_n) * exp(-dt_th/tau_n) - r*b_n*tau_n

        Ts_old = Ts_new

    // End of loop: (Ts_old, A1, A2, A3) is the initial state handed to
    // gplate/cplate. No profile reconstruction (R) is needed during this
    // process — it is pure bookkeeping of the four numbers.

**Iteration count.** ~700–1000 hourly steps depending on the chosen window
length (see below), independent of mechanical dt. Trivial cost at any point
count — run this per point, not per grid cell (see next paragraph for why).

**Do not precompute per grid cell.** heat_note.pdf suggests precomputing
amplitudes once per forcing-grid cell and copying them to points. This is
only valid if every point sharing a cell also shares H, since tau_n depends
on H. Since thickness varies point-to-point, per-point spin-up is simpler and
still cheap enough — do not implement the per-cell shortcut.

**Window length.** Must exceed the slowest relevant timescale — tau_1 of the
thickest ice in the domain — by a comfortable margin, so the arbitrary
A_n = 0 initial condition has decayed out before simulation_start. At H = 3 m,
tau_1 ≈ 10 days; a window of several weeks (e.g. May 1 for a June event)
satisfies this with margin. Exact start date is an open decision (§9).

**Data requirement.** `preparer` needs the CARRA surface-temperature series
for the full lead-in window — same ingestion path as the wind data, just a
longer time range than the mechanical run itself.

## 7. Validation tests

    1. Worked example of heat_note.pdf: Ts piecewise linear through
       (−11 °C, May 1), (−3.3, Jun 1), (−1.5, Jun 6), (−0.11, Jun 11),
       constant after; H = 3 m, Tb = −1.8. Mid-depth (z = 1.5 m) values via
       (R) must reproduce: −3.95 (Jun 1), −3.33 (Jun 6), −2.63 (Jun 11),
       −1.63 (Jun 20) °C, within ±0.02.
    2. Cross-check against an explicit finite-difference solve of the same
       problem (121 nodes): max deviation ≲ 0.01 °C at all depths and times.
    3. Steady-warming limit: constant r > 0 drives A_n → −r·b_n·tau_n
       (interior colder than the instantaneous linear profile).
    4. Limits of (U): dt_th ≪ tau_n reproduces the Euler step;
       dt_th ≫ tau_n lands exactly on −r·b_n·tau_n.
    5. Thickness scaling: same forcing, H = 0.8 m vs 3 m — the thin column
       must track the surface with tau_1 ≈ 0.7 d lag, the thick one ≈ 10 d.

## 8. Rules and edge cases

- H in tau_n and (R) is the point's initial/reference thickness. Dynamic
  thickness change during the run (J_p ≠ 1) is ignored by the thermal model
  in this version.
- Points spawned mid-run (if that path exists): A_n = 0, Ts_old = current
  local forcing value at spawn — i.e. the same initialization rule as §6's
  first line, just triggered at spawn time instead of at window_start.
- Precision: float is sufficient for all four fields (temperatures are O(10)
  and the update is a contraction); match the SOA's existing layout if it is
  uniform double.
- No clamping anywhere in this module: A_n and reconstructed T are stored
  and returned as-is, including transient T ≥ 0 near the surface. Any
  clamping belongs to the downstream consumer of the profile.

## 9. Open decisions (ask, do not resolve silently)

- Thermal step inside `partition_kernel_g2p` vs. its own kernel (§4).
- Unify `current_alpha_temp` with `current_alpha_wind` if cadences coincide (§3).
- Exact lead-in window start date for spin-up (§6).
