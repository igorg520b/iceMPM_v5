#ifndef P_SIM_H
#define P_SIM_H

#include <spdlog/spdlog.h>
#include <spdlog/fmt/fmt.h>

#define LOGR(fmtstr, ...) spdlog::info(fmt::format(fmt::runtime(fmtstr), ##__VA_ARGS__))

#include <iostream>
#include <string>
#include <filesystem>
#include <fstream>
#include <map>
#include <vector>

#include <Eigen/Core>
#include "rapidjson/reader.h"
#include "rapidjson/document.h"
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>

#include "projection.h"

// variables related to the formulation of the model

struct SimParams
{
public:
    constexpr static float disabled_pts_proportion_threshold = 0.05; // when exceeded, re-balance occurs
    constexpr static float free_space_threshold = 0.01; // when crossed, re-balance occurs
    constexpr static double pi = 3.14159265358979323846;
    constexpr static double gravity = 9.81;
    constexpr static double rho_water = 1030.0;
    constexpr static double rho_air = 1.2;

    // Per-point vertical heat-conduction model (see _python_code/heat_equation_spec.md)
    constexpr static double kappa_ice = 1.1e-6;   // m^2/s, ice thermal diffusivity
    constexpr static double T_bottom = -1.8;      // deg C, fixed ocean/bottom boundary temperature

    constexpr static int dim = 2;
    constexpr static int MAX_REGIONS = 255;
    constexpr static int ModelledAreaIndicator = 255;

    // landmask_buffer sentinel for an open boundary (from ImageFootprintMask,
    // black = boundary) -- overlaid with priority over the ordinary
    // land(0)/modelled-area(255) values. Points reaching a cell marked with
    // this value are meant to be disabled (GPU kernel-side; not yet wired up).
    constexpr static int OpenBoundaryIndicator = 128;

    // Status flags (bit masks for utility_data)
    constexpr static uint64_t status_cracked = 0x20000; // 0x10000 (status_crushed) is free

    constexpr static uint64_t status_disabled = 0x40000;

    // status flags for types of fracture (recorded for visualization)
    constexpr static uint64_t fracture_tension = 0x80000; // bit 19
    constexpr static uint64_t fracture_compression_shear = 0x100000; // bit 20
    constexpr static uint64_t fracture_crush = 0x200000; // bit 21 (22 and 23 are free)

    // GPU allocation
    constexpr static double points_transfer_buffer_fraction = 0.07;  // % of points that could "fly over" during a given cycle

    // layout of the grid arrays
    constexpr static int grid_arrays_to_clear = 8;  // at reset_grid, which should be cleared

    enum GPUGridArrayIndex : size_t {
        // --- Persistent Arrays (Group 0) ---
        // These arrays persist across the entire time step and are not overwritten by visualization logic
        gpu_grid_idx_mass = 0,
        gpu_grid_idx_px = 1,
        gpu_grid_idx_py = 2,
        gpu_grid_idx_thickness = 3,

        // Fractured field (cracked/crushed material), two-field contact MPM.
        // Fixed +4 offset from the corresponding intact-field slot above, so
        // kernels can branch once (field_offset = is_fractured ? 4 : 0)
        // instead of four separate ternaries.
        gpu_grid_idx_mass_f = 4,
        gpu_grid_idx_px_f = 5,
        gpu_grid_idx_py_f = 6,
        gpu_grid_idx_thickness_f = 7,
        nGridArraysGPU = 8, // total count for allocation on GPU

        // group 1
        gpu_grid_idx_vis_r = 0,
        gpu_grid_idx_vis_g = 1,
        gpu_grid_idx_vis_b = 2,

        // group 2
        gpu_grid_idx_vis_Jpinv = 0,
        gpu_grid_idx_vis_P = 1,
        gpu_grid_idx_vis_Q = 2,          

        // group 3
        gpu_grid_idx_vis_pts_density = 0,
        gpu_grid_idx_vis_strain_EqvGreenLagrange = 1,
        gpu_grid_idx_vis_strain_vonMises = 2,

        // group 4
        gpu_grid_idx_vis_crushed = 0,
        gpu_grid_idx_vis_cracked = 1,
        gpu_grid_idx_vis_thickness = 2,

        // group 5
        gpu_grid_idx_fracture_tension = 0,
        gpu_grid_idx_fracture_shear = 1,
        gpu_grid_idx_fracture_crush = 2,

        // group 6
        gpu_grid_idx_vis_ice_strength = 0,
        gpu_grid_idx_vis_gamma_p = 1
    };


    enum HostGridArrayIndex : size_t {
        host_grid_idx_mass = 0,
        grid_idx_px = 1,
        grid_idx_py = 2,

        grid_idx_vis_r = 3,
        grid_idx_vis_g = 4,
        grid_idx_vis_b = 5,
        grid_idx_vis_Jpinv = 6,
        grid_idx_vis_P = 7,
        grid_idx_vis_Q = 8,
        grid_idx_vis_strain_EqvGreenLagrange = 9,
        grid_idx_vis_strain_vonMises = 10,
        grid_idx_vis_pts_density = 11,

        grid_idx_vis_crushed = 12,
        grid_idx_vis_cracked = 13,
        grid_idx_vis_thickness = 14,
        
        grid_idx_fracture_tension = 15,
        grid_idx_fracture_shear = 16,
        grid_idx_fracture_crush = 17,

        grid_idx_vis_ice_strength = 18,
        grid_idx_vis_gamma_p = 19,

        nGridArraysHost = 20
    };

    static bool IsPersistentGridArray(int idx);


    // indices in the grid_forcing_buffer to access
    enum GridForcingFramesIndex : size_t {
         grid_idx_current_vx_frame0 = 0,    // ocean current vector field
         grid_idx_current_vy_frame0 = 1,
         grid_idx_current_vx_frame1 = 2,
         grid_idx_current_vy_frame1 = 3,
         
         grid_idx_wind_vx_frame0 = 4,   // wind vector vield
         grid_idx_wind_vy_frame0 = 5,
         grid_idx_wind_vx_frame1 = 6,
         grid_idx_wind_vy_frame1 = 7,

        grid_idx_temperature_frame0 = 8,   // CARRA1 surface temperature in Kelvins
         grid_idx_temperature_frame1 = 9,

         grid_idx_scaling_factor_inverse = 10,  // scaling factor (scalar)

         nGridForcingArrays = 11
    };

    // storage of point data: index of the corresponding array in SoA
    enum PtArrIdx : size_t {
        idx_utility_data = 0,   // flags such as cracked/crushed, also RGB color
        integer_cell_idx = 1,   // two-integer (i,j) index of point's cell
        posx = 2,               // [-0.5, 0.5] local coordinates within cell
        posy = 3,
        velx = 4,               // point's velocity
        vely = 5,
        idx_thickness = 6,      // local ice thickness
        idx_Jp_inv = 7,         // volume change due to plastic deformation

        // Squared projection point-scale-factor (m^2 = Projection::AreaScaleFactor)
        // at the point's seeded location, cached ONCE at generation time (see
        // DataPreparer::PopulatePoints_RAM_Optimized) -- unlike
        // grid_idx_scaling_factor_inverse (recomputed per grid cell, current
        // position), this stays fixed at the point's ORIGINAL location for its
        // whole lifetime.
        idx_m_squared = 8,

        // Per-point vertical heat-conduction state (see
        // _python_code/heat_equation_spec.md, sections 1-2). Sine-mode
        // amplitudes A1..A3 and the previous thermal step's surface
        // temperature -- advanced by preparer's thermal spin-up tool and
        // (later) the gplate/cplate runtime thermal step. idx_xi is
        // reserved for a future material-strength coefficient; no formula
        // for it exists yet, so it is currently just initialized to 0.0
        // and otherwise unused.
        idx_A1 = 9,
        idx_A2 = 10,
        idx_A3 = 11,
        idx_Ts_old = 12,
        idx_xi = 13,

        Fe00 = 14,                   // size 4: deformation gradient
        Bp00 = 18,                  // size 4: grad of v with respect to x,y

        // (p, q) trial-stress values at the exact step a point first
        // transitions from intact to cracked (tension or compression/shear
        // -- either is possible; see partition_kernel_g2p in kernels.cu).
        // Written exactly once, guarded by the intact/cracked state
        // transition itself -- never overwritten afterward. A point that
        // never cracks keeps (0, 0).
        idx_stored_P = 22,
        idx_stored_Q = 23,

        // Accumulated plastic shear strain -- deviatoric counterpart to
        // idx_Jp_inv (volumetric). Initialized to 0 at point generation,
        // incremented in PlasticProjection (kernels.cu) whenever a shear
        // return occurs. Visualization only for now (pt_gamma_p) -- not
        // read by any yield-envelope calculation yet.
        idx_gamma_p = 24,

        // Historical peak of idx_Jp_inv -- compressive counterpart to
        // gamma_p's permanent memory, but for volumetric crushing rather
        // than shear. idx_Jp_inv itself is the point's *current* plastic
        // area ratio and moves both ways (grows under crushing, shrinks
        // again as the point re-expands); idx_Jp_inv_max only ever grows.
        // The gap between the two records material permanently lost to
        // crushing (piled onto neighboring ice or dispersed into the
        // water) -- a re-expanded point should not regain compressive
        // resistance until it is recompacted past its own past peak, not
        // merely back to the original reference value of 1. Initialized
        // to 1.0 at point generation; see PlasticProjection's dispersed-
        // material checks (kernels.cu) for where it replaces the fixed
        // threshold of 1.
        idx_Jp_inv_max = 25,

        nPtsArrays = 26
    };

    // GPU and multi-GPU-related params
    int tpb_P2G, tpb_Upd, tpb_G2P;  // threads per block for each operation
    unsigned nPartitions;           // number of partitions split between GPU devices
    unsigned GridHaloSize;
    unsigned HaloDiffusionThreshold;    // must be <GridHaloSize-1
    unsigned PointTransferPeriod;       // how often do we try to transfer points (~GridHaloSize)
    double extra_space_pts;               // reserved additional space on devices for points

    int nPtsInitial;
    double InitialTimeStep, SimulationEndTime;
    double AnimationFramePeriod;
    int SimulationStep;
    double SimulationTime;
    bool SaveSnapshots;
    int SnapshotPeriod;

    bool SaveAnimationAsJPG;
    int JPG_OffsetX, JPG_OffsetY, JPG_Crop_Width, JPG_Crop_Height;

    // grid
    int GridXTotal, GridYTotal;     // actually used in simulation
    int ModeledRegionOffsetX, ModeledRegionOffsetY;
    int InitializationImageSizeX, InitializationImageSizeY;

    // wind and/or current data
    double waterDragNormalized, windDragNormalized;

    // Synthetic-wind test mode: a spatially-uniform, constant-magnitude,
    // constant-direction wind, for testing the ice response to a known
    // forcing instead of the real CARRA1 record. 0 (default) means "off,
    // use real CARRA1/GLO12 forcing as configured." A nonzero value (m/s)
    // both sets the synthetic wind's magnitude and disables real GLO12
    // ocean forcing entirely (set to zero) -- CARRA1 itself is still loaded,
    // since live ice strength needs its real surface temperature; only its
    // wind component is overridden. Direction is fixed (see
    // WindInterpolator::ComputeDefaultTestDirection) -- the bearing across
    // Nares Strait, rotated into grid-local coordinates once and cached.
    // Uses the same wind-drag formula as real wind (windDragNormalized
    // above) -- nothing about the drag physics itself changes.
    double TestWindSpeed;

    // Linear spin-up: wind/current forcing (v_wind, v_w in
    // partition_kernel_update_nodes) is scaled by min(1, simulation_time /
    // ForcingSpinUpDuration), so both ramp from 0 to full strength over the
    // first ForcingSpinUpDuration seconds of the simulation instead of
    // snapping on at full strength at step 0.
    double ForcingSpinUpDuration;

    // Absolute Unix time (seconds) corresponding to simulation time 0; i.e.
    // simulation_time + SimulationStartTime = absolute time used to look up
    // CARRA1/GLO12 frames. Parsed from a human-readable date in JSON
    // ("SimulationStartDate", e.g. "2024-06-08") -- replaces the old
    // raw-seconds TimeOffset field.
    long long SimulationStartTime;

    // Absolute Unix time marking the earliest reanalysis data of interest --
    // e.g. the start of a CPU pre-simulation that runs on temperatures from
    // before SimulationStartTime (May 1, if the real simulation starts June
    // 8). Parsed from "PreSimulationStartDate". Also used as the lower bound
    // of the preparer's CARRA1/GLO12 visualization slider. 0 (unset) means
    // "same as SimulationStartTime" -- callers should fall back accordingly.
    long long PreSimulationStartTime;

    // Absolute Unix time marking the end of the reanalysis-data span of
    // interest. Parsed from "SimulationEndDate". This is deliberately
    // distinct from SimulationEndTime above (which is an elapsed-seconds
    // duration used by the main GPU simulation loop's termination check) --
    // it is only used as the upper bound of the preparer's data-
    // visualization slider. 0 (unset) means "same as SimulationStartTime".
    long long SimulationEndDateTime;

    // Parses "YYYY-MM-DD" or "YYYY-MM-DDTHH:MM:SS" as UTC and returns Unix
    // epoch seconds. CARRA1/GLO12 timestamps are UTC, so this must NOT use
    // the local-timezone mktime().
    static long long ParseISODateToUnixTime(const std::string &dateStr);

    // Inverse of ParseISODateToUnixTime: formats Unix epoch seconds (UTC) as
    // "YYYY-MM-DDTHH:MM:SS".
    static std::string FormatUnixTimeToISO(long long epoch);

    // CARRA1 Data / GLO12 data. UseWindData is derived, not user-configurable:
    // it's true iff CARRA1Data path(s) were provided (see ParseFile) -- there's
    // no way to load CARRA1 data and suppress wind forcing.
    bool UseWindData;
    bool UseGLO12Data;
    bool UseGLO12Tides;

    // Map projection (stereographic, tangent at a configurable center --
    // see projection.h). Single shared source of truth for projection
    // parameters and math, used by WindInterpolator, CurrentInterpolator,
    // and the GLO12 ice-thickness renderer.
    Projection proj;

    // material properties
    double IceDensity, PoissonsRatio, YoungsModulus;
    double IceTensileStrength, IceShearStrength;
    double IceCompressiveThreshold;     // exceding this causes the material to crush

    // Intact fracture envelope, piecewise-linear (PlasticProjection, kernels.cu):
    // p<0 leg runs from (-IceTensileFailureStrength, 0) to (0, IceShearStrength);
    // p>=0 leg is IceShearStrength + tan(IceFractureAngle)*p. IceTensileStrength
    // itself is a separate, older field no longer read by this curve.
    double IceTensileFailureStrength;
    double IceFractureAngle, IceFractureAngle_tan;   // degrees; tan precomputed once

    // Fractured/yield surface, piecewise-linear (Q_From_Yield_Surface, kernels.cu):
    // steep leg is the existing DP_phi/DP_threshold_p line, up to q_crossover
    // = q_intersect*Cf, where q_intersect is the point where that DP line
    // would otherwise cross the intact envelope's compression-shear leg --
    // so Cf=1 puts the fractured/yield surface's kink exactly on the intact
    // envelope, Cf<1 pulls it in below that (weaker fractured material).
    // Past the kink, a second leg with slope tan(IceYieldFrictionAngle).
    //
    // Cf itself (c_f0 below) softens with accumulated plastic shear strain,
    // gamma_p: c_f(gamma_p) = Cf_residual + (Cf - Cf_residual)*exp(-gamma_p/GammaStar).
    // See the "Cf(gamma_p) softening" block in PlasticProjection.
    double Cf;
    double Cf_residual;   // c_res: Cf's asymptotic value as gamma_p -> infinity
    double GammaStar;     // gamma_star: softening's characteristic strain scale
    double IceYieldFrictionAngle, IceYieldFrictionAngle_tan;   // degrees; tan precomputed once

    // Duvaut-Lions viscous relaxation toward the fractured/yield surface
    // (PlasticProjection's main shear-return branch only -- not tension, not
    // the dispersed/degraded case). TauInv = 1/tau (s^-1); TauInv<=0 is a
    // sentinel for instant projection (no viscosity), not the literal math
    // limit. RelaxationAlpha = exp(-InitialTimeStep*TauInv), precomputed
    // once in ComputeHelperVariables since InitialTimeStep is fixed.
    double TauInv;
    double RelaxationAlpha;

    // When true, the yield-surface strength scale (q0/pmin in PlasticProjection) is derived from the live per-point idx_xi instead of the flat IceShearStrength/IceTensileFailureStrength constants above.
    bool UseLiveIceStrength;
    // pmin = -(idx_xi * IceStrengthFactorTensile) when UseLiveIceStrength is true, instead of -IceTensileFailureStrength.
    double IceStrengthFactorTensile;

    // When false, PlasticProjection (kernels.cu) is never invoked -- material is purely elastic, no plasticity or fracture at all. Default true.
    bool AllowFracture;

    // Runtime switch for the two-field intact/fractured contact MPM scheme (was ENABLE_DOUBLE_GRID, a compile-time flag). Default true.
    bool UseDoubleGrid;

    // Coulomb friction coefficient between the intact and fractured/crushed
    // grid fields at contact (see partition_kernel_contact). Governs the
    // stick/slip threshold: fields stick (move together) when the tangential
    // relative velocity is within IceFrictionCoeff * normal approach speed,
    // otherwise they slip with friction capped at that limit. 0.3 is a
    // defensible starting value for ice-ice; set very high (e.g. 10) to
    // approximate full stick contact as a regression check against the
    // single-field behavior.
    double IceFrictionCoeff;

    double DP_phi, DP_tan_phi, DP_threshold_p;

    double cellsize;
    double ParticleArea;

    // Live ice-strength model (see _python_code/heat_equation_spec.md and
    // partition_kernel_compute_ice_strength in kernels.cu)
    double IceSalinity;             // ppt, for Frankenstein & Garner brine volume
    double IceStrengthFactor;       // multiplier applied before storing into idx_xi
    double IceStrengthFactorCompressive;  // reserved for a fracture-side compressive check; not currently used in PlasticProjection -- see IceStrengthFactorCompressiveYield
    double IceStrengthFactorCompressiveYield;  // xi*Cf_now*this is the yield/post-crack pressure cap in PlasticProjection; see kernels.cu
    double IceStrengthUpdatePeriod; // nominal seconds of sim time between updates

    // computed parameters/properties
    double dt_area_Dpinv, vmax;
    double lambda, mu, kappa; // Lame
    double ParticleMass;
    double cellsize_inv, Dp_inv;
    int UpdateEveryNthStep;
    int IceStrengthUpdateStepInterval;  // round(IceStrengthUpdatePeriod / InitialTimeStep)

    void Reset();
    // Returns additional filenames to load. carra1FilesOut, if non-null, is
    // filled with the (possibly multi-file) "CARRA1Data" path list in
    // chronological order -- this can't be a SimParams member itself since
    // SimParams is copied wholesale to CUDA __constant__ memory, which
    // requires a trivially-constructible type (a std::vector member breaks
    // that -- see simulation/gpu_partition.cpp's update_constants()).
    std::map<std::string,std::string> ParseFile(std::string fileName, std::vector<std::string>* carra1FilesOut = nullptr);

    void ComputeLame();
    void ComputeHelperVariables();
    int AnimationFrameNumber() { return SimulationStep / UpdateEveryNthStep;}

    void Printout();    // for testing
    size_t getHaloElementCount() { return GridYTotal*GridHaloSize*2; }
};

#endif
