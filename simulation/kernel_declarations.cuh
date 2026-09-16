#ifndef KERNEL_DECLARATIONS_CUH
#define KERNEL_DECLARATIONS_CUH

#include "parameters_sim.h"
#include "partition_params.h"

// ============================================================================
// DEVICE STATE - Accessed by all kernels
// ============================================================================

extern __device__ uint32_t gpu_error_indicator;  // Accumulates error codes during kernel execution
extern __constant__ SimParams gprms;             // Simulation parameters (constant memory)

// ============================================================================
// PUBLIC COMPUTE INTERFACE - Main MPM Operations
// ============================================================================


// Piecewise-linear fractured/yield surface for already-cracked material --
// steep DP leg up to q_crossover, shallower second leg beyond it. See
// definition in kernels.cu.
__device__ double Q_From_Yield_Surface(double p_tr, double q_crossover);

// Per-point vertical thermal model (see _python_code/heat_equation_spec.md)
// -- reconstruction formula (R), brine volume, and flexural strength.
__device__ double reconstruct_T(double z, double H, double Ts, double A1, double A2, double A3);
__device__ double brine_volume_frankenstein_garner(double T, double S);
__device__ double flexural_strength_timco_obrien(double vb);

// reconsturction after projection
__device__ void ReconstructDeformationGradient(Eigen::Matrix2d &Fe, double &Jp_inv,
                                               const double &p_new, const double &q_new,
                                               const double &q_tr, const double &Je_tr);

// Piecewise-linear brittle projection. Two separate compressive caps, both live-strength-scaled
// (xi*factor, or a flat gprms.IceCompressiveThreshold fallback if UseLiveIceStrength is off):
// compressive_threshold_fracture cracks still-intact material once p exceeds it, even under low
// shear; compressive_threshold_yield then re-caps p every step post-crack, scaled by Cf_now
// (unlike the fracture one) since worked material holds pressure less well. Shear-return and
// tension branches use Duvaut-Lions relaxation (gprms.TauInv/RelaxationAlpha) toward the yield
// surface and toward 0 respectively; the dispersed/degraded branch stays instant.
// xi: point's idx_xi (live flexural strength, Pa). Used for the intact
// envelope's q0/negT iff gprms.UseLiveIceStrength; otherwise ignored in
// favor of the flat IceShearStrength/IceTensileFailureStrength constants.
// gamma_p: accumulated plastic shear strain, incremented on any shear
// return (not pure tension) -- see kernels.cu.
// Jp_inv_max: historical peak of Jp_inv (permanent, never decreases) -- the
// dispersed/no-resistance checks compare against this instead of the fixed
// reference value of 1, so a point crushed past 1 and later re-expanded
// does not regain resistance until it recompacts past its own past peak.
__device__ void PlasticProjection(unsigned long long &utility_data,
                        const double &p_tr, const double &q_tr, const double &Je_tr,
                        Eigen::Matrix2d &Fe, double &Jp_inv, double &Jp_inv_max, double &gamma_p, const double &dt,
                        const double &xi);

// Particle-to-Grid transfer: distributes particle data (mass, momentum) to grid nodes
// using B-spline interpolation
__global__ void partition_kernel_p2g(const PartitionParams pparams);

// Grid node update: computes grid node velocities and applies forces
// Parameters:
//   - simulation_time: current simulation time (used for wind/current interpolation)
//   - current_alpha: temporal interpolation factor for wind/current data (0=frame1, 1=frame2)
__global__ void partition_kernel_update_nodes(const PartitionParams pparams,
                                              const double simulation_time,
                                              const double current_alpha,
                                              const double current_alpha_wind);

// Two-field contact: resolves velocity interaction between the intact and
// fractured/crushed grid fields (stick/slip Coulomb friction where both
// fields have mass, free separation otherwise). Runs after
// partition_kernel_update_nodes and before partition_kernel_g2p.
__global__ void partition_kernel_contact(const PartitionParams pparams);

// Grid-to-Particle transfer: updates particle velocities and deformation gradients
// from grid velocities; optionally records P (stress) and Q (second stress invariant)
// Parameters:
//   - recordPQ: if true, compute and record stress values for visualization
__global__ void partition_kernel_g2p(const PartitionParams pparams,
                                     const bool recordPQ, const int step);

// Advances per-point thermal state (Ts_old, A1-3) via formula (U) using the
// live CARRA1 temperature, then evaluates depth-averaged flexural strength
// into idx_xi. Parameters:
//   - current_alpha_temp: temporal interpolation factor for the CARRA1
//     temperature frames (shares WindInterpolator's current_wind_alpha,
//     same frame cadence/upload as wind)
//   - dt_th: elapsed thermal-step duration in seconds
//     (IceStrengthUpdateStepInterval * InitialTimeStep)
__global__ void partition_kernel_compute_ice_strength(const PartitionParams pparams,
                                                       const double current_alpha_temp,
                                                       const double dt_th);

// ============================================================================
// RENDERING KERNELS - Visualization Data Preparation
// ============================================================================

// Renders visualization data: prepares per-particle values (pressure, stress, etc.)
// for display in the GUI by gathering and formatting data from particle state
// Renders in three groups: (1) mass/momentum/strain, (2) RGB/stress, (3) curvature/rotation
__global__ void partition_kernel_render_results(const PartitionParams pparams, int group);

// Normalizes rendered data: computes final visualization values after halo exchange
// (for multi-GPU partitions, halo data needs to be additively blended)
__global__ void partition_kernel_normalize_render(const PartitionParams pparams);

// Summarizes grid forces: aggregates forces on the grid for visualization/analysis
//__global__ void partition_kernel_summarize_forces(const PartitionParams pparams);

// ============================================================================
// MULTI-GPU COMMUNICATION KERNELS - Halo Exchange and Point Transfer
// ============================================================================

// Receives grid halo data from neighboring partitions and adds it to local grid
// Used in P2G phase for grid/force values computed in neighboring partitions
// Parameters:
//   - transfer_buffer_idx: which buffer contains the incoming halo data (0 or 1)
//   - receive_offset: starting grid X index where halo data should be placed
//   - receive_width: width (in X) of the halo region being received
__global__ void partition_kernel_receive_subgrid(const PartitionParams pparams,
                                                 const size_t transfer_buffer_idx,
                                                 const size_t receive_offset,
                                                 const size_t receive_width);

// Checks which points need to be transferred to neighboring partitions
// Sets flags in partition state (transfer_to_left, transfer_to_right counts)
__global__ void partition_kernel_check_if_transfer_needed(const PartitionParams pparams);

// Transfers points to neighboring partitions: copies particle data that has moved
// outside the partition's X domain to staging buffers for sending to neighbors
__global__ void partition_kernel_point_transfer(const PartitionParams pparams);

// Receives points from neighboring partitions and integrates them into local point array
// Parameters:
//   - nPts: number of points being received
//   - bufferIdx: which receive buffer to use (2=from left, 3=from right)
__global__ void partition_kernel_receive_points(const PartitionParams pparams,
                                                const unsigned nPts,
                                                const unsigned bufferIdx);

// ============================================================================
// DEVICE HELPER FUNCTIONS - Called by kernels for computation
// ============================================================================

// SVD decomposition for 2x2 matrices: mA = mU * diag(mS) * mV^T
// Used for deformation gradient analysis and stress computation
__device__ void svd2x2(const Eigen::Matrix2d &mA, Eigen::Matrix2d &mU, Eigen::Vector2d &mS, Eigen::Matrix2d &mV);

// Computes pressure (p) and second stress invariant (q) from deformation and material properties
// Used for failure surface checks and visualization
__device__ void ComputePQ(double &Je_tr, double &p_tr, double &q_tr, const Eigen::Matrix2d &F);



// Checks if a material point has exceeded the failure surface (yield criterion)
// Sets status flags if failure has occurred
__device__ void CheckIfPointIsInsideFailureSurface(unsigned long long &utility_data,
                                                   const double &p, const double &q);

// Retrieves grain-specific material parameters (strength bounds, hardening)
__device__ void GetParametersForGrain(uint32_t utility_data, double &pmin, double &pmax, double &qmax,
                                      double &beta, double &mSq, double &pmin2);

// Computes Kirchhoff stress from deformation gradient using Wolper material model
__device__ Eigen::Matrix2d KirchhoffStress_Wolper(const Eigen::Matrix2d &F);

// compute strain energy density
__device__ double StrainEnergyDensity(const Eigen::Matrix2d &F);

// how Jp_inv affects bulk modulus
__device__ double BulkModulusReductionCoeff(const double &Jp_inv);

// Extracts deviatoric (traceless) part of a 2D diagonal matrix
__device__ Eigen::Vector2d dev_d(Eigen::Vector2d Adiag);

// Extracts deviatoric (traceless) part of a 2D matrix
__device__ Eigen::Matrix2d dev(Eigen::Matrix2d A);

// Computes B-spline weight coefficients for particle-grid interpolation
// ww[i+1] contains weights for axis i at positions i-1, i, i+1
__device__ void CalculateWeightCoeffs(const Eigen::Vector2d &pos, Eigen::Array2d ww[3]);

// Retrieves wind vector at given position and time from interpolated wind field data
__device__ Eigen::Vector2d get_wind_vector(float lat, float lon, float tb);

// obtain point's (i,j) cell index from raw double value (stored in points buffer)
__device__ Eigen::Vector2i getIntegerCellIndex(double raw_value);






/*
__global__ void partition_kernel_summarize_forces(const PartitionParams pparams)
{
    // forces that were recorded (accumulated) in grid_idx_fx/fy are now summarized by region
    const size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t nNodes = (pparams.partition_gridX + 2*gprms.GridHaloSize) * gprms.GridYTotal;
    if(idx >= nNodes) return;

    //const int &gridY = gprms.GridYTotal;
    const size_t &pitch_grid = pparams.pitch_grid;
    double* const &bgrid = pparams.buffer_grid;

    double fx = bgrid[SimParams::GPUGridArrayIndex::gpu_grid_idx_fx*pitch_grid + idx];
    double fy = bgrid[SimParams::GPUGridArrayIndex::gpu_grid_idx_fy*pitch_grid + idx];

    uint8_t area_idx = pparams.buffer_grid_regions[idx];
    if(area_idx < SimParams::MAX_REGIONS && (fx != 0 || fy != 0))
    {
        atomicAdd(&pparams.grid_forces_summary_per_region[area_idx*2+0], fx);
        atomicAdd(&pparams.grid_forces_summary_per_region[area_idx*2+1], fy);
    }
}
*/


#endif // KERNEL_DECLARATIONS_CUH
