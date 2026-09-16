#include "parameters_sim.h"
#include "gpu_partition.h"
#include "kernel_declarations.cuh"
#include "helper_math.cuh"

using namespace Eigen;

constexpr double coeff1 = 1.4142135623730950; // sqrt((6-d)/2.);
//constexpr double coeff1_inv = 0.7071067811865475;

__constant__ SimParams gprms;
// error indicator and codes
__device__ uint32_t gpu_error_indicator;

#ifdef ENABLE_NAN_CHECKS
constexpr uint32_t error_code_point_pos_nan = 0x0001;           // point's position is NaN
constexpr uint32_t error_code_point_vel_nan = 0x0002;           // point's velocity is NaN
constexpr uint32_t error_code_point_jump_cells = 0x0004;    // point is flying too fast
constexpr uint32_t error_code_point_left_area = 0x0008;     // point is outside of bounds
constexpr uint32_t error_code_point_left_global = 0x0010;     // point is outside of global bounds
constexpr uint32_t error_code_point_Bp_nan = 0x0020;
constexpr uint32_t error_code_point_Fe_nan = 0x0040;
constexpr uint32_t error_code_grid_p2g_nan_vel = 0x0100;    // during P2G writing NaN velocity into grid
constexpr uint32_t error_code_grid_p2g_nan_mass = 0x0200;   // during P2G writing NaN velocity into grid
constexpr uint32_t error_code_grid_nan = 0x0400;            // velocity on the grid is NaN (during grid update)
constexpr uint32_t error_code_point_thermal_nan = 0x0800;   // Ts_old/A1-3/idx_xi is NaN (compute_ice_strength)
#endif



__device__ void PlasticProjection(unsigned long long &utility_data,
                                  const double &p_tr, const double &q_tr, const double &Je_tr,
                                  Eigen::Matrix2d &Fe, double &Jp_inv, double &Jp_inv_max, double &gamma_p, const double &dt,
                                  const double &xi)
{
    const double DP_threshold_p = gprms.DP_threshold_p;

    // xi is the raw flexural strength; q0/negT/compressive_threshold_fracture scale off it below.
    // compressive_threshold_fracture caps intact pressure directly (elastic-domain check below),
    // unscaled by Cf_now -- intact material has no plastic-strain history yet.
    double q0, negT, compressive_threshold_fracture;
    if(gprms.UseLiveIceStrength)
    {
        q0 = xi * gprms.IceStrengthFactor;
        negT = -(xi * gprms.IceStrengthFactorTensile);
        compressive_threshold_fracture = xi * gprms.IceStrengthFactorCompressive;
    }
    else
    {
        q0 = gprms.IceShearStrength;
        negT = -gprms.IceTensileFailureStrength;
        compressive_threshold_fracture = gprms.IceCompressiveThreshold;
    }
    // Cf(gamma_p) softening -- comment out this line and uncomment the next to fall back to a constant Cf.
    const double Cf_now = gprms.Cf_residual + (gprms.Cf - gprms.Cf_residual) * exp(-gamma_p / gprms.GammaStar);
    //const double Cf_now = gprms.Cf;

    // Yield-side (post-crack) pressure cap -- unlike compressive_threshold_fracture above,
    // this one scales with Cf_now: heavily shear-worked material holds pressure less well.
    double compressive_threshold_yield;
    if(gprms.UseLiveIceStrength)
        compressive_threshold_yield = xi * gprms.IceStrengthFactorCompressiveYield * Cf_now;
    else
        compressive_threshold_yield = gprms.IceCompressiveThreshold * Cf_now;

    // Crossover (steep DP leg -> shallower 2nd leg, see Q_From_Yield_Surface) sits Cf_now of the
    // way from DP_threshold_p to where that DP line would cross the intact envelope's compression
    // leg -- Cf_now=1 puts the yield surface's kink exactly on the intact envelope.
    const double p_intersect = (q0 + gprms.DP_tan_phi * DP_threshold_p) / (gprms.DP_tan_phi - gprms.IceFractureAngle_tan);
    const double q_intersect = q0 + gprms.IceFractureAngle_tan * p_intersect;
    const double q_crossover = q_intersect * Cf_now;

    // Intact envelope evaluated at p_tr -- both the elastic-domain check below and the
    // viscous-relaxation clamp further down need this same value at this same p_tr.
    const double q_envelope_at_p_tr = (p_tr <= 0) ? (q0/(-negT))*(p_tr - negT) : q0 + gprms.IceFractureAngle_tan*p_tr;

    double q_new = 0, p_new = 0;
    bool shearReturn = false;   // true for every branch except pure tension -- feeds gamma_p below
    constexpr double Jp_inv_threshold = 0.1;
    //    double dp_comp = 0;

    if(!(utility_data & SimParams::status_cracked))
    {
        // Elastic-domain check: does the trial state exceed intact material's strength?
        // Only decides whether to flip to status_cracked -- the actual return mapping for
        // cracked material always goes through Q_From_Yield_Surface below, not this envelope.
        if(p_tr < negT)
        {
            utility_data |= SimParams::status_cracked;
            utility_data |= SimParams::fracture_tension;
        }
        else  // p_tr >= negT
        {
            if(q_tr <= q_envelope_at_p_tr && p_tr <= compressive_threshold_fracture)
            {
                return; // we are inside the fracture surface
            }
            else
            {
                utility_data |= SimParams::status_cracked;
                if(p_tr < 0)
                    utility_data |= SimParams::fracture_tension;
                else if(p_tr > compressive_threshold_fracture)
                    utility_data |= SimParams::fracture_crush;
                else
                    utility_data |= SimParams::fracture_compression_shear;
            }
        }
    }

    if(utility_data & SimParams::status_cracked)
    {
        if(p_tr < DP_threshold_p)
        {
            utility_data |= SimParams::fracture_tension;
            utility_data &= ~SimParams::fracture_compression_shear;
            utility_data &= ~SimParams::fracture_crush;

            // Viscous relaxation toward zero (same RelaxationAlpha as the main branch),
            // not an instant drop. No gamma_p (not shear flow); no clamp needed --
            // decaying toward 0 can't overshoot upward, unlike the main branch.
            q_new = q_tr * gprms.RelaxationAlpha;

            if(Jp_inv < Jp_inv_threshold)
            {
                // prevent infinite expansion
                p_new = p_tr;
            }
            else
            {
                // allow to expand
            }
        }
        else if(Jp_inv < Jp_inv_max && p_tr > 0)
        {
            // Material is fractured and dispersed - zero stress in this dispersed state.
            // Compared to Jp_inv_max, not the fixed reference value of 1: material once
            // crushed past 1 loses that fraction permanently (piled onto neighboring ice
            // or dispersed into the water), so a re-expanded point should not regain
            // resistance until it closes back up to its own past peak compaction.
            utility_data &= ~SimParams::fracture_tension;
            utility_data |= SimParams::fracture_compression_shear;
            utility_data &= ~SimParams::fracture_crush;
            shearReturn = true;
        }
        else
        {
            // check if we are inside the yield envelope
            double q_yield = Q_From_Yield_Surface(p_tr, q_crossover);

            if(q_tr > q_yield)
            {
                shearReturn = true;
                // outside of yield envelope
                if(Jp_inv < Jp_inv_max)
                {
                    // material is dispersed - no resistance in shear (see the Jp_inv_max
                    // comment above)
                    utility_data &= ~SimParams::fracture_crush;
                }
                else
                {
                    // Fixed pressure cap every step -- no growing rubble-pile capacity.
                    // fracture_crush now tracks "is at the compressive cap this step," toggled
                    // on/off exactly like fracture_compression_shear/fracture_tension, not sticky.
                    if(p_tr > compressive_threshold_yield)
                    {
                        p_new = compressive_threshold_yield;
                        utility_data |= SimParams::fracture_crush;
                        utility_data &= ~SimParams::fracture_compression_shear;
                        utility_data &= ~SimParams::fracture_tension;
                    }
                    else
                    {
                        p_new = p_tr;
                        utility_data |= SimParams::fracture_compression_shear;
                        utility_data &= ~SimParams::fracture_tension;
                        utility_data &= ~SimParams::fracture_crush;
                    }

                    q_yield = Q_From_Yield_Surface(p_new, q_crossover);
                    // Duvaut-Lions relaxation toward the yield surface (gprms.RelaxationAlpha,
                    // from TauInv), not instant projection -- clamped to the intact envelope
                    // at p_tr so no choice of tau can push q_new past it.
                    q_new = q_yield + (q_tr - q_yield) * gprms.RelaxationAlpha;
                    q_new = min(q_new, q_envelope_at_p_tr);
                }
            }
            else
            {
                return; // within fracture envelope
            }
        }
    }

    if(shearReturn)
        gamma_p += (q_tr - q_new) / (2.0 * gprms.mu);

    // After having calculated p_new and q_new, obtain the new Fe
    ReconstructDeformationGradient(Fe, Jp_inv, p_new, q_new, q_tr, Je_tr);

    // Track the point's historical peak compaction (permanent, never decreases) --
    // see the idx_Jp_inv_max comment in parameters_sim.h.
    Jp_inv_max = fmax(Jp_inv_max, Jp_inv);
}



__global__ void partition_kernel_p2g(const PartitionParams pparams)
{
    const size_t pt_idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if(pt_idx >= pparams.count_pts) return;

    const double &h = gprms.cellsize;

    const int &gridY = gprms.GridYTotal;
    const unsigned &halo = gprms.GridHaloSize;
    const size_t &pitch = pparams.pitch_pts;
    const size_t &gridX_offset = pparams.gridX_offset;
    double* const &bpts = pparams.buffer_pts;
    double* const &bgrid = pparams.buffer_grid;

    const double utility_double = bpts[pt_idx + pitch*SimParams::PtArrIdx::idx_utility_data];
    const long long utility_data = __double_as_longlong(utility_double);
    if(utility_data & SimParams::status_disabled) return; // point is disabled

    // Two-field contact MPM: intact material scatters onto field-0, cracked/crushed onto field-1 ("_f", +4 offset).
    const size_t field_offset = (gprms.UseDoubleGrid && (utility_data & SimParams::status_cracked)) ? 4 : 0;

    // pull point data from SOA
    Eigen::Vector2d pos, velocity;
    Eigen::Matrix2d Cp, Fe, PFt;
    Eigen::Matrix2d stress_contribution = Eigen::Matrix2d::Zero();

    const double thickness = bpts[pt_idx + pitch*SimParams::PtArrIdx::idx_thickness];
    Eigen::Vector2i cell_i = getIntegerCellIndex(bpts[pt_idx + pitch*SimParams::PtArrIdx::integer_cell_idx]);

#ifdef ENABLE_SCALING_FACTOR
    // per-point m^2 (seeded once) and per-cell 1/m (nearest cell)
    const double m_squared = bpts[pt_idx + pitch*SimParams::PtArrIdx::idx_m_squared];
    const size_t idx_own_cell = cell_i[1] + (cell_i[0]-gridX_offset+halo)*(size_t)gridY;
    const float m_p_inv = pparams.buffer_grid_forcing[SimParams::GridForcingFramesIndex::grid_idx_scaling_factor_inverse*pparams.pitch_grid_forcing + idx_own_cell];
#else
    const double m_squared = 1.0;
    const float m_p_inv = 1.0;
#endif
    const double particle_mass = gprms.ParticleMass * thickness * m_squared;

    for(int i=0; i<SimParams::dim; i++)
    {
        velocity[i] = bpts[pt_idx + pitch*(SimParams::PtArrIdx::velx+i)];
        pos[i] = bpts[pt_idx + pitch*(SimParams::PtArrIdx::posx+i)];

        for(int j=0; j<SimParams::dim; j++)
        {
            Fe(i,j) = bpts[pt_idx + pitch*(SimParams::PtArrIdx::Fe00 + i*SimParams::dim + j)];
            Cp(i,j) = bpts[pt_idx + pitch*(SimParams::PtArrIdx::Bp00 + i*SimParams::dim + j)];
        }
    }
    double Jp_inv = bpts[pt_idx + pitch*SimParams::PtArrIdx::idx_Jp_inv];

    // PFt is 1st Piola-Kirchhoff Stress times F-transposed
    PFt = KirchhoffStress_Wolper(Fe);
    stress_contribution = -(gprms.dt_area_Dpinv * Jp_inv * thickness * m_p_inv) * PFt;
    stress_contribution += Cp * particle_mass;    // this is part of the linear term from the velocity approximateion

    Eigen::Array2d ww[3];
    CalculateWeightCoeffs(pos, ww);

    for (int i = -1; i <= 1; i++)
        for (int j = -1; j <= 1; j++)
        {
            const double Wip = ww[i+1][0]*ww[j+1][1];
            const Eigen::Vector2d dpos((i-pos[0])*h, (j-pos[1])*h);

            // index of the cell takes into accout the partition's offset of the gird fragment
            const size_t idx_gridnode = (j+cell_i[1]) + (i+cell_i[0]-gridX_offset+halo)*(size_t)gridY;

            const double incM = Wip*particle_mass;
            atomicAdd(&bgrid[(SimParams::GPUGridArrayIndex::gpu_grid_idx_mass+field_offset)*pparams.pitch_grid + idx_gridnode], incM);

            const double incTh = incM*thickness;
            atomicAdd(&bgrid[(SimParams::GPUGridArrayIndex::gpu_grid_idx_thickness+field_offset)*pparams.pitch_grid + idx_gridnode], incTh);

            const Eigen::Vector2d incV = Wip*(velocity*particle_mass + stress_contribution*dpos);

            // distribute values to the grid (mass and momentum)
            atomicAdd(&bgrid[(SimParams::GPUGridArrayIndex::gpu_grid_idx_px+field_offset)*pparams.pitch_grid + idx_gridnode], incV[0]);
            atomicAdd(&bgrid[(SimParams::GPUGridArrayIndex::gpu_grid_idx_py+field_offset)*pparams.pitch_grid + idx_gridnode], incV[1]);
#ifdef ENABLE_NAN_CHECKS
            // sanity checks
            if(isnan(incV[0]) || isnan(incV[1])) gpu_error_indicator |= error_code_grid_p2g_nan_vel;
            if(isnan(incM)) gpu_error_indicator |= error_code_grid_p2g_nan_mass;
#endif
        }


#ifdef ENABLE_NAN_CHECKS
    // global bounds
    // check if a point is out of bounds of the local grid partition
    const int lboundX = 1 + (int)pparams.gridX_offset - (int)gprms.GridHaloSize;
    const int hboundX = pparams.partition_gridX + pparams.gridX_offset - 2 + gprms.GridHaloSize;

    if(cell_i[0] < 1 || cell_i[1] < 1 || cell_i[0] > (gprms.GridXTotal-2) || cell_i[1] > gridY-2)
        gpu_error_indicator |= error_code_point_left_global;
    else if(cell_i[0] < lboundX || cell_i[0] > hboundX)
        gpu_error_indicator |= error_code_point_left_area;
#endif
}


__global__ void partition_kernel_update_nodes(const PartitionParams pparams,
                                              const double simulation_time, const double current_alpha, const double current_alpha_wind)
{
    const size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t nNodes = (pparams.partition_gridX + 2*gprms.GridHaloSize) * gprms.GridYTotal;
    if(idx >= nNodes) return;

    //const int &gridY = gprms.GridYTotal;
    const size_t &pitch_grid = pparams.pitch_grid;
    double* const &bgrid = pparams.buffer_grid;
    const size_t &pitch_grid_forcing = pparams.pitch_grid_forcing;
    float* const &bgrid_forcing = pparams.buffer_grid_forcing;

    //const double &cellsize = gprms.cellsize;
    const double &dt = gprms.InitialTimeStep;               // time step

    // Normalize momentum to get velocity
    const Vector2i gi((int)idx/gprms.GridYTotal+(int)pparams.gridX_offset-(int)gprms.GridHaloSize, idx%gprms.GridYTotal);   // integer x-y index of the grid node
    const Eigen::Vector2d gnpos = gi.cast<double>()*gprms.cellsize;    // position of the grid node in the whole grid

    uint8_t is_modeled_area = pparams.buffer_grid_regions[idx];
    // Land is [0, OpenBoundaryIndicator), not just 0. OpenBoundaryIndicator needs water-like physics so
    // points keep momentum and cross into it, where g2p's advection check disables them (else they pile up).
    const bool is_land = (is_modeled_area != SimParams::ModelledAreaIndicator && is_modeled_area != SimParams::OpenBoundaryIndicator);

    // Horizontal current/wind velocity from 2 time frames for drag -- node geometry/forcing only, shared by both fields.
    Eigen::Vector2d v_frame0(bgrid_forcing[SimParams::GridForcingFramesIndex::grid_idx_current_vx_frame0*pitch_grid_forcing + idx],
                             bgrid_forcing[SimParams::GridForcingFramesIndex::grid_idx_current_vy_frame0*pitch_grid_forcing + idx]);

    Eigen::Vector2d v_frame1(bgrid_forcing[SimParams::GridForcingFramesIndex::grid_idx_current_vx_frame1*pitch_grid_forcing + idx],
                             bgrid_forcing[SimParams::GridForcingFramesIndex::grid_idx_current_vy_frame1*pitch_grid_forcing + idx]);

    Eigen::Vector2d v_w = (1.0 - current_alpha) * v_frame0 + current_alpha * v_frame1;

    // Linear spin-up over ForcingSpinUpDuration so drag doesn't snap on at full strength at step 0.
    const double forcing_ramp = min(1.0, simulation_time / gprms.ForcingSpinUpDuration);
    v_w *= forcing_ramp;

    // obtain wind velocity from GPU global memory
    Eigen::Vector2d v_frame0_wind(bgrid_forcing[SimParams::GridForcingFramesIndex::grid_idx_wind_vx_frame0*pitch_grid_forcing + idx],
                                  bgrid_forcing[SimParams::GridForcingFramesIndex::grid_idx_wind_vy_frame0*pitch_grid_forcing + idx]);

    Eigen::Vector2d v_frame1_wind(bgrid_forcing[SimParams::GridForcingFramesIndex::grid_idx_wind_vx_frame1*pitch_grid_forcing + idx],
                                  bgrid_forcing[SimParams::GridForcingFramesIndex::grid_idx_wind_vy_frame1*pitch_grid_forcing + idx]);

    Eigen::Vector2d v_wind = (1.0 - current_alpha_wind) * v_frame0_wind + current_alpha_wind * v_frame1_wind;
    v_wind *= forcing_ramp;

    // Two-field contact MPM: intact (field 0) and cracked/crushed (field 1, "_f" slots) solved independently, each through its own thickness, until the contact kernel couples them.
    const int kNumFields = gprms.UseDoubleGrid ? 2 : 1;
    for(int field = 0; field < kNumFields; field++)
    {
        const size_t off = field*4;

        const double mass = bgrid[(SimParams::GPUGridArrayIndex::gpu_grid_idx_mass+off)*pitch_grid + idx];
        if(mass == 0) continue; // this field isn't present at this node

        double vx = bgrid[(SimParams::GPUGridArrayIndex::gpu_grid_idx_px+off)*pitch_grid + idx];
        double vy = bgrid[(SimParams::GPUGridArrayIndex::gpu_grid_idx_py+off)*pitch_grid + idx];
        double thickness = bgrid[(SimParams::GPUGridArrayIndex::gpu_grid_idx_thickness+off)*pitch_grid + idx]/mass;

        Eigen::Vector2d momentum(vx, vy);  // at this point it is momentum
        Eigen::Vector2d velocity = momentum/mass;

        if(is_land)
        {
            velocity.setZero();
        }
        else
        {
            // effect of the water drag on horizontal velocity
            const double C_water = gprms.waterDragNormalized * (SimParams::rho_water / (gprms.IceDensity * thickness));
            const double kQp = C_water * dt; // quadratic
            Eigen::Vector2d U_rel = (v_w - velocity);  // relative velocity
            double k = kQp * U_rel.norm();
            k = min(k, 0.5);   // k cannot exceed 0.5

            // effect of the wind drag
            const double C_wind = gprms.windDragNormalized * (SimParams::rho_air / (gprms.IceDensity * thickness));

            Eigen::Vector2d U_rel_wind = (v_wind - velocity);  // relative velocity
            double k_wind = C_wind * dt * U_rel_wind.norm();
            k_wind = min(k_wind, 0.5);   // cannot exceed 0.5

            velocity += (k*U_rel);
            if(gprms.UseWindData) velocity += k_wind*U_rel_wind;
        }

        // write the updated grid velocity back to memory
        if(velocity.squaredNorm() > gprms.vmax*gprms.vmax*0.5) velocity.setZero();

        bgrid[(SimParams::GPUGridArrayIndex::gpu_grid_idx_px+off)*pitch_grid + idx] = velocity[0];
        bgrid[(SimParams::GPUGridArrayIndex::gpu_grid_idx_py+off)*pitch_grid + idx] = velocity[1];

#ifdef ENABLE_NAN_CHECKS
        if(isnan(velocity[0]) || isnan(velocity[1])) gpu_error_indicator |= error_code_grid_nan;
#endif
    }
}

// Two-field contact MPM: resolves the interaction between the intact-field and fractured-field velocities at nodes where both are present, after partition_kernel_update_nodes and before partition_kernel_g2p. No-op when UseDoubleGrid is off.
__global__ void partition_kernel_contact(const PartitionParams pparams)
{
    if(!gprms.UseDoubleGrid) return;

    const size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const int gridYT = gprms.GridYTotal;
    const size_t nNodes = (pparams.partition_gridX + 2*gprms.GridHaloSize) * gridYT;
    if(idx >= nNodes) return;

    const size_t &pitch_grid = pparams.pitch_grid;
    double* const &bgrid = pparams.buffer_grid;

    // land nodes have prescribed zero velocity for both fields; nothing to resolve
    const uint8_t is_modeled_area = pparams.buffer_grid_regions[idx];
    if(is_modeled_area != SimParams::ModelledAreaIndicator &&
       is_modeled_area != SimParams::OpenBoundaryIndicator) return;

    const double m0 = bgrid[SimParams::gpu_grid_idx_mass  *pitch_grid + idx];
    const double m1 = bgrid[SimParams::gpu_grid_idx_mass_f*pitch_grid + idx];
    const double mtot = m0 + m1;
    if(mtot == 0) return;

    // single-field node: no interface. Relative threshold suppresses trace-mass stencil-tail noise.
    constexpr double mass_frac_eps = 1e-3;
    if(m0 < mass_frac_eps*mtot || m1 < mass_frac_eps*mtot) return;

    // Common normal from grad(m0 - m1), central differences. Guard outermost columns (halo-crossing, out of bounds).
    const int ix = (int)(idx / gridYT);
    const int iy = (int)(idx % gridYT);
    const int nx_total = pparams.partition_gridX + 2*gprms.GridHaloSize;
    if(ix == 0 || ix == nx_total-1 || iy == 0 || iy == gridYT-1) return;

    auto massDiff = [&](size_t j) -> double {
        return bgrid[SimParams::gpu_grid_idx_mass  *pitch_grid + j]
             - bgrid[SimParams::gpu_grid_idx_mass_f*pitch_grid + j];
    };
    // cellsize factor cancels on normalization; omit it
    Eigen::Vector2d g(massDiff(idx + gridYT) - massDiff(idx - gridYT),
                      massDiff(idx + 1)      - massDiff(idx - 1));

    const double gnorm = g.norm();
    // Degenerate normal (uniform mixture): no meaningful orientation, skip rather than apply a random impulse.
    constexpr double grad_eps = 1e-10;
    if(gnorm < grad_eps) return;

    // g points toward increasing (m0 - m1), i.e., into the intact side.
    // Outward normal of the intact field (pointing at the fractured side):
    const Eigen::Vector2d n0 = -g / gnorm;

    // ---- velocities (update kernel has already normalized momenta) ----
    Eigen::Vector2d v0(bgrid[SimParams::gpu_grid_idx_px  *pitch_grid + idx],
                       bgrid[SimParams::gpu_grid_idx_py  *pitch_grid + idx]);
    Eigen::Vector2d v1(bgrid[SimParams::gpu_grid_idx_px_f*pitch_grid + idx],
                       bgrid[SimParams::gpu_grid_idx_py_f*pitch_grid + idx]);

    const Eigen::Vector2d v_cm = (m0*v0 + m1*v1) / mtot;

    // approach test in the center-of-mass frame; by momentum conservation
    // it suffices to test and correct field 0 and mirror the impulse
    const Eigen::Vector2d dv0 = v0 - v_cm;
    const double dvn = dv0.dot(n0);
    if(dvn <= 0) return;            // separating or sliding apart: free separation,
                                     // fields keep their own velocities -> crack opens

    // ---- Coulomb friction ----
    const Eigen::Vector2d dvt = dv0 - dvn*n0;
    const double dvt_norm = dvt.norm();

    Eigen::Vector2d v0_new;
    if(dvt_norm <= gprms.IceFrictionCoeff * dvn)
        v0_new = v_cm;                                                    // stick
    else
        v0_new = v0 - dvn*n0 - gprms.IceFrictionCoeff*dvn*(dvt/dvt_norm);  // slip

    // equal-and-opposite impulse on the fractured field
    const Eigen::Vector2d dp = m0*(v0_new - v0);
    const Eigen::Vector2d v1_new = v1 - dp/m1;

    bgrid[SimParams::gpu_grid_idx_px  *pitch_grid + idx] = v0_new[0];
    bgrid[SimParams::gpu_grid_idx_py  *pitch_grid + idx] = v0_new[1];
    bgrid[SimParams::gpu_grid_idx_px_f*pitch_grid + idx] = v1_new[0];
    bgrid[SimParams::gpu_grid_idx_py_f*pitch_grid + idx] = v1_new[1];

#ifdef ENABLE_NAN_CHECKS
    if(isnan(v0_new[0]) || isnan(v0_new[1]) || isnan(v1_new[0]) || isnan(v1_new[1]))
        gpu_error_indicator |= error_code_grid_nan;
#endif
}

__global__ void partition_kernel_g2p(const PartitionParams pparams, const bool recordPQ, const int step)
{
    const size_t pt_idx = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if(pt_idx >= pparams.count_pts) return;

    const unsigned &halo = gprms.GridHaloSize;
    const size_t &pitch_pts = pparams.pitch_pts;
    const size_t &pitch_grid = pparams.pitch_grid;
    const size_t &gridX_offset = pparams.gridX_offset;
    double* const &bpts = pparams.buffer_pts;
    double* const &bgrid = pparams.buffer_grid;

    const double &h_inv = gprms.cellsize_inv;
    const double &dt = gprms.InitialTimeStep;
    const int &gridY = gprms.GridYTotal;

    // skip if a point is disabled
    const double utility_double_g2p = bpts[pt_idx + pitch_pts*SimParams::PtArrIdx::idx_utility_data];
    unsigned long long utility_data = __double_as_longlong(utility_double_g2p);
    if(utility_data & SimParams::status_disabled) return; // point is disabled
    const unsigned long long utility_original = utility_data;

    // Gather from the field matching this particle's state at the START of the step (before PlasticProjection below may flip intact->cracked).
    const size_t field_offset = (gprms.UseDoubleGrid && (utility_data & SimParams::status_cracked)) ? 4 : 0;

    Eigen::Vector2d pos;
    Eigen::Vector2d p_velocity; p_velocity.setZero();
    double Je_tr, p_tr, q_tr;

    Eigen::Matrix2d Fe;         // deformation gradient
    Eigen::Matrix2d p_Bp; p_Bp.setZero();

    // pull point data from SOA
    const double xi = bpts[pt_idx + pitch_pts*SimParams::PtArrIdx::idx_xi];
    double Jp_inv = bpts[pt_idx + pitch_pts*SimParams::PtArrIdx::idx_Jp_inv];
    const double Jp_inv_old = Jp_inv;
    double Jp_inv_max = bpts[pt_idx + pitch_pts*SimParams::PtArrIdx::idx_Jp_inv_max];
    const double Jp_inv_max_old = Jp_inv_max;
    double gamma_p = bpts[pt_idx + pitch_pts*SimParams::PtArrIdx::idx_gamma_p];
    const double gamma_p_old = gamma_p;
    Eigen::Vector2i cell_i  = getIntegerCellIndex(bpts[pt_idx + pitch_pts*SimParams::PtArrIdx::integer_cell_idx]);

#ifdef ENABLE_SCALING_FACTOR
    const size_t &pitch_grid_forcing = pparams.pitch_grid_forcing;
    float* const &bgrid_forcing = pparams.buffer_grid_forcing;
    // Inverse projection scale factor (1/m)
    const size_t idx_own_cell = cell_i[1] + (cell_i[0]-gridX_offset+halo)*(size_t)gridY;
    const float m_p_inv = bgrid_forcing[SimParams::GridForcingFramesIndex::grid_idx_scaling_factor_inverse*pitch_grid_forcing + idx_own_cell];
#else
    const float m_p_inv = 1.0;
#endif

    for(int i=0; i<SimParams::dim; i++)
    {
        pos[i] = bpts[pt_idx + pitch_pts*(SimParams::PtArrIdx::posx+i)];
        for(int j=0; j<SimParams::dim; j++)
        {
            Fe(i,j) = bpts[pt_idx + pitch_pts*(SimParams::PtArrIdx::Fe00 + i*SimParams::dim + j)];
        }
    }

    // optimized method of computing the quadratic weight function without conditional operators
    Eigen::Array2d ww[3];
    CalculateWeightCoeffs(pos, ww);
    // pull velocity from the grid
    for (int i = -1; i <= 1; i++)
        for (int j = -1; j <= 1; j++)
        {
            Eigen::Vector2d dpos = Eigen::Vector2d(i, j) - pos;
            double weight = ww[i+1][0]*ww[j+1][1];

            // grid node index within the 3x3 loop
            const size_t idx_gridnode = (j+cell_i[1]) + (i+cell_i[0]-gridX_offset+halo)*(size_t)gridY;

            Eigen::Vector2d node_velocity;  // normal in-plane velocity
            node_velocity[0] = bgrid[(SimParams::grid_idx_px+field_offset)*pitch_grid + idx_gridnode];
            node_velocity[1] = bgrid[(SimParams::grid_idx_py+field_offset)*pitch_grid + idx_gridnode];
            p_velocity += weight * node_velocity;
            p_Bp += (4.*h_inv*weight) * (node_velocity*dpos.transpose());
        }

    // Advection and update of the deformation gradient
    bool cell_updated = false; // record if the point moved into another cell
    pos += p_velocity * (dt*h_inv*m_p_inv); // position is in local cell coordinates [-0.5 to 0.5]

#ifdef ENABLE_NAN_CHECKS
    // check if there is an error
    if(isnan(p_velocity[0]) || isnan(p_velocity[1])) gpu_error_indicator |= error_code_point_vel_nan;
    if(isnan(pos[0]) || isnan(pos[1])) gpu_error_indicator |= error_code_point_pos_nan;
    if(isnan(p_Bp(0,0)) || isnan(p_Bp(1,0)) || isnan(p_Bp(0,1)) || isnan(p_Bp(1,1))) gpu_error_indicator |= error_code_point_Bp_nan;
#endif

    // encode the position of the point as coordinates + cell index
    // if a point moves to the next cell, account for the change
    if(pos.x() > 0.5) { pos.x() -= 1.0; cell_i.x()++; cell_updated = true; }
    else if(pos.x() < -0.5) { pos.x() += 1.0; cell_i.x()--; cell_updated = true; }
    if(pos.y() > 0.5) { pos.y() -= 1.0; cell_i.y()++; cell_updated = true; }
    else if(pos.y() < -0.5) { pos.y() += 1.0; cell_i.y()--; cell_updated = true; }

    // this allows the points to leave the simulation area and become disabled
    if(cell_updated)
    {
        if(cell_i.x() <= 1 || cell_i.x() >= gprms.GridXTotal-2 || cell_i.y() <= 1 || cell_i.y() >= gridY-2)
        {
            utility_data |= SimParams::status_disabled;
            atomicAdd(pparams.disabled_points_count, 1);
        }
        else
        {
            // Open boundary: unlike the rectangular edge check above, follows the actual footprint mask,
            // disabling a point as soon as it enters that area (see SimParams::OpenBoundaryIndicator).
            const size_t idx_new_cell = cell_i[1] + (cell_i[0]-gridX_offset+halo)*(size_t)gridY;
            if(pparams.buffer_grid_regions[idx_new_cell] == SimParams::OpenBoundaryIndicator)
            {
                utility_data |= SimParams::status_disabled;
                atomicAdd(pparams.disabled_points_count, 1);
            }
        }
    }

    Fe = (Eigen::Matrix2d::Identity() + dt*p_Bp*m_p_inv) * Fe;     // Bp plays the role of the gradient of the velocity vector
    ComputePQ(Je_tr, p_tr, q_tr, Fe);    // computes P, Q

    // TEMPORARY: dumping every point's live (p_tr, q_tr) every step for envelope calibration -- normally this only records (p,q) at first crack, revert after calibration
    bpts[pt_idx + pitch_pts*SimParams::PtArrIdx::idx_stored_P] = p_tr;
    bpts[pt_idx + pitch_pts*SimParams::PtArrIdx::idx_stored_Q] = q_tr;

    if(gprms.AllowFracture) PlasticProjection(utility_data, p_tr, q_tr, Je_tr, Fe, Jp_inv, Jp_inv_max, gamma_p, gprms.InitialTimeStep, xi);

    for(int i=0; i<SimParams::dim; i++)
    {
        bpts[pt_idx + pitch_pts*(SimParams::PtArrIdx::posx+i)] = pos[i];
        bpts[pt_idx + pitch_pts*(SimParams::PtArrIdx::velx+i)] = p_velocity[i];
        for(int j=0; j<SimParams::dim; j++)
        {
            bpts[pt_idx + pitch_pts*(SimParams::PtArrIdx::Fe00 + i*SimParams::dim + j)] = Fe(i,j);
            bpts[pt_idx + pitch_pts*(SimParams::PtArrIdx::Bp00 + i*SimParams::dim + j)] = p_Bp(i,j);
        }
    }

    if(Jp_inv_old != Jp_inv)
        bpts[pt_idx + pitch_pts*SimParams::PtArrIdx::idx_Jp_inv] = Jp_inv;

    if(Jp_inv_max_old != Jp_inv_max)
        bpts[pt_idx + pitch_pts*SimParams::PtArrIdx::idx_Jp_inv_max] = Jp_inv_max;

    if(gamma_p_old != gamma_p)
        bpts[pt_idx + pitch_pts*SimParams::PtArrIdx::idx_gamma_p] = gamma_p;

    if(cell_updated)
    {
        long long cell = ((long long)cell_i[1] << 32) | (long long)cell_i[0];
        bpts[pt_idx + pitch_pts*SimParams::PtArrIdx::integer_cell_idx] = __longlong_as_double(cell);
    }

    // save crushed/disabled status (preserves upper 32 bits with color info)
    if(utility_data != utility_original)
        bpts[pt_idx + pitch_pts*SimParams::PtArrIdx::idx_utility_data] = __longlong_as_double(utility_data);

#ifdef ENABLE_NAN_CHECKS
    if(isnan(Fe(0,0)) || isnan(Fe(1,0)) || isnan(Fe(0,1)) || isnan(Fe(1,1))) gpu_error_indicator |= error_code_point_Fe_nan;
    // ensure the coordinates are valid
    if(pos.x() > 0.5 || pos.x() < -0.5 || pos.y() > 0.5 || pos.y() < -0.5)
        gpu_error_indicator |= error_code_point_jump_cells;
#endif
}


// ======================================== END OF P2G/UDPATE/G2P KERNELS


// Advances thermal state (formula (U)/(R)), converts to brine volume + Timco & O'Brien flexural
// strength, and stores the raw (unscaled) depth-average into idx_xi -- factors applied at point of use.
// See _python_code/heat_equation_spec.md.
__global__ void partition_kernel_compute_ice_strength(const PartitionParams pparams,
                                                        const double current_alpha_temp,
                                                        const double dt_th)
{
    const size_t pt_idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if(pt_idx >= pparams.count_pts) return;

    const unsigned &halo = gprms.GridHaloSize;
    const int &gridY = gprms.GridYTotal;
    const size_t &pitch_pts = pparams.pitch_pts;
    const size_t &pitch_grid_forcing = pparams.pitch_grid_forcing;
    const size_t &gridX_offset = pparams.gridX_offset;
    double* const &bpts = pparams.buffer_pts;
    float* const &bgrid_forcing = pparams.buffer_grid_forcing;

    // skip disabled points -- identical check to p2g/g2p
    const double utility_double = bpts[pt_idx + pitch_pts*SimParams::PtArrIdx::idx_utility_data];
    const long long utility_data = __double_as_longlong(utility_double);
    if(utility_data & SimParams::status_disabled) return;

    Eigen::Vector2d pos;
    for(int i=0; i<SimParams::dim; i++)
        pos[i] = bpts[pt_idx + pitch_pts*(SimParams::PtArrIdx::posx+i)];
    Eigen::Vector2i cell_i = getIntegerCellIndex(bpts[pt_idx + pitch_pts*SimParams::PtArrIdx::integer_cell_idx]);

    // 3x3 quadratic-weight stencil gather of the two CARRA1 temperature frames, same pattern as g2p's grid velocity gather.
    Eigen::Array2d ww[3];
    CalculateWeightCoeffs(pos, ww);
    double T_frame0 = 0.0, T_frame1 = 0.0;
    for(int i=-1; i<=1; i++)
        for(int j=-1; j<=1; j++)
        {
            const double weight = ww[i+1][0]*ww[j+1][1];
            const size_t idx_gridnode = (j+cell_i[1]) + (i+cell_i[0]-gridX_offset+halo)*(size_t)gridY;
            T_frame0 += weight * bgrid_forcing[SimParams::GridForcingFramesIndex::grid_idx_temperature_frame0*pitch_grid_forcing + idx_gridnode];
            T_frame1 += weight * bgrid_forcing[SimParams::GridForcingFramesIndex::grid_idx_temperature_frame1*pitch_grid_forcing + idx_gridnode];
        }
    // Kelvin -> Celsius, matching DataPreparer::ThermalSpinUpStep
    const double Ts_new = ((1.0-current_alpha_temp)*T_frame0 + current_alpha_temp*T_frame1) - 273.15;

    const double H = bpts[pt_idx + pitch_pts*SimParams::PtArrIdx::idx_thickness];
    const double Ts_old = bpts[pt_idx + pitch_pts*SimParams::PtArrIdx::idx_Ts_old];
    const double r = (Ts_new - Ts_old) / dt_th;

    static constexpr SimParams::PtArrIdx modeIdx[3] = {
        SimParams::PtArrIdx::idx_A1, SimParams::PtArrIdx::idx_A2, SimParams::PtArrIdx::idx_A3};
    double A[3];
    for(int n=1; n<=3; n++)
    {
        const double tau_n = (H*H) / (SimParams::kappa_ice * SimParams::pi*SimParams::pi * (double)(n*n));
        const double b_n = 2.0 / ((double)n * SimParams::pi);
        const double A_old = bpts[pt_idx + pitch_pts*modeIdx[n-1]];
        A[n-1] = (A_old + r*b_n*tau_n) * exp(-dt_th/tau_n) - r*b_n*tau_n;
        bpts[pt_idx + pitch_pts*modeIdx[n-1]] = A[n-1];
    }
    bpts[pt_idx + pitch_pts*SimParams::PtArrIdx::idx_Ts_old] = Ts_new;

    constexpr int N_DEPTH = 8;   // midpoints of 8 equal sub-layers, per heat_equation_spec.md §5
    double sigma_sum = 0.0;
    for(int k=0; k<N_DEPTH; k++)
    {
        const double z = (k + 0.5) * H / N_DEPTH;
        const double T = reconstruct_T(z, H, Ts_new, A[0], A[1], A[2]);
        const double vb = brine_volume_frankenstein_garner(T, gprms.IceSalinity);
        sigma_sum += flexural_strength_timco_obrien(vb);
    }
    const double xi_new = sigma_sum / N_DEPTH;
    bpts[pt_idx + pitch_pts*SimParams::PtArrIdx::idx_xi] = xi_new;

#ifdef ENABLE_NAN_CHECKS
    if(isnan(Ts_new) || isnan(A[0]) || isnan(A[1]) || isnan(A[2]) || isnan(xi_new))
        gpu_error_indicator |= error_code_point_thermal_nan;
#endif
}


__global__ void partition_kernel_render_results(const PartitionParams pparams, int group)
{
    // From point data, populate grid_idx_vis_{r,g,b,Jpinv,P,Q,strain_EqvGreenLagrange,strain_vonMises,ice_strength}.

    const size_t pt_idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if(pt_idx >= pparams.count_pts) return;

    const unsigned &halo = gprms.GridHaloSize;
    const int &gridY = gprms.GridYTotal;
    const size_t &gridX_offset = pparams.gridX_offset;
    const size_t &pitch = pparams.pitch_pts;
    const size_t &pitch_g = pparams.pitch_grid;

    double* const &bpts = pparams.buffer_pts;
    float* const &bgrid = (float*)pparams.buffer_grid;  // for rendering, we treat the grid buffer as 'float'

    const double utility_double = bpts[pt_idx + pitch*SimParams::PtArrIdx::idx_utility_data];
    unsigned long long utility = __double_as_longlong(utility_double);

    if(utility & SimParams::status_disabled) return; // point is disabled

    Eigen::Vector2d pos;
    for(int i=0; i<SimParams::dim; i++)
        pos[i] = bpts[pt_idx + pitch*(SimParams::PtArrIdx::posx+i)];
    Eigen::Array2d ww[3];
    CalculateWeightCoeffs(pos, ww);

    Eigen::Vector2i cell_i = getIntegerCellIndex(bpts[pt_idx + pitch*SimParams::PtArrIdx::integer_cell_idx]);
    const float thickness = (float)bpts[pt_idx + pitch*SimParams::PtArrIdx::idx_thickness];

#ifdef ENABLE_SCALING_FACTOR
    const float m_squared = (float)bpts[pt_idx + pitch*SimParams::PtArrIdx::idx_m_squared];
#else
    const float m_squared = 1.0f;
#endif
    const float particle_mass = (float)gprms.ParticleMass * thickness * m_squared;

    if(group == 0)
    {
        Eigen::Vector2f velocity;
        for(int i=0; i<SimParams::dim; i++) velocity[i] = (float)bpts[pt_idx + pitch*(SimParams::PtArrIdx::velx+i)];

        for (int i = -1; i <= 1; i++)
            for (int j = -1; j <= 1; j++)
            {
                const float Wip = ww[i+1][0]*ww[j+1][1];
                // index of the cell takes into accout the partition's offset of the gird fragment
                const size_t idx_gridnode = (j+cell_i[1]) + (i+cell_i[0]-gridX_offset+halo)*(size_t)gridY;

                // if(idx_gridnode >= (size_t)gridY*(pparams.partition_gridX+2*halo)) gpu_error_indicator |= error_code_point_left_area;


                const float incM = Wip*particle_mass;
                const Eigen::Vector2f incV = incM*velocity;

                // distribute values to the grid
                atomicAdd(&bgrid[SimParams::GPUGridArrayIndex::gpu_grid_idx_mass*pitch_g + idx_gridnode], incM);
                atomicAdd(&bgrid[SimParams::GPUGridArrayIndex::gpu_grid_idx_px*pitch_g + idx_gridnode], incV.x());
                atomicAdd(&bgrid[SimParams::GPUGridArrayIndex::gpu_grid_idx_py*pitch_g + idx_gridnode], incV.y());
            }
    }
    else if(group == 1)
    {
        // group 1: Color
        // Extract RGB (R: 24-31, G: 32-39, B: 40-47)
        uint8_t r = (utility >> 24) & 0xFF;
        uint8_t g = (utility >> 32) & 0xFF;
        uint8_t b = (utility >> 40) & 0xFF;

        const float rR = (double)r / 255.0;
        const float rG = (double)g / 255.0;
        const float rB = (double)b / 255.0;

        for (int i = -1; i <= 1; i++)
            for (int j = -1; j <= 1; j++)
            {
                const float Wip = ww[i+1][0]*ww[j+1][1];
                // index of the cell takes into accout the partition's offset of the gird fragment
                const size_t idx_gridnode = (j+cell_i[1]) + (i+cell_i[0]-gridX_offset+halo)*(size_t)gridY;

                const float incM = Wip*particle_mass;
                atomicAdd(&bgrid[SimParams::GPUGridArrayIndex::gpu_grid_idx_vis_r*pitch_g + idx_gridnode], rR*incM);
                atomicAdd(&bgrid[SimParams::GPUGridArrayIndex::gpu_grid_idx_vis_g*pitch_g + idx_gridnode], rG*incM);
                atomicAdd(&bgrid[SimParams::GPUGridArrayIndex::gpu_grid_idx_vis_b*pitch_g + idx_gridnode], rB*incM);
            }
    }

    else if(group == 2)
    {
        // P and Q are scalars (pressure and deviatoric stress measure)
        const float Jp_inv = (float)bpts[pt_idx + pitch*SimParams::PtArrIdx::idx_Jp_inv];

        double Je_tr, p_tr, q_tr;
        Eigen::Matrix2d Fe;
        for(int i=0; i<SimParams::dim; i++)
            for(int j=0; j<SimParams::dim; j++)
                Fe(i,j) = bpts[pt_idx + pitch*(SimParams::PtArrIdx::Fe00 + i*SimParams::dim + j)];
        ComputePQ(Je_tr, p_tr, q_tr, Fe);    // computes P, Q

        for (int i = -1; i <= 1; i++)
            for (int j = -1; j <= 1; j++)
            {
                const float Wip = ww[i+1][0]*ww[j+1][1];
                // index of the cell takes into accout the partition's offset of the gird fragment
                const size_t idx_gridnode = (j+cell_i[1]) + (i+cell_i[0]-gridX_offset+halo)*(size_t)gridY;
                const float incM = Wip*particle_mass;

                atomicAdd(&bgrid[SimParams::GPUGridArrayIndex::gpu_grid_idx_vis_Jpinv*pitch_g + idx_gridnode], Jp_inv*incM);
                atomicAdd(&bgrid[SimParams::GPUGridArrayIndex::gpu_grid_idx_vis_P*pitch_g + idx_gridnode], (float)p_tr*incM);
                atomicAdd(&bgrid[SimParams::GPUGridArrayIndex::gpu_grid_idx_vis_Q*pitch_g + idx_gridnode], (float)q_tr*incM);
            }
    }

    else if(group == 3)
    {
        Eigen::Matrix2d Fe;
        for(int i=0; i<SimParams::dim; i++)
            for(int j=0; j<SimParams::dim; j++)
                Fe(i,j) = bpts[pt_idx + pitch*(SimParams::PtArrIdx::Fe00 + i*SimParams::dim + j)];

        Eigen::Matrix2d E = 0.5f*(Fe.transpose()*Fe-Eigen::Matrix2d::Identity()); // GreenLagrangeStrainTensor
        Eigen::Matrix2d E_dev = dev(E);
        const float str_vonMises = (float)(std::sqrt((2.0f / 3.0f) * (E_dev.array() * E_dev.array()).sum()));
        const float str_EqvGreenLagrange = (float)std::sqrt(E.squaredNorm());

        for (int i = -1; i <= 1; i++)
            for (int j = -1; j <= 1; j++)
            {
                const float Wip = ww[i+1][0]*ww[j+1][1];
                // index of the cell takes into accout the partition's offset of the gird fragment
                const size_t idx_gridnode = (j+cell_i[1]) + (i+cell_i[0]-gridX_offset+halo)*(size_t)gridY;
                const float incM = Wip*particle_mass;
                atomicAdd(&bgrid[SimParams::GPUGridArrayIndex::gpu_grid_idx_vis_pts_density*pitch_g + idx_gridnode], Wip);
                atomicAdd(&bgrid[SimParams::GPUGridArrayIndex::gpu_grid_idx_vis_strain_EqvGreenLagrange*pitch_g + idx_gridnode], str_EqvGreenLagrange*incM);
                atomicAdd(&bgrid[SimParams::GPUGridArrayIndex::gpu_grid_idx_vis_strain_vonMises*pitch_g + idx_gridnode], str_vonMises*incM);
            }
    }

    else if(group == 4)
    {
        const float val_crushed = (utility & SimParams::fracture_crush) ? 1.0f : 0.0f;
        const float val_cracked = (utility & SimParams::status_cracked) ? 1.0f : 0.0f;
        const float thickness = (float)bpts[pt_idx + pitch * SimParams::PtArrIdx::idx_thickness];

        for (int i = -1; i <= 1; i++)
            for (int j = -1; j <= 1; j++)
            {
                const float Wip = ww[i+1][0]*ww[j+1][1];
                // index of the cell takes into accout the partition's offset of the gird fragment
                const size_t idx_gridnode = (j+cell_i[1]) + (i+cell_i[0]-gridX_offset+halo)*(size_t)gridY;
                const float incM = Wip*particle_mass;
                atomicAdd(&bgrid[SimParams::GPUGridArrayIndex::gpu_grid_idx_vis_thickness*pitch_g + idx_gridnode], thickness*incM);
                // Determine status from utility flags read at kernel start
                atomicAdd(&bgrid[SimParams::GPUGridArrayIndex::gpu_grid_idx_vis_crushed*pitch_g + idx_gridnode], val_crushed*incM);
                atomicAdd(&bgrid[SimParams::GPUGridArrayIndex::gpu_grid_idx_vis_cracked*pitch_g + idx_gridnode], val_cracked*incM);
            }
    }

    else if(group == 5)
    {
        const float val_tension = (utility & SimParams::fracture_tension) ? 1.0f : 0.0f;
        const float val_shear = (utility & SimParams::fracture_compression_shear) ? 1.0f : 0.0f;
        const float val_crush = (utility & SimParams::fracture_crush) ? 1.0f : 0.0f;

        for (int i = -1; i <= 1; i++)
            for (int j = -1; j <= 1; j++)
            {
                const float Wip = ww[i+1][0]*ww[j+1][1];
                // index of the cell takes into accout the partition's offset of the gird fragment
                const size_t idx_gridnode = (j+cell_i[1]) + (i+cell_i[0]-gridX_offset+halo)*(size_t)gridY;
                const float incM = Wip*particle_mass;
                // Determine fracture status
                atomicAdd(&bgrid[SimParams::GPUGridArrayIndex::gpu_grid_idx_fracture_tension*pitch_g + idx_gridnode], val_tension*incM);
                atomicAdd(&bgrid[SimParams::GPUGridArrayIndex::gpu_grid_idx_fracture_shear*pitch_g + idx_gridnode], val_shear*incM);
                atomicAdd(&bgrid[SimParams::GPUGridArrayIndex::gpu_grid_idx_fracture_crush*pitch_g + idx_gridnode], val_crush*incM);
            }
    }

    else if(group == 6)
    {
        // idx_xi is already raw/unscaled (see partition_kernel_compute_ice_strength) -- no factor to divide out.
        const float xi = (float)bpts[pt_idx + pitch * SimParams::PtArrIdx::idx_xi];
        const float gamma_p = (float)bpts[pt_idx + pitch * SimParams::PtArrIdx::idx_gamma_p];

        for (int i = -1; i <= 1; i++)
            for (int j = -1; j <= 1; j++)
            {
                const float Wip = ww[i+1][0]*ww[j+1][1];
                // index of the cell takes into accout the partition's offset of the gird fragment
                const size_t idx_gridnode = (j+cell_i[1]) + (i+cell_i[0]-gridX_offset+halo)*(size_t)gridY;
                const float incM = Wip*particle_mass;
                atomicAdd(&bgrid[SimParams::GPUGridArrayIndex::gpu_grid_idx_vis_ice_strength*pitch_g + idx_gridnode], xi*incM);
                atomicAdd(&bgrid[SimParams::GPUGridArrayIndex::gpu_grid_idx_vis_gamma_p*pitch_g + idx_gridnode], gamma_p*incM);
            }
    }
}





// ======================================== KERNELS RELATED TO MULTI-GPU IMPLEMENTATION


__global__ void partition_kernel_receive_subgrid(const PartitionParams pparams,
                                                 const size_t transfer_buffer_idx,
                                                 const size_t receive_offset,
                                                 const size_t receive_width)
{
    const size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t receive_elem_count = gprms.GridYTotal * receive_width;
    if(idx >= receive_elem_count) return;

    for(int i=0; i<SimParams::grid_arrays_to_clear; i++)
    {
        const size_t elem_idx = idx + i*pparams.pitch_grid + receive_offset*gprms.GridYTotal;
        const size_t buffer_idx = idx + i*pparams.transfer_buffer_width*gprms.GridYTotal;
        pparams.buffer_grid[elem_idx] += pparams.halo_transfer_buffer[transfer_buffer_idx][buffer_idx];
    }
}


__global__ void partition_kernel_check_if_transfer_needed(const PartitionParams pparams)
{
    const size_t pt_idx = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if(pt_idx >= pparams.count_pts) return;

    double* const &bpts = pparams.buffer_pts;
    const int threshold = gprms.HaloDiffusionThreshold;
    const size_t &pitch_pts = pparams.pitch_pts;
    const size_t &gridX_offset = pparams.gridX_offset;
    const size_t &gridX = pparams.partition_gridX;

    // skip if a point is disabled
    const double utility_double_pt = bpts[pt_idx + pitch_pts*SimParams::PtArrIdx::idx_utility_data];
    const unsigned long long utility_data = __double_as_longlong(utility_double_pt);
    if(utility_data & SimParams::status_disabled) return; // point is disabled

    Eigen::Vector2i cell_i  = getIntegerCellIndex(bpts[pt_idx + pitch_pts*SimParams::PtArrIdx::integer_cell_idx]);
    const int cx = cell_i.x() - (int)gridX_offset; // x-index of the cell of the point

    if(cx < ( -threshold))
    {
        const int diffusion = (int)(-cx);
        atomicMax(&pparams.pud->diffusion_distance_into_halo, diffusion);
    }
    else if(cx >= (threshold + gridX))
    {
        const int diffusion = (int)(cx-gridX);
        atomicMax(&pparams.pud->diffusion_distance_into_halo, diffusion);
    }
}



__global__ void partition_kernel_point_transfer(const PartitionParams pparams)
{
    const size_t pt_idx = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if(pt_idx >= pparams.count_pts) return;

    double* const &bpts = pparams.buffer_pts;
    const size_t &pitch_pts = pparams.pitch_pts;
    const size_t &gridX_offset = pparams.gridX_offset;
    const size_t &gridX = pparams.partition_gridX;

    // skip if a point is disabled
    const double utility_double_g2p = bpts[pt_idx + pitch_pts*SimParams::PtArrIdx::idx_utility_data];
    unsigned long long utility_data = __double_as_longlong(utility_double_g2p);
    if(utility_data & SimParams::status_disabled) return; // point is disabled

    Eigen::Vector2i cell_i  = getIntegerCellIndex(bpts[pt_idx + pitch_pts*SimParams::PtArrIdx::integer_cell_idx]);
    const int cx = cell_i.x() - (int)gridX_offset; // x-index of the cell in the current partition
    int transfer_threshold = 0;

    auto transferPoint = [&](int bufferIndex, unsigned int* counter) {
        unsigned flyIdx = atomicAdd(counter, 1);
        if (flyIdx < pparams.point_transfer_buffer_capacity)
        {
            double *ptb = pparams.point_transfer_buffer[bufferIndex];
            for(int i=0;i<SimParams::PtArrIdx::nPtsArrays;i++)
                ptb[i + flyIdx*SimParams::PtArrIdx::nPtsArrays] = bpts[pt_idx + pitch_pts*i];

            atomicAdd(pparams.disabled_points_count, 1);
            utility_data |= SimParams::status_disabled;
            bpts[pt_idx + pitch_pts * SimParams::PtArrIdx::idx_utility_data] = __longlong_as_double(utility_data);
        }
    };

    if (cx < -transfer_threshold)
    {
        transferPoint(0, &pparams.pud->transfer_to_left);
    }
    else if (cx >= (transfer_threshold + gridX))
    {
        transferPoint(1, &pparams.pud->transfer_to_right);
    }
}


__global__ void partition_kernel_receive_points(const PartitionParams pparams, const unsigned nPts,
                                                const unsigned bufferIdx)
{
    const size_t pt_idx = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if(pt_idx >= nPts) return;

    double* const &bpts = pparams.buffer_pts;
    const size_t &pitch_pts = pparams.pitch_pts;
    double* const transfer_buffer = pparams.point_transfer_buffer[bufferIdx];
    size_t idx_in_soa = pparams.count_pts + pt_idx;
    if(idx_in_soa >= pitch_pts) { gpu_error_indicator = 0xfffe; return; } // no space for incoming points

    // copy point data
    for(int i=0;i<SimParams::PtArrIdx::nPtsArrays;i++)
    {
        bpts[idx_in_soa + i*pitch_pts] = transfer_buffer[i + SimParams::PtArrIdx::nPtsArrays*pt_idx];
    }
}


// =========================================  DEVICE FUNCTIONS



__device__ void CalculateWeightCoeffs(const Eigen::Vector2d &pos, Eigen::Array2d ww[3])
{
    // optimized method of computing the quadratic (!) weight function (no conditional operators)
    Eigen::Array2d arr_v0 = 0.5 - pos.array();
    Eigen::Array2d arr_v1 = pos.array();
    Eigen::Array2d arr_v2 = pos.array() + 0.5;
    ww[0] = 0.5*arr_v0*arr_v0;
    ww[1] = 0.75-arr_v1*arr_v1;
    ww[2] = 0.5*arr_v2*arr_v2;
}





__device__ void svd2x2(const Eigen::Matrix2d &mA, Eigen::Matrix2d &mU, Eigen::Vector2d &mS, Eigen::Matrix2d &mV)
{
    double U[4], V[4], S[2];

    GivensRotation<double> gv(0, 1);
    GivensRotation<double> gu(0, 1);
    singular_value_decomposition(mA.data(), gu, S, gv);
    gu.template fill<2, double>(U);
    gv.template fill<2, double>(V);

    mU << U[0],U[1],U[2],U[3];
    mS << S[0],S[1];
    mV << V[0],V[1],V[2],V[3];
}


// deviatoric part of a diagonal matrix
__device__ Eigen::Vector2d dev_d(Eigen::Vector2d Adiag)
{
    return Adiag - Adiag.sum()/2.*Eigen::Vector2d::Constant(1.);
}

__device__ Eigen::Matrix2d dev(Eigen::Matrix2d A)
{
    return A - A.trace()/2*Eigen::Matrix2d::Identity();
}


__device__ Eigen::Matrix2d KirchhoffStress_Wolper(const Eigen::Matrix2d &F)
{
    const double kappa = gprms.kappa;
    const double &mu = gprms.mu;

    // Kirchhoff stress as per Wolper (2019)
    double Je = F.determinant();
    Eigen::Matrix2d b = F*F.transpose();
    Eigen::Matrix2d PFt = mu*(1./Je)*dev(b) + kappa*0.5*(Je*Je-1.)*Eigen::Matrix2d::Identity();
    return PFt;
}

__device__ void ComputePQ(double &Je_tr, double &p_tr, double &q_tr,
                          const Eigen::Matrix2d &F)
{
    const double kappa = gprms.kappa;
    const double &mu = gprms.mu;

    Je_tr = F.determinant();
    p_tr = -(kappa/2.) * (Je_tr*Je_tr - 1.);
    q_tr = coeff1*mu*(1./Je_tr)*dev(F*F.transpose()).norm();
}


__device__ double StrainEnergyDensity(const Eigen::Matrix2d &F)
{
    const double kappa = gprms.kappa;
    const double &mu = gprms.mu;

    // Strain energy density as per Wolper (2019)
    const double trace = (F.transpose()*F).trace();
    double J = F.determinant();
    double term1 = mu*(trace/J - SimParams::dim);
    double term2 = kappa*((J*J-1.)*0.5 - log(J));
    const double result = 0.5*(term1+term2);
    return result;
}




// ============================================================================
// HELPER FUNCTIONS
// ============================================================================


__device__ Eigen::Vector2i getIntegerCellIndex(double raw_value)
{
    const long long cell = __double_as_longlong(raw_value);
    Eigen::Vector2i cell_i((int)(cell & 0xffffffff), (int)(cell >> 32));
    return cell_i;
}





__device__ void ReconstructDeformationGradient(Eigen::Matrix2d &Fe, double &Jp_inv,
                                                    const double &p_new, const double &q_new,
                                                    const double &q_tr, const double &Je_tr)
{
    const double &kappa = gprms.kappa;

    Eigen::Vector2d vSigma, vSigmaSquared;
    Eigen::Matrix2d U, V;
    svd2x2(Fe, U, vSigma, V);
    vSigmaSquared = vSigma.array().square().matrix();

    // determinant → from p
    const double Je_new = sqrt(-2*p_new/kappa + 1);

    // --- Step 1: old deviatoric part (trace-based)
    double mean_old = vSigmaSquared.sum() * 0.5;
    Eigen::Vector2d dev_old = vSigmaSquared - Eigen::Vector2d::Constant(mean_old);

    // --- Step 2: scale deviatoric part using q
    Eigen::Vector2d dev_new = Eigen::Vector2d::Zero();
    if(q_tr > 1e-12)
        dev_new = (q_new/q_tr) * dev_old;

    // --- Step 3: extract scalar deviatoric magnitude (2D special structure)
    double d = 0.5 * (dev_new[0] - dev_new[1]);
    // equivalent to (λ1 - λ2)/2

    // --- Step 4: compute mean from determinant constraint
    // m^2 - d^2 = J^2  →  m = sqrt(J^2 + d^2)
    double m = sqrt(Je_new*Je_new + d*d);

    // --- Step 5: reconstruct eigenvalues (λ = σ^2)
    Eigen::Vector2d vSigma_new_squared;
    vSigma_new_squared[0] = m + d;
    vSigma_new_squared[1] = m - d;

    // --- Safety (very important)
    vSigma_new_squared = vSigma_new_squared.cwiseMax(1e-12);

    // --- Back to singular values
    Eigen::Vector2d vSigma_new = vSigma_new_squared.array().sqrt().matrix();

    // --- Reconstruct F
    Fe = U * vSigma_new.asDiagonal() * V.transpose();
    // Hard limit on compaction -- past this, the point is treated as fully
    // consolidated, and further crushing no longer accumulates any more
    // plastic volume change.
    constexpr double Jp_inv_hard_limit = 100.0;
    Jp_inv = fmin(Jp_inv * (Je_new / Je_tr), Jp_inv_hard_limit);


#ifdef ENABLE_NAN_CHECKS
    // check if something went wrong
    if(isnan(Fe(0,0)) || isnan(Fe(1,0)) || isnan(Fe(0,1)) || isnan(Fe(1,1)))
    {
        gpu_error_indicator |= error_code_point_Fe_nan;
    }
#endif
}





// Piecewise-linear fractured/yield surface: steep Drucker-Prager leg from
// DP_threshold_p, up to q_crossover (computed by the caller, see PlasticProjection);
// past that, a shallower second leg anchored exactly at the crossover
// point, so the kink is structural (min() of two lines meeting at one
// point) rather than something that has to be separately calibrated.
__device__ double Q_From_Yield_Surface(double p_tr, double q_crossover)
{
    const double DP_threshold_p = gprms.DP_threshold_p;

    if(p_tr < DP_threshold_p) return 0;

    const double q_steep = (p_tr - DP_threshold_p) * gprms.DP_tan_phi;
    const double p_crossover = DP_threshold_p + q_crossover / gprms.DP_tan_phi;
    const double q_second_leg = q_crossover + gprms.IceYieldFrictionAngle_tan * (p_tr - p_crossover);
    return fmin(q_steep, q_second_leg);
}


// Vertical temperature profile reconstruction, formula (R) --
// _python_code/heat_equation_spec.md section 1. Pure read-out, no state update.
__device__ double reconstruct_T(double z, double H, double Ts, double A1, double A2, double A3)
{
    return Ts + (SimParams::T_bottom - Ts) * (z / H)
    + A1*sin(SimParams::pi*z/H) + A2*sin(2.0*SimParams::pi*z/H) + A3*sin(3.0*SimParams::pi*z/H);
}

// Frankenstein & Garner (1967). T in degC, S in ppt. Returns Vb/V (fraction, clamped to [0,1]).
__device__ double brine_volume_frankenstein_garner(double T, double S)
{
    // Macaulay bracket <-T> = max(-T, eps), NOT |T| -- for T>=0 this saturates vb to 1 instead of
    // wrongly folding a warm T onto the same low vb as an equally-cold -T.
    double Tabs = fmax(-T, 1e-6);
    return fmin(1.0, fmax(0.0, S * (49.185/Tabs + 0.532) * 1e-3));
}

// Timco & O'Brien (1994). vb is Vb/V (fraction, 0-1). Returns sigma_f in Pa.
__device__ double flexural_strength_timco_obrien(double vb)
{
    return 1.76e6 * exp(-5.88 * sqrt(vb));
}


