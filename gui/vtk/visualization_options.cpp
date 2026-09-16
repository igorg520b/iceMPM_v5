// visualization_options.cpp
#include "visualization_options.h"
#include <fstream>
#include <sstream>
#include "parameters_sim.h"

void VisOpt::SetDefaultValues()
{
    // Initialize all to 0
    for(int i = 0; i < max_vis_opts; ++i) {
        range_from[i] = 0.0;
        range_to[i] = 0.0;
        transparency_coeffs[i] = 0.0;
    }
    for(int i = 0; i < max_hedgehog_opts; ++i) {
        hedgehog_ranges[i] = 1.0;
    }

    hedgehog_ranges[1] = 1.3;
    hedgehog_ranges[2] = 30;

    // Defaults below are chosen to match the visual behavior of the old
    // log-scale scheme (val / 10^ranges[...]) at the moment it was replaced:
    // options that used to be symmetric about zero ([-range, range]) get
    // symmetric [from, to]; options that used to be one-sided ([0, range])
    // get [0, to]. From here on they're independently tunable via the GUI.
    range_from[(int)VisOpt::pt_Jp_inv] = range_from[(int)VisOpt::grid_Jpinv] = -0.1;
    range_to[(int)VisOpt::pt_Jp_inv]   = range_to[(int)VisOpt::grid_Jpinv]   = 0.1;

    range_from[(int)VisOpt::grid_P] = -562341.3;
    range_to[(int)VisOpt::grid_P]   = 562341.3;
    range_from[(int)VisOpt::grid_Q] = 0.0;
    range_to[(int)VisOpt::grid_Q]   = 562341.3;

    range_from[(int)VisOpt::grid_mass] = 0.0;
    range_to[(int)VisOpt::grid_mass]   = 5623.4;
    range_from[(int)VisOpt::grid_pt_count] = 0.0;
    range_to[(int)VisOpt::grid_pt_count]   = 1.0;
    range_from[(int)VisOpt::grid_vnorm] = 0.0;
    range_to[(int)VisOpt::grid_vnorm]   = 1.0;

    range_from[(int)VisOpt::str_vonMises] = range_from[(int)VisOpt::str_EqvGreenLagrange] = 0.0;
    range_to[(int)VisOpt::str_vonMises]   = range_to[(int)VisOpt::str_EqvGreenLagrange]   = 0.001;

    // Ice thickness: physical units, not naturally bounded at zero. Was
    // previously the special-cased pt_thickness_range_min/max; the preparer
    // overrides these from simulation.json's ThicknessFrom/ThicknessTo.
    range_from[(int)VisOpt::pt_thickness] = range_from[(int)VisOpt::grid_thickness] = 0.7;
    range_to[(int)VisOpt::pt_thickness]   = range_to[(int)VisOpt::grid_thickness]   = 4.0;

    // Ice strength (idx_xi, Pa): raw, unscaled Timco & O'Brien flexural
    // strength, bounded in (0, 1.76e6] Pa. IceStrengthFactor/
    // IceStrengthFactorTensile are applied where idx_xi is consumed
    // (PlasticProjection), not baked into idx_xi itself.
    range_from[(int)VisOpt::pt_ice_strength] = range_from[(int)VisOpt::grid_ice_strength] = 0.0;
    range_to[(int)VisOpt::pt_ice_strength]   = range_to[(int)VisOpt::grid_ice_strength]   = 1.76e6;

    // Accumulated plastic shear strain (gamma_p): brand-new quantity, no
    // established physical bound yet -- placeholder, retune once real data exists.
    range_from[(int)VisOpt::pt_gamma_p] = range_from[(int)VisOpt::grid_gamma_p] = 0.0;
    range_to[(int)VisOpt::pt_gamma_p]   = range_to[(int)VisOpt::grid_gamma_p]   = 1.0;

    // Historical peak Jp_inv: always >= 1 (starts at 1, hard-capped at 100
    // in ReconstructDeformationGradient) -- placeholder range, retune once
    // real data exists.
    range_from[(int)VisOpt::pt_Jp_inv_max] = 1.0;
    range_to[(int)VisOpt::pt_Jp_inv_max]   = 2.0;

    range_from[(int)VisOpt::carra1_wind] = 0.0;
    range_to[(int)VisOpt::carra1_wind]   = 17.78;
    range_from[(int)VisOpt::glo12_ocean] = 0.0;
    range_to[(int)VisOpt::glo12_ocean]   = 0.56;

    // Surface temperature is rendered in Celsius for visualization (CARRA1
    // gives Kelvin -- see host_side_data.cpp's RenderGridRaster). Celsius
    // isn't bounded at zero, which is exactly the kind of range the old
    // scheme couldn't express -- a plain [from, to] handles it directly.
    range_from[(int)VisOpt::carra1_temperature] = -25.0;
    range_to[(int)VisOpt::carra1_temperature]   = 10.0;

    // GLO12 ice thickness, meters -- live/time-varying (via the flow-time
    // slider), unlike the one-shot prepare-time thickness render. 0 both
    // where GLO12 has no coverage and for dates outside GLO12's time range.
    range_from[(int)VisOpt::v_glo12_thickness] = 0.0;
    range_to[(int)VisOpt::v_glo12_thickness]   = 3.0;

    transparency_coeffs[(int)VisOpt::grid_Jpinv] = transparency_coeffs[(int)VisOpt::pt_Jp_inv] = 1.0;
    transparency_coeffs[(int)VisOpt::grid_fracture_type] = 0.6;
    transparency_coeffs[(int)VisOpt::grid_schematic] = 0.6;
}

void VisOpt::LoadVisualizationState()
{
    SetDefaultValues(); // Start with defaults

    std::ifstream in(state_file_name);
    if (!in.is_open()) {
        LOGR("Visualization state file not found or could not be opened: {}", state_file_name);
        return;
    }

    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;

        std::istringstream iss(line);
        std::string keyName;

        // Try the current format first: Key From To Transparency.
        double fromVal, toVal, transVal;
        if (iss >> keyName >> fromVal >> toVal >> transVal) {
            bool matched = false;
            for (const auto& pair : descriptions) {
                if (pair.second.first == keyName) {
                    range_from[pair.first] = fromVal;
                    range_to[pair.first] = toVal;
                    transparency_coeffs[pair.first] = transVal;
                    matched = true;
                    break;
                }
            }
            if (matched) continue;
        }

        // Not a vis-option line (or didn't parse as one) -- try hedgehog
        // format: Key Scale 0. Also covers stale pre-[from,to] state files,
        // whose vis-option lines ("Key Range Transparency") won't match the
        // 4-token parse above and are simply skipped, leaving that entry at
        // its default.
        std::istringstream iss2(line);
        std::string keyName2;
        double scaleVal, unused;
        if (iss2 >> keyName2 >> scaleVal >> unused) {
            for (const auto& pair : hedgehog_descriptions) {
                if (pair.second.first == keyName2) {
                    hedgehog_ranges[pair.first] = scaleVal;
                    break;
                }
            }
        }
    }
    LOGR("Visualization state loaded from {}", state_file_name);
}

void VisOpt::SaveVisualizationState()
{
    std::ofstream out(state_file_name);
    if (!out.is_open()) {
        LOGR("Could not open visualization state file for writing: {}", state_file_name);
        return;
    }

    out << "# Visualization State (Key From To Transparency)\n";
    for (const auto& pair : descriptions) {
        if (pair.first == none) continue;

        out << pair.second.first << " "
            << range_from[pair.first] << " "
            << range_to[pair.first] << " "
            << transparency_coeffs[pair.first] << "\n";
    }

    out << "# Hedgehog State (Key Range 0)\n";
    for (const auto& pair : hedgehog_descriptions) {
        if (pair.first == hedgehog_none) continue;
        
        out << pair.second.first << " "
            << hedgehog_ranges[pair.first] << " "
            << 0.0 << "\n";
    }
    
    LOGR("Visualization state saved to {}", state_file_name);
}
