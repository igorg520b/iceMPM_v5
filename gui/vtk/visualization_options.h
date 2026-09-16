// visualization_options.h
#ifndef VISUALIZATION_OPTIONS_H
#define VISUALIZATION_OPTIONS_H

#include <map>
#include <string>
#include <utility>

struct VisOpt {

    enum Type {
        none,
        regions,
        // points must be available
        pt_status,
        pt_color,
        pt_Jp_inv,
        pt_thickness,
        pt_partitions,
        pt_fracture_type,
        pt_ice_strength,
        pt_gamma_p,
        pt_Jp_inv_max,
        // grid-based visualizations
        grid_mass,
        grid_pt_count,
        grid_Jpinv,
        grid_P,
        grid_Q,
        grid_colors,
        grid_vnorm,
        grid_cracked,
        grid_thickness,
        grid_ice_strength,
        grid_gamma_p,
        grid_fracture_type,
        grid_schematic,
        str_EqvGreenLagrange,
        str_vonMises,
        // visualization of external currents/forces
        glo12_ocean,
        carra1_wind,
        carra1_temperature,
        v_glo12_thickness
    };

    enum HedgehogType {
        hedgehog_none,
        hedgehog_wind,
        hedgehog_ocean,
        hedgehog_applied_shear
    };

    inline static const std::map<VisOpt::Type, std::pair<std::string,std::string>> descriptions = {
        {none,              {"none",            "Basic Outline"}},
        {regions,           {"regions",         "Regions"}},
        {pt_status,         {"pt_status",       "(point) Status"}},
        {pt_color,          {"pt_color",        "(point) Color"}},
        {pt_Jp_inv,         {"pt_Jp_inv",       "(point) Change in Surf. Density (Jp_inv-1)"}},
        {pt_thickness,      {"pt_thickness",    "(point) Thickness"}},
        {pt_partitions,     {"pt_partitions",   "(point) GPU Partitions"}},
        {pt_fracture_type,  {"pt_fracture_type","(point) Fracture Type"}},
        {pt_ice_strength,   {"pt_ice_strength", "sigma_flexural"}},
        {pt_gamma_p,        {"pt_gamma_p",      "Accum. Plastic Shear Strain"}},
        {pt_Jp_inv_max,     {"pt_Jp_inv_max",   "Historical Peak Jp_inv (Crushing)"}},

        {grid_mass,             {"grid_mass",               "Mass"}},
        {grid_pt_count,         {"grid_pt_count",           "Point Density per Cell"}},
        {grid_Jpinv,            {"grid_Jpinv",              "Change in Surf. Density (Jp_inv - 1)"}},
        {grid_P,                {"grid_P",                  "In-plane Pressure P"}},
        {grid_Q,                {"grid_Q",                  "Deviatoric Stress Q"}},
        {grid_colors,           {"grid_colors",             "Natural Color"}},
        {grid_vnorm,            {"grid_vnorm",              "Ice Velocity Norm"}},
        {grid_cracked,          {"grid_cracked",            "Cracked/Crushed Material"}},
        {grid_thickness,        {"grid_thickness",          "Ice Thickness"}},
        {grid_ice_strength,     {"grid_ice_strength",       "Ice Strength (grid)"}},
        {grid_gamma_p,          {"grid_gamma_p",            "Accum. Plastic Shear Strain (grid)"}},
        {grid_fracture_type,    {"grid_fracture_type",      "Fracture Type"}},
        {grid_schematic,        {"grid_schematic",          "Schematic Colors"}},

        {str_EqvGreenLagrange,  {"str_EqvGreenLagrange",    "Green-Lagrange Strain"}},
        {str_vonMises,          {"str_vonMises",            "von Mises Strain"}},
        {glo12_ocean,          {"glo12_ocean",            "Ocean Current Velocity Norm"}},
        {carra1_wind,           {"carra1_wind",             "Wind Velocity Norm"}},
        {carra1_temperature, {"carra1_temperature",   "T (C)"}},
        {v_glo12_thickness,     {"v_glo12_thickness",       "GLO12 Ice Thickness (m)"}}
    };

    inline static const std::map<VisOpt::HedgehogType, std::pair<std::string,std::string>> hedgehog_descriptions = {
        {hedgehog_none,             {"none",                    "None"}},
        {hedgehog_wind,             {"wind",                    "Wind"}},
        {hedgehog_ocean,            {"ocean",                   "Ocean"}},
        {hedgehog_applied_shear,    {"applied_shear",           "Applied Shear"}}
    };

    static constexpr int max_vis_opts = 50;

    // Plain [from, to] color-mapping bounds per visualization option -- every
    // renderer computes frac = (val - range_from[opt]) / (range_to[opt] -
    // range_from[opt]) and feeds that into the colormap (ColorMap::getColor
    // clamps to [0,1] internally, so over/under-range values just saturate).
    // Replaces the old single log-scale "ranges[]" (val/10^ranges[...],
    // implicitly either [0, range] or [-range, range] depending on the
    // option) -- that scheme couldn't represent options like ice thickness
    // or Celsius temperature, which aren't naturally bounded at zero.
    inline static double range_from[max_vis_opts] = {};
    inline static double range_to[max_vis_opts] = {};

    inline static double transparency_coeffs[max_vis_opts] = {};

    static constexpr int max_hedgehog_opts = 4;
    inline static double hedgehog_ranges[max_hedgehog_opts] = {};

    inline static const std::string state_file_name = "plateMPM_vis_state.txt";

    static void LoadVisualizationState();
    static void SaveVisualizationState();
    static void SetDefaultValues();
};

#endif

