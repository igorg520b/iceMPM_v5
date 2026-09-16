#ifndef PARAMETERPARSER_H
#define PARAMETERPARSER_H

#include <string>

#include <rapidjson/reader.h>
#include <rapidjson/document.h>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>

#include <Eigen/Core>

// Preparer-only fields (images, PointsPerCell, ...) -- deliberately does not
// touch RuntimeDirectory/ProjectName/DirectoryManager at all: those are
// SimParams/runtime concerns (see simulation/data_manager/directory_manager.h),
// parsed and owned by SimParams::ParseFile + HostSideData::dirs instead, so
// there is exactly one DirectoryManager instance per run, not two. Callers
// needing the project directory should use hsd.dirs.ProjectDirectory().
struct ParameterParser
{
    std::string ImageColor, ImageCrushedMask, ImageCrackedMask, ImageIceMask, ImageLandMask, ImageThicknessMask;
    // Black = open boundary condition, overlaid with priority over the
    // ordinary land/water landmask (see SimParams::OpenBoundaryIndicator).
    // May be any resolution sharing ImageColor's aspect ratio -- DataPreparer
    // resizes it automatically to the working resolution if needed.
    std::string ImageFootprintMask;
    // Region mask for post-simulation diagnostics (mean/max ice strength,
    // wind, current within the marked cells) -- see visualizer's "Export
    // Region Diagnostics". White = included, black = excluded. Re-running
    // preparer with GeneratePoints=false lets this be updated after a
    // simulation has already completed, without touching points/output.
    std::string AnalysisRegionMask;
    std::string ConfigFileDirectory;  // directory containing the JSON config file
    int PointsPerCell = 5;

    // When false, skip Poisson-disk point generation entirely -- grid
    // sizing (from ImageColor) and any grid-level rendering (thickness,
    // temperature, wind, current) still run. Useful for testing the
    // reanalysis-data rendering pipeline without needing ImageIceMask/
    // ImageLandMask or paying for point sampling.
    bool GeneratePoints = true;

    // When non-empty, skip point generation entirely (overriding
    // GeneratePoints) and instead load points from this previously-saved
    // .h5 snapshot -- relative to this project's own snapshots directory
    // (hsd.dirs.SnapshotsDirectory()) if not absolute, matching how
    // model.cpp resolves the simulation-runtime "Snapshot" key -- NOT
    // relative to ConfigFileDirectory.
    // Lets thermal spin-up (and anything else the preparer does) run
    // repeatedly on a fixed, previously generated point set -- e.g. the
    // s00000.h5 that PopulatePoints_RAM_Optimized itself writes -- instead
    // of regenerating geometry (with its own random Poisson-disk sampling
    // and thickness noise) from scratch every time. Point to a project's
    // OWN pristine, pre-spinup snapshot from a DIFFERENT project's config
    // (different RuntimeDirectory/ProjectName) so ThermalSpinUpSave's
    // output doesn't overwrite the source snapshot you're loading from.
    std::string LoadSnapshotForSpinUp;

    double ThicknessFrom = 1.0;  // default: no scaling
    double ThicknessTo = 1.0;    // default: no scaling

    // Random perturbation parameters
    double ProportionOfCrackedPoints = 0.0;
    double StdDevOfThickness = 0.0;

    void LoadParamsFile(std::string fileName);
};

#endif // PARAMETERPARSER_H
