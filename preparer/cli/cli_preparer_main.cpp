#include <iostream>
#include <functional>
#include <string>
#include <filesystem>
#include <atomic>
#include <thread>
#include <chrono>

#include <cxxopts.hpp>
#include <fmt/format.h>
#include "parameters_sim.h"
#include "host_side_data.h"
#include "data_preparer.h"
#include "parameterparser.h"

namespace fs = std::filesystem;

int main(int argc, char *argv[])
{
    std::string prepare_parameter_filename;
    // parse options
    std::string parameter_filename;
    std::string resume_filename; // Defaults to empty

    cxxopts::Options options("Ice MPM", "CLI version of MPM simulation");
    options.add_options()
        // Define options and link them to variables
        ("parameter", "Preapare parameter file (JSON, required)", cxxopts::value<std::string>(parameter_filename));

    // Mark 'parameter' as the positional argument
    options.parse_positional({"parameter"});
    auto result = options.parse(argc, argv); // Parse arguments

    // Check if required parameter file was provided
    if (!result.count("parameter")) {
        LOGR("Error: Parameter file argument is required."); // Minimal error output
        throw std::runtime_error("Parameter file is required."); // Still need to signal failure
    }
    parameter_filename = result["parameter"].as<std::string>();

    // Check if the provided path is a directory (if so, append "simulation.json")
    if (fs::is_directory(parameter_filename)) {
        prepare_parameter_filename = (fs::path(parameter_filename) / "simulation.json").string();
    } else {
        prepare_parameter_filename = parameter_filename;
    }

    LOGR("Loading parameters from: {}", prepare_parameter_filename);

    try {
        // Single consolidated JSON file: ParameterParser reads preparer-only
        // fields (images, PointsPerCell, ...), SimParams reads sim-runtime
        // fields (projection, CARRA1/GLO12 paths, start date, ...). Both
        // parsers open the same file and simply ignore keys they don't
        // recognize.
        ParameterParser params;
        params.LoadParamsFile(prepare_parameter_filename);

        // Construct file paths
        std::string configDir = params.ConfigFileDirectory;
        std::string landmaskPath = params.ImageLandMask.empty() ? "" : (configDir + "/" + params.ImageLandMask);
        std::string colorPath = configDir + "/" + params.ImageColor;
        std::string icemaskPath = params.ImageIceMask.empty() ? "" : (configDir + "/" + params.ImageIceMask);
        std::string crushedmaskPath = params.ImageCrushedMask.empty() ? "" : (configDir + "/" + params.ImageCrushedMask);
        std::string crackedmaskPath = params.ImageCrackedMask.empty() ? "" : (configDir + "/" + params.ImageCrackedMask);
        std::string thicknessmaskPath = params.ImageThicknessMask.empty() ? "" : (configDir + "/" + params.ImageThicknessMask);
        std::string footprintmaskPath = params.ImageFootprintMask.empty() ? "" : (configDir + "/" + params.ImageFootprintMask);
        std::string analysismaskPath = params.AnalysisRegionMask.empty() ? "" : (configDir + "/" + params.AnalysisRegionMask);

        // Create HostSideData and DataPreparer
        HostSideData hsd;

        // preparer_cli renders a single date and exits -- no benefit from
        // background-preloading a "next" frame that will never be consumed.
        hsd.windInterp.SetPreloadEnabled(false);
        hsd.currentInterp.SetPreloadEnabled(false);

        // Projection (and CARRA1/GLO12 paths, start date) must be loaded
        // BEFORE PrepareGridAndPoints, since cellsize is now derived from
        // the projection instead of a separately specified dimension.
        std::filesystem::path simConfigDir = fs::path(prepare_parameter_filename).parent_path();
        std::vector<std::string> carra1Files;
        std::map<std::string, std::string> simParseResult = hsd.prms.ParseFile(prepare_parameter_filename, &carra1Files);
        // The one and only DirectoryManager instance for this run -- see
        // ParameterParser's header comment for why it doesn't have its own.
        hsd.dirs.Configure(simParseResult["RuntimeDirectory"], simParseResult["ProjectName"]);
        std::string projectDir = hsd.dirs.ProjectDirectory();
        fs::create_directories(projectDir);

        // Relative to this project's own snapshots directory (RuntimeDirectory-
        // derived), NOT the simulation.json directory -- mirrors how model.cpp
        // resolves the simulation-runtime "Snapshot" key.
        std::string snapshotToLoadPath;
        if (!params.LoadSnapshotForSpinUp.empty()) {
            fs::path p(params.LoadSnapshotForSpinUp);
            snapshotToLoadPath = (p.is_relative() ? (fs::path(hsd.dirs.SnapshotsDirectory()) / p) : p).string();
        }

        // Point generation sources thickness from the image mask, not GLO12 --
        // GLO12 data is loaded here only for currents/tides, ahead of PrepareGridAndPoints.
        if (simParseResult.count("GLO12Data") && !simParseResult["GLO12Data"].empty()) {
            fs::path glo12Path = simConfigDir / simParseResult["GLO12Data"];
            if (fs::exists(glo12Path)) {
                hsd.currentInterp.SetGLO12Path(glo12Path.string());
                LOGR("GLO12 Ocean Data loaded from: {}", glo12Path.string());
            } else {
                LOGR("Warning: GLO12Data path not found: {}", glo12Path.string());
            }
        }

        if (simParseResult.count("GLO12ThicknessData") && !simParseResult["GLO12ThicknessData"].empty()) {
            fs::path glo12ThicknessPath = simConfigDir / simParseResult["GLO12ThicknessData"];
            if (fs::exists(glo12ThicknessPath)) {
                hsd.currentInterp.SetGLO12ThicknessPath(glo12ThicknessPath.string());
                LOGR("GLO12 Thickness Data loaded from: {}", glo12ThicknessPath.string());
            } else {
                LOGR("Warning: GLO12ThicknessData path not found: {}", glo12ThicknessPath.string());
            }
        }

        DataPreparer preparer(hsd);

        LOGR("Starting CLI Data Preparation...");

        // Execute preparation
        // allocate_dense_grid = false (CLI optimization)
        preparer.PrepareGridAndPoints(landmaskPath, colorPath, icemaskPath, crushedmaskPath, crackedmaskPath,
                                      projectDir, params.PointsPerCell,
                                      params.ThicknessFrom, params.ThicknessTo,
                                      params.ProportionOfCrackedPoints, params.StdDevOfThickness,
                                      thicknessmaskPath,
                                      false, false, params.GeneratePoints,
                                      footprintmaskPath, analysismaskPath, snapshotToLoadPath);

        LOGR("Preparation Complete.");
        LOGR("  Grid Dimensions: {} x {}", hsd.prms.GridXTotal, hsd.prms.GridYTotal);
        LOGR("  Total Points: {}", hsd.prms.nPtsInitial);
        LOGR("  Data saved to: {}", hsd.data_directory);

        // Load CARRA1 data now -- must happen AFTER PrepareGridAndPoints:
        // WindInterpolator::PrecomputeGridMapping() (triggered by
        // SetCARRA1Path below) sizes its mapping arrays from
        // prms.GridXTotal/GridYTotal, which are only set once
        // DetermineExtents() runs inside PrepareGridAndPoints. Loading CARRA1
        // any earlier sizes those arrays to 0 and crashes the very next time
        // a wind frame is loaded. (Mirrors preparer_mainwindow.cpp.)
        bool useCARRA1 = false;
        for (const std::string& carra1File : carra1Files) {
            fs::path p(carra1File);
            if (p.is_relative()) p = simConfigDir / p;
            if (fs::exists(p)) {
                hsd.windInterp.SetCARRA1Path(p.string());
                LOGR("CARRA1 Wind Data loaded from: {}", p.string());
                useCARRA1 = true;
            } else {
                LOGR("Warning: CARRA1Data path not found: {}", p.string());
            }
        }
        hsd.prms.UseWindData = useCARRA1;
        if (useCARRA1) hsd.windInterp.SetEnabled(true);

        // Seeds every point's idx_Ts_old from the real CARRA1 temperature at
        // PreSimulationStartTime, before the spin-up loop below advances it
        // step by step. Without this, idx_Ts_old stays at whatever placeholder
        // point generation left it at (0.0), corrupting the very first
        // ThermalSpinUpStep's rate-of-change calculation. Mirrors
        // preparer_mainwindow.cpp's call at project load.
        if (hsd.prms.PreSimulationStartTime > 0) {
            preparer.InitThermalState();
        }

        // Thermal spin-up (best-effort): only if CARRA1 data and a valid
        // [PreSimulationStartDate, SimulationStartDate) window are configured --
        // prepare-only configs (no spin-up prerequisites) continue to work
        // exactly as before, just without this step.
        if (hsd.windInterp.GetCARRA1NumFrames() <= 0) {
            LOGR("No CARRA1 data loaded -- skipping thermal spin-up.");
        } else if (hsd.prms.PreSimulationStartTime <= 0 ||
                   hsd.prms.SimulationStartTime <= hsd.prms.PreSimulationStartTime) {
            LOGR("PreSimulationStartDate/SimulationStartDate not configured -- skipping thermal spin-up.");
        } else {
            const std::string tempCacheFile = hsd.windInterp.TemperatureCacheFilename();
            const std::string windCacheFile = hsd.windInterp.WindCacheFilename();

            if (!fs::exists(tempCacheFile)) {
                LOGR("Thermal cache not found ({}) -- generating...", tempCacheFile);
                fs::create_directories(fs::path(tempCacheFile).parent_path());
                fs::create_directories(fs::path(windCacheFile).parent_path());
                preparer.GenerateTempWindCache(tempCacheFile, windCacheFile,
                    [](int frameIdx, int numFrames) {
                        if (frameIdx % 25 == 0) LOGR("  cache frame {}/{}", frameIdx, numFrames);
                        return true;
                    });
                LOGR("Thermal cache generated.");
            } else {
                LOGR("Thermal cache found ({}) -- reusing.", tempCacheFile);
            }

            LOGR("Running thermal spin-up...");
            auto [firstFrameIdx, lastFrameIdx] = hsd.windInterp.GetFrameRangeForTimeSpan(
                hsd.prms.PreSimulationStartTime, hsd.prms.SimulationStartTime);
            const long long N = lastFrameIdx - firstFrameIdx;
            for (long long s = 1; s <= N; ++s) {
                preparer.ThermalSpinUpStep((int)(firstFrameIdx + s));
                if (s % 50 == 0) LOGR("  spin-up step {}/{}", s, N);
            }
            preparer.ThermalSpinUpSave(); // computes+persists idx_xi, prints average
            LOGR("Thermal spin-up complete.");
        }

    } catch (const std::exception& e) {
        LOGR("Error during preparation: {}", e.what());
        return 1;
    }

    return 0;
}
