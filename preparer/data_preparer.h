#ifndef DATA_PREPARER_H
#define DATA_PREPARER_H

#include <vector>
#include <string>
#include <array>
#include <filesystem>
#include <functional>

#include "host_side_data.h"
#include "parameters_sim.h"

class DataPreparer
{
public:
    DataPreparer(HostSideData& hsd);

    // Bitmask flags for m_flags
    static constexpr uint8_t FLAG_WATER   = 1 << 0;
    static constexpr uint8_t FLAG_ICE     = 1 << 1;
    static constexpr uint8_t FLAG_CRUSHED = 1 << 2;
    static constexpr uint8_t FLAG_CRACKED = 1 << 3;
    static constexpr uint8_t FLAG_OPEN_BOUNDARY = 1 << 4;
    static constexpr uint8_t FLAG_ANALYSIS_REGION = 1 << 5;

    // cellsize is derived from hsd.prms.proj -- must be populated (ParseFile) before calling this.
    // Ice thickness always comes from fileNameThicknessMask (scaled between thicknessFrom/thicknessTo).
    // fileNameSnapshotToLoad: when non-empty, skip point generation entirely
    // (regardless of generatePoints) and load points from this .h5 snapshot
    // instead -- see ParameterParser::LoadSnapshotForSpinUp.
    void PrepareGridAndPoints(std::string fileNameLandMask, std::string fileNameColor,
                              std::string fileNameIceMask, std::string fileNameCrushedMask,
                              std::string fileNameCrackedMask,
                              std::string projectDirectory, int pointsPerCell,
                              double thicknessFrom, double thicknessTo,
                              double probCracked, double stdDevThickness,
                              std::string fileNameThicknessMask,
                              bool allocate_dense_grid,
                              bool reload_after_save = true,
                              bool generatePoints = true,
                              std::string fileNameFootprintMask = "",
                              std::string fileNameAnalysisMask = "",
                              std::string fileNameSnapshotToLoad = "");

    // Sets every point's idx_Ts_old to the real CARRA1 temperature at PreSimulationStartTime.
    // Call once, right after CARRA1 is loaded, if PreSimulationStartTime is configured.
    void InitThermalState();

    // Advances every point's (A1, A2, A3, Ts_old) using CARRA1 frame frameIdx,
    // unblended. Call for frameIdx = firstFrameIdx+1 .. lastFrameIdx, in order.
    void ThermalSpinUpStep(int frameIdx);

    // Computes idx_xi from each point's CURRENT (A1, A2, A3, Ts_old, thickness)
    // -- the CPU mirror of the second half of partition_kernel_compute_ice_strength
    // (simulation/kernels.cu). Call after ThermalSpinUpStep for the same frame.
    // Keep in sync with kernels.cu if the strength formula changes.
    void ComputeIceStrengthCPU();

    // Saves the current point state into the initial snapshot (s00000.h5).
    // Also (re)computes idx_xi from the final thermal state and logs its
    // average across all points -- see ComputeIceStrengthCPU.
    void ThermalSpinUpSave();

    // Renders and writes the combined CARRA1 temperature+wind HDF5 cache for
    // the current grid size (one combined RenderTempAndWindFrameUncached call
    // per frame -- see WindInterpolator). Qt-free, usable from both the GUI
    // (preparer) and the headless preparer_cli. progressFn, if provided, is
    // called once per frame as (frameIdx, numFrames); returning false aborts
    // after that frame (whatever was written so far remains a valid partial
    // cache). Throws std::runtime_error on failure -- callers decide how to
    // surface that (GUI: catch and qFatal; CLI: let main()'s top-level catch
    // report it).
    void GenerateTempWindCache(const std::string& tempCacheFile,
                                const std::string& windCacheFile,
                                const std::function<bool(int,int)>& progressFn = {});

private:
    HostSideData& hsd;

    int m_width = 0;
    int m_height = 0;

    // Reused across ThermalSpinUpStep calls to avoid reallocating a gx*gy buffer every step.
    std::vector<float> m_spinup_temp_scratch;

    // Ice region bounding box (detected in PrepareGrid)
    int IceRegionOffsetX = 0;
    int IceRegionOffsetY = 0;
    int IceRegionWidth = 0;
    int IceRegionHeight = 0;
    std::vector<uint8_t> m_flags;      // Bitpacked flags (water, ice, crushed, cracked)
    std::vector<uint8_t> m_thickness;  // Thickness values (0-255), PNG-mask/random path
    std::vector<uint8_t> m_color;      // RGB color data (3 bytes per pixel)

    // Auto-resizes the source PNG (nearest-neighbor) to m_width x m_height if needed.
    // Returns true if the image was loaded and processed.
    bool ProcessMaskLayer(const std::string& filename, uint8_t flag, bool invert = false, int threshold = 128);

    // Like ProcessMaskLayer, but black = open boundary unconditionally, priority over other flags.
    bool ProcessFootprintMask(const std::string& filename);

    // Nearest-neighbor resize of an RGBA buffer (e.g. a full-resolution footprint mask).
    static void ResizeMaskNearest(const std::vector<uint8_t>& src, int srcW, int srcH,
                                   std::vector<uint8_t>& dst, int dstW, int dstH);

    void PrepareGrid(std::string projectDirectory, bool allocate_dense_grid);

    void PopulatePoints_RAM_Optimized(int pointsPerCell, double thicknessFrom, double thicknessTo,
                                      double probCracked, double stdDevThickness, bool compress = true);

    // Logs total ice volume (sum over points of ParticleArea * thickness) for
    // whatever points currently sit in hsd.hssoa -- used after loading points
    // from a previous snapshot, where PopulatePoints_RAM_Optimized's own
    // (generation-time) volume log does not run.
    void LogTotalIceVolume();

    // Determines the simulation domain (ModeledRegionOffsetX/Y, GridXTotal/
    // GridYTotal) from FLAG_WATER and FLAG_OPEN_BOUNDARY alone. Must not
    // read FLAG_ICE or any other flag -- see DetermineIceRegion() for the
    // separate, later, ice-aware bounding box.
    void DetermineExtents();

    // Determines IceRegionOffsetX/Y/Width/Height (the point-generation scan
    // area) from FLAG_ICE + FLAG_WATER + FLAG_OPEN_BOUNDARY. Deliberately
    // separate from DetermineExtents() (called after it, only when points
    // are actually about to be generated) so the simulation domain itself
    // can never depend on the ice mask, even though m_flags already holds
    // FLAG_ICE bits by the time either function runs (all masks are loaded
    // into the same shared array up front, in PrepareGridAndPoints).
    void DetermineIceRegion();

    // Poisson point generation helpers
    std::string prepare_cache_filename(int gx, int gy, int ppc);
    bool attempt_to_fill_from_cache(int gx, int gy, int ppc, std::vector<std::array<float, 2>> &buffer);
    void generate_and_save_poisson(int gx, int gy, float points_per_cell, std::vector<std::array<float, 2>> &buffer);

    // Image loading helper
    static bool LoadPng(const std::string& filename, int& w, int& h, int& channels, std::vector<uint8_t>& data);
};

#endif // DATA_PREPARER_H
