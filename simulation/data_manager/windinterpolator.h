#ifndef WINDINTERPOLATOR_H
#define WINDINTERPOLATOR_H

#include <string>
#include <vector>
#include <utility>
#include <memory>
#include <functional>
#include <future>
#include <mutex>
#include <cstdint>
#include "parameters_sim.h"
#include "frame_ring_buffer.h"
#include "directory_manager.h"

namespace netCDF { class NcFile; }
namespace H5 { class H5File; }

class WindInterpolator
{
public:
    WindInterpolator(SimParams& params, DirectoryManager& dm);
    ~WindInterpolator();

    SimParams &prms;
    DirectoryManager &dirs;

    // Adds a CARRA1 file to the (possibly multi-file) time series -- add
    // more than once, in chronological order, to cover multiple month-
    // partitioned .nc files. All files must share the same lat/lon grid
    // (validated against the first file's, which sets the CARRA1<->sim-grid mapping).
    void SetCARRA1Path(const std::string& filePath);

    // Set current time; return true if wind frames changed
    bool SetTime(double t);

    // Get interpolated wind velocity at grid cell (i,j)
    // Assumes SetTime(t) has been called beforehand
    // Returns (vx, vy) pair using current_wind_alpha for interpolation
    std::pair<double, double> GetWindValue(int i, int j) const;

    // Same as GetWindValue, but rotated back to true geographic
    // (eastward, northward) components instead of the grid's local
    // (projected-plane) x/y frame that GetWindValue itself returns (see
    // RenderFrameFields's TransformWind_GeographicToGrid step). The grid's
    // local axes only coincide with true east/north near the projection's
    // tangent point and rotate away from it elsewhere, so anything that
    // needs a real compass bearing (e.g. a wind rose) should call this
    // instead of GetWindValue -- using GetWindValue's raw components as if
    // they were geographic would give a position-dependent directional bias
    // across a wide domain.
    std::pair<double, double> GetWindValueGeographic(int i, int j) const;

    // Get interpolated surface (skin) temperature in Kelvin at grid cell
    // (i,j). Assumes SetTime(t) has been called beforehand. Scalar, so no
    // rotation is needed (unlike wind) -- just bilinear interpolation +
    // temporal blending via current_wind_alpha.
    double GetTemperatureValue(int i, int j) const;

    // Get Data Pointer for GPU transfer logic
    // logicalFrame: 0 for the "first" frame (t), 1 for the "second" frame (t+dt)
    // component: 0 for X, 1 for Y
    const float* GetWindDataPointer(int logicalFrame, int component) const;

    // Get Data Pointer for GPU transfer logic (temperature; single component)
    const float* GetTempDataPointer(int logicalFrame) const;

    // Get the mapped (lat_idx, lon_idx) for a grid cell
    std::pair<int, int> GetMappedIndices(int i, int j) const;

    // Renders an exact (unblended) temperature frame for a global CARRA1
    // frame index -- no SetTime() needed, bypasses the ring buffer. Checks
    // the HDF5 temperature cache first. For callers that walk frames
    // directly rather than interpolating between them (e.g. thermal spin-up).
    void GetTemperatureFrameExact(int globalFrameIdx, std::vector<float>& out);

    // Combined temperature+wind render, bypassing both HDF5 caches. Used by
    // preparer's "Precompute Temperature+Wind Cache" action: does the shared
    // per-cell grid<->CARRA1 bilinear-quad solve once instead of twice, and
    // reads u10/v10/skt for a given frame back-to-back in a single netCDF
    // pass instead of two separate full end-to-end passes over the file --
    // see RenderFrameFields.
    void RenderTempAndWindFrameUncached(int globalFrameIdx, std::vector<float>& out_vx,
                                         std::vector<float>& out_vy, std::vector<float>& out_temp);

    // Finds the CARRA1 cell nearest (lat_deg, lon_deg) and writes a CSV time
    // series of its raw 'skt' surface temperature (Celsius), one row per
    // loaded frame. Reads directly from the source netCDF file(s), bypassing
    // the sim grid and HDF5 cache -- works even before PrepareGridAndPoints.
    void ExportTemperatureTimeSeries(double lat_deg, double lon_deg, const std::string& csvPath);

    // Resolves [t_start, t_end] (Unix epoch seconds) to the [firstIdx, lastIdx]
    // CARRA1 frame range. Throws unless both endpoints exactly match real frame timestamps.
    std::pair<int, int> GetFrameRangeForTimeSpan(long long t_start, long long t_end) const;

    int GetCARRA1NumFrames() const { return carra1_num_frames; }
    long long GetCARRA1Timestamp(int globalFrameIdx) const;

    // Whether a complete, usable HDF5 cache exists for temperature/wind
    // (opens it on first call; a stale/mismatched cache throws here too,
    // rather than reporting "unavailable"). For callers deciding whether
    // per-frame reads are cheap enough for live UI updates.
    bool HasTemperatureCache();
    bool HasWindCache();

    // Forces the next temperature-cache lookup to re-check disk -- needed
    // after this session writes/regenerates the cache, so a prior "no file" result isn't reused.
    void InvalidateTemperatureCache();

    // Same as InvalidateTemperatureCache, but for the wind (vx/vy) cache --
    // needed after this session writes/regenerates it.
    void InvalidateWindCache();

    // Wind buffers
    std::vector<float> wind_vx_frame_buffer[3];
    std::vector<float> wind_vy_frame_buffer[3];

    // Surface temperature buffer (Kelvin, scalar -- 3 ring-buffer slots,
    // same slot indices as wind_vx/vy_frame_buffer since they're loaded
    // together per frame)
    std::vector<float> wind_temp_frame_buffer[3];

    // Interpolation parameter for temporal interpolation between frame buffers
    double current_wind_alpha = 0.0;

    // Optimization: Optional enabling
    bool interpolation_enabled = true;
    void SetEnabled(bool enabled);

    // Synthetic-forcing test mode: for finding the wind velocity at which
    // ice breakup begins. Deliberately NOT SimParams/JSON-configurable --
    // plain in-class settings, edited directly in source (or via a setter,
    // if a runtime toggle is ever needed). Unlike CurrentInterpolator, this
    // does NOT bypass the real CARRA1 pipeline -- ProcessCARRA1 still runs
    // every call, so surface temperature (wind_temp_frame_buffer) stays
    // real, since it feeds the live ice-strength model
    // (partition_kernel_compute_ice_strength via current_wind_alpha, see
    // Model::Step). Only wind_vx/vy_frame_buffer's active slots are
    // overwritten afterward -- see SetTime()/ApplyWindTestMode() in
    // windinterpolator.cpp.
    //
    // magnitude(t) = test_initial_magnitude
    //              + (test_growth_rate_per_day/86400) * max(0, t - prms.SimulationStartTime)
    // so test_growth_rate_per_day==0 gives a constant (or, with
    // test_initial_magnitude==0 too, zero) wind; a nonzero rate gives linear
    // growth from test_initial_magnitude. Direction is fixed (see
    // ComputeDefaultTestDirection()): the bearing from (80.738, -66.900) to
    // (78.654, -73.357) -- Nares Strait's long axis, pointing south
    // (northerly flow) -- rotated into grid-local coordinates.
    //
    // Default: enabled, starts at 0, grows 20 m/s per day (per your
    // immediate ask).
    bool test_mode_enabled = false;
    double test_initial_magnitude = 20.0;     // m/s, at t == SimulationStartTime
    double test_growth_rate_per_day = 2.0;  // m/s per day

    // Disables background preloading of the "next" frame (default:
    // enabled). Intended for trackbar-driven callers (preparer, visualizer)
    // that jump around non-sequentially, where preloading is usually wasted
    // work -- see FrameRingBuffer::SetPreloadEnabled.
    void SetPreloadEnabled(bool enabled) { wind_ring.SetPreloadEnabled(enabled); }

    // Optional hook invoked once per frame after its netCDF slice is loaded
    // (the expensive part of SetTime() -- typically a couple of seconds per
    // frame, dominated by file I/O). Lets a GUI caller pump its event loop
    // (e.g. QCoreApplication::processEvents) between frame loads so the OS
    // doesn't flag the app as unresponsive during a slow SetTime() call.
    // No-op by default; deliberately framework-agnostic -- this class is
    // also linked into headless CLI/simulation targets with no Qt present.
    std::function<void()> on_loading_progress;

private:

    // One opened CARRA1 file plus its own local time axis. carra1_times
    // (below) is the concatenation of every file's `times`, in the order
    // files were added; global_offset is that file's position within the
    // concatenated/global frame index space.
    struct CARRA1File {
        std::string path;
        std::unique_ptr<netCDF::NcFile> file;
        std::vector<long long> times;
        int num_frames = 0;
        int global_offset = 0;
    };
    std::vector<CARRA1File> carra1_files;

    // Metadata from CARRA1 (grid: shared across all files, from the first one)
    std::vector<double> carra1_lats;      // 2D latitude array
    std::vector<double> carra1_lons;      // 2D longitude array
    std::vector<long long> carra1_times;  // Linux timestamps, concatenated across all files
    int carra1_nx = 0;
    int carra1_ny = 0;
    long long carra1_start_time = 0;
    int carra1_num_frames = 0;            // total frames across all files
    std::vector<float> raw_u;    // Preallocated buffer for raw u wind frame
    std::vector<float> raw_v;    // Preallocated buffer for raw v wind frame
    std::vector<float> raw_skt;  // Preallocated buffer for raw surface temperature frame

    // Ring-buffered frame cache: 2 active slots + at most 1 async preload
    // in flight. See FrameRingBuffer for the synchronization invariants.
    FrameRingBuffer wind_ring;
    static constexpr int NUM_SLOTS = FrameRingBuffer::NUM_SLOTS;

    // Synthetic-forcing test mode -- see test_mode_enabled above.
    void ApplyWindTestMode(double t);
    Eigen::Vector2d ComputeDefaultTestDirection() const;
    double test_dir_x = 0.0, test_dir_y = 0.0;
    bool test_dir_initialized = false;

    // Helper methods
    void LoadCARRA1FileMetadata(CARRA1File& f);
    void PrecomputeGridMapping();
    void LoadWindFrame(int frameIdx, int bufferSlot);
    bool ProcessCARRA1(double t);

    // Renders one CARRA1 frame's wind/temperature fields onto the sim grid.
    // computeWind/computeTemp select which fields are rendered (unrequested
    // out_* may be nullptr). skipCache disables the temperature-cache
    // lookup -- used only while generating that same cache.
    void RenderFrameFields(int frameIdx, bool computeWind, bool computeTemp,
                            float* out_vx, float* out_vy, float* out_temp,
                            bool skipCache = false);

    // HDF5 temperature-frame cache, keyed by grid size, generated by
    // preparer's "Precompute Temperature Cache" tool (always all frames in
    // one pass) -- so it's either complete or absent; a mismatched frame
    // count/grid size is stale/broken and throws rather than partial-hitting.
public:
    std::string TemperatureCacheFilename() const;
private:
    void EnsureTemperatureCacheOpen();
    bool TryGetCachedTemperatureFrame(int frameIdx, float* out_temp);

    std::unique_ptr<H5::H5File> temp_cache_file;
    int temp_cache_num_frames = 0;
    bool temp_cache_open_attempted = false;
    std::mutex temp_cache_mutex;

    // HDF5 wind-frame cache: a pre-rendered vx/vy plane per CARRA1 frame,
    // keyed by grid size (see WindCacheFilename), generated by preparer's
    // "Precompute Wind Cache" tool. Structurally identical to the
    // temperature cache above (separate file/datasets, same lookup pattern).
public:
    std::string WindCacheFilename() const;
private:
    void EnsureWindCacheOpen();
    bool TryGetCachedWindFrame(int frameIdx, float* out_vx, float* out_vy);

    std::unique_ptr<H5::H5File> wind_cache_file;
    int wind_cache_num_frames = 0;
    bool wind_cache_open_attempted = false;
    std::mutex wind_cache_mutex;

    // Resolve a global (concatenated-timeline) frame index to which file it
    // lives in and that file's own local frame index. Returns (-1, -1) if
    // out of range.
    std::pair<int, int> ResolveGlobalFrame(int globalIdx) const;

    // Mapped indices
    std::vector<int> mapped_lat_idx;
    std::vector<int> mapped_lon_idx;

    // Thin wrapper over prms.proj.InverseProjectPixel (see projection.h) that
    // additionally applies this class's grid sub-region offset/Y-flip.
    std::pair<double, double> GridInverseProjectPixel(double lat_deg, double lon_deg) const;
    Eigen::Matrix2d ComputeRotationMatrix(double lat_rad, double lon_rad) const;
    Eigen::Vector2d TransformWind_LocalToGeographic(double u_local, double v_local, const Eigen::Vector2d& p00, const Eigen::Vector2d& p01, const Eigen::Vector2d& p10) const;
    Eigen::Vector2d TransformWind_GeographicToGrid(const Eigen::Vector2d& geo_wind, double lat_rad, double lon_rad) const;
};

#endif // WINDINTERPOLATOR_H
