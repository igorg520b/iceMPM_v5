#ifndef CURRENTINTERPOLATOR_H
#define CURRENTINTERPOLATOR_H

#include <string_view>
#include <string>
#include <utility>
#include <vector>
#include <Eigen/Core>
#include <memory>
#include <functional>
#include <future>
#include <mutex>
#include <cstdint>

#include "parameters_sim.h"
#include "frame_ring_buffer.h"
#include "directory_manager.h"

// Forward declaration to avoid including heavy/conflicting HDF5 headers in .h
namespace H5 { class H5File; }


class CurrentInterpolator
{
public:
    CurrentInterpolator(SimParams& params, DirectoryManager& dm);
    ~CurrentInterpolator();

    SimParams &prms;
    DirectoryManager &dirs;

    // Set GLO12 NetCDF file path (ocean current, uo/vo)
    void SetGLO12Path(const std::string& filePath);
    // Set GLO12 Tidal Currents file path (utide/vtide, added to uo/vo)
    void SetGLO12TidesPath(const std::string& filePath);
    // Set GLO12 Ice Thickness file path (sithick) -- a separate file from
    // the currents file in the real Kane Basin deployment (same lat/lon/
    // time grid, different variable), used by both RenderThicknessAtTime
    // and the live GetThicknessValue() visualization.
    void SetGLO12ThicknessPath(const std::string& filePath);

    // Set current time; return ocean_changed
    bool SetTime(double t);

    // Get interpolated velocity at grid cell (i,j)
    // Assumes SetTime(t) has been called beforehand to set up frame buffers
    // Returns (vx, vy) pair using current_alpha for interpolation
    std::pair<double, double> GetOceanValue(int i, int j) const;

    // Same as GetOceanValue, but rotated back to true geographic
    // (eastward, northward) components instead of the grid's local
    // (projected-plane) x/y frame that GetOceanValue itself returns (see
    // RenderGLO12FrameFields's rot.ex/ey/nx/ny step). Mirrors
    // WindInterpolator::GetWindValueGeographic -- see its header comment
    // for the full rationale. Returns {NaN, NaN} if this cell falls
    // outside the projection's valid domain (distinguishable from a
    // legitimate {0,0} = genuinely still water).
    std::pair<double, double> GetOceanValueGeographic(int i, int j) const;

    // Get Data Pointer for GPU transfer logic
    // logicalFrame: 0 for the "first" frame (t), 1 for the "second" frame (t+dt)
    // component: 0 for X, 1 for Y
    const float* GetOceanDataPointer(int logicalFrame, int component) const;

    // Get Latitude and Longitude at grid cell (i,j)
    // Returns (lat, lon) pair in degrees
    std::pair<double, double> GetLatLon(int i, int j) const;

    // Render GLO12 sea-ice thickness ("sithick", meters) onto a width x
    // height IMAGE-space buffer (the full source image, before any grid
    // cropping/offset -- NOT prms.GridXTotal/GridYTotal), using the single
    // GLO12 frame nearest absolute epoch time t. This is a one-shot prepare-
    // time render (an initial condition), not a continuously-interpolated
    // forcing like wind/current, so it doesn't use the frame ring buffer.
    // Requires SetGLO12Path() to have been called first.
    //
    // A pixel is only written with a real value and out_valid[idx]=1 if all
    // 4 of its bilinear-interpolation source corners are non-fill GLO12
    // cells; otherwise out_valid[idx]=0 and out_thickness[idx] is left at
    // 0. Callers must NOT treat 0/invalid as "zero ice" -- a GLO12 fill
    // value (land, missing day) is not the same as open water -- and should
    // instead nearest-fill + blur invalid cells from valid neighbors.
    void RenderThicknessAtTime(double t, int width, int height,
                                std::vector<float>& out_thickness,
                                std::vector<uint8_t>& out_valid);

    // Live, continuously-interpolated GLO12 ice thickness at grid cell
    // (i,j) -- for visualization/verification via the flow-time slider, NOT
    // the one-shot prepare-time initial condition (see RenderThicknessAtTime
    // above). Assumes SetTime(t) has been called beforehand. Returns 0 when
    // GLO12 isn't configured, the cell has no GLO12 coverage (same zero-fill
    // convention as GetOceanValue), or -- unlike GetOceanValue -- when t
    // falls entirely outside the GLO12 file's time coverage (no nearest-
    // frame clamping/extrapolation, by design: a value from a different day
    // shouldn't masquerade as "the" thickness for an uncovered date).
    double GetThicknessValue(int i, int j) const;

    // Total number of frames on the primary GLO12 file's own time axis
    // (glo12_times) -- the frame-index space RenderCurrentFrameUncached and
    // GetGLO12Timestamp operate on. 0 if SetGLO12Path was never called.
    int GetGLO12NumFrames() const { return glo12_num_frames; }
    long long GetGLO12Timestamp(int frameIdx) const;

    // Renders one GLO12 frame's ocean current (vx/vy) and ice-thickness
    // fields directly from the source HDF5 files, bypassing the current
    // cache even if a matching one exists. Used only by cache-generation
    // tooling (preparer's "Precompute Current Cache" action), which must not
    // read from the cache it's in the process of building.
    void RenderCurrentFrameUncached(int frameIdx, std::vector<float>& out_vx,
                                     std::vector<float>& out_vy, std::vector<float>& out_thickness);

    // Forces the next current-cache lookup to re-check for (and, if present,
    // re-open) the cache file on disk -- needed after this same session
    // writes/regenerates it (see WindInterpolator::InvalidateTemperatureCache
    // for the same pattern).
    void InvalidateCurrentCache();

    // Whether a complete, currently-usable HDF5 cache exists for GLO12
    // current+thickness data (see WindInterpolator::HasTemperatureCache for
    // the same open-on-first-call / throw-on-mismatch contract).
    bool HasCurrentCache();

    // GPU-accessible buffers (3 frames in RAM for ring buffering)
    std::vector<float> ocean_vx_frame_buffer[3];
    std::vector<float> ocean_vy_frame_buffer[3];
    std::vector<float> ocean_thickness_frame_buffer[3];

    // Interpolation parameter for temporal interpolation between frame buffers
    // Range: [0.0, 1.0] where 0.0 = at first frame, 1.0 = at second frame
    // Computed during SetTime() to indicate position between current_first_idx and current_second_idx
    double current_ocean_alpha = 0.0;

    // Optimization: Optional enabling
    bool interpolation_enabled = true;
    void SetEnabled(bool enabled);

    // Synthetic-forcing test mode: for finding the current velocity at which
    // ice breakup begins. Deliberately NOT SimParams/JSON-configurable --
    // plain in-class settings, edited directly in source (or via a setter,
    // if a runtime toggle is ever needed). When enabled, ProcessGLO12 (and
    // therefore all real GLO12 current/thickness data) is bypassed entirely
    // -- see SetTime()/ProcessTestMode() in currentinterpolator.cpp.
    //
    // magnitude(t) = test_initial_magnitude
    //              + (test_growth_rate_per_day/86400) * max(0, t - prms.SimulationStartTime)
    // so test_growth_rate_per_day==0 gives a constant (or, with
    // test_initial_magnitude==0 too, zero) current; a nonzero rate gives
    // linear growth from test_initial_magnitude. Direction is fixed (see
    // ComputeDefaultTestDirection()): the bearing from (80.738, -66.900) to
    // (78.654, -73.357) -- Nares Strait's long axis, pointing south
    // (northerly flow) -- rotated into grid-local coordinates.
    // Default: enabled, zero current ("no current" -- per your immediate
    // ask). Change these three to run a constant or linearly-growing
    // synthetic current instead.
    bool test_mode_enabled = false;
    double test_initial_magnitude = 0.0;     // m/s, at t == SimulationStartTime
    double test_growth_rate_per_day = 0.0;   // m/s per day

    // Substantial spatial blur applied to the combined mean+tide GLO12
    // current (raw_u/raw_v, native GLO12 resolution) before it's resampled
    // onto the sim grid. GLO12 wasn't designed for local ice-fracture
    // forcing -- its native node-to-node gradient is sharp enough to break
    // ice locally in a way that's an artifact of the data, not the physics.
    // Blurring keeps the large-scale drift while smoothing that local
    // gradient out. Deliberately NOT SimParams/JSON-configurable -- plain
    // in-class settings, edited directly in source. Baked into the HDF5
    // current cache if enabled at "Precompute Current Cache" time --
    // regenerate the cache after changing either value to pick up the new
    // blur. Only applies to the real GLO12 data path (RenderGLO12FrameFields)
    // -- irrelevant to test_mode_enabled's spatially-uniform synthetic current.
    bool enable_current_blur = true;
    int current_blur_radius = 4;   // GLO12 grid cells (native resolution, not sim-grid cells)

    // Disables background preloading of the "next" frame (default:
    // enabled). Intended for trackbar-driven callers (preparer, visualizer)
    // that jump around non-sequentially, where preloading is usually wasted
    // work -- see FrameRingBuffer::SetPreloadEnabled.
    void SetPreloadEnabled(bool enabled) { ocean_ring.SetPreloadEnabled(enabled); }

    // Optional hook invoked once per frame after its HDF5 slice is loaded
    // (the expensive part of SetTime() -- typically a couple of seconds per
    // frame, dominated by file I/O). Lets a GUI caller pump its event loop
    // (e.g. QCoreApplication::processEvents) between frame loads so the OS
    // doesn't flag the app as unresponsive during a slow SetTime() call.
    // No-op by default; deliberately framework-agnostic -- this class is
    // also linked into headless CLI/simulation targets with no Qt present.
    // NOTE: LoadGLO12Frame() can run on a background thread (async preload
    // of the next expected frame) as well as the calling thread, so this
    // hook may fire off the GUI thread -- callers that pump a Qt event loop
    // must guard for that themselves (processEvents() is main-thread-only).
    std::function<void()> on_loading_progress;

private:
  
    std::string glo12_path;
    std::unique_ptr<H5::H5File> file_glo12;

    std::string glo12_tides_path;
    std::unique_ptr<H5::H5File> file_glo12_tides;

    std::string glo12_thickness_path;
    std::unique_ptr<H5::H5File> file_glo12_thickness;

    // Metadata from HDF5 (Flow)
    double time_interval = 0.0;         // time between frames
    int num_frames = 0;                 // total number of frames
    int loop_mode = 0;                  // 0 = periodic, 1 = hold last frame
    
    // Metadata from GLO12
    std::vector<double> glo12_lats;      // 1D latitude array (descending usually, but we check)
    std::vector<double> glo12_lons;      // 1D longitude array
    std::vector<long long> glo12_times;  // Linux timestamps (or hours since epoch converted)
    long long glo12_start_time = 0;
    int glo12_num_frames = 0;
    std::vector<float> raw_u;    // Preallocated buffer for raw u ocean frame
    std::vector<float> raw_v;    // Preallocated buffer for raw v ocean frame
    std::vector<float> raw_thick; // Preallocated buffer for raw sithick frame (ring-buffered path)

    // sithick's own time axis, loaded from file_glo12_thickness by
    // SetGLO12ThicknessPath() -- the ice-thickness product is typically a
    // much coarser cadence (e.g. daily) than the primary current file
    // (e.g. hourly) and covers a different date range, so it cannot share
    // glo12_times/frame indices with uo/vo. Empty when no separate
    // thickness file was set (sithick, if present at all, is then read
    // from the primary file using glo12_times directly).
    std::vector<long long> glo12_thickness_times;

    // Set at the top of ProcessGLO12(), before any nearest-frame clamping:
    // whether the last SetTime(t) fell within the effective thickness time
    // axis's coverage (glo12_thickness_times if a separate thickness file
    // was set, else glo12_times). GetThicknessValue() uses this to return 0
    // for out-of-coverage dates instead of silently showing a clamped/stale
    // frame (GetOceanValue intentionally keeps the old clamp-to-nearest
    // behavior -- this is scoped to the thickness visualization only).
    bool glo12_thickness_time_covered = true;

    // Reads and converts a GLO12 file's "time"/"time_counter" variable
    // (hours since 1950-01-01) to Unix epoch seconds. Shared by the primary
    // GLO12 file (LoadGLO12Metadata) and the ice-thickness file
    // (SetGLO12ThicknessPath), which can have a different/coarser time axis.
    std::vector<long long> LoadGLO12TimeAxis(H5::H5File* file) const;

    // Ring-buffered frame cache: 2 active slots + at most 1 async preload
    // in flight. See FrameRingBuffer for the synchronization invariants.
    FrameRingBuffer ocean_ring;
    static constexpr int NUM_SLOTS = FrameRingBuffer::NUM_SLOTS;

    // Flow descriptor (read from HDF5 "/" group)
    std::string flow_type_id = "";

    // Helper methods
    bool ProcessGLO12(double t);
    void LoadGLO12Metadata();
    void LoadGLO12Frame(int frameIdx, int bufferSlot);

    // Synthetic-forcing test mode -- see test_mode_enabled above.
    bool ProcessTestMode(double t);
    Eigen::Vector2d ComputeDefaultTestDirection() const;
    double test_dir_x = 0.0, test_dir_y = 0.0;
    bool test_dir_initialized = false;

    // Separable box blur (horizontal pass, then vertical pass), boundary-
    // clamped -- see enable_current_blur above. radius<=0 is a no-op.
    static void BoxBlur(int width, int height, std::vector<float>& field, int radius);

    // Renders one GLO12 frame's vx/vy/thickness fields into caller-owned
    // buffers -- the actual body shared by LoadGLO12Frame (ring-buffer slot,
    // skipCache=false) and RenderCurrentFrameUncached (cache-generation
    // tooling, skipCache=true). Checks the HDF5 current cache (see
    // CurrentCacheFilename) before touching the source HDF5 files, unless
    // skipCache is set.
    void RenderGLO12FrameFields(int frameIdx, float* out_vx, float* out_vy, float* out_thick, bool skipCache);

    // HDF5 current-frame cache: a pre-rendered vx/vy/thickness plane per
    // GLO12 frame, keyed by grid size (see CurrentCacheFilename), generated
    // by preparer's "Precompute Current Cache" tool, which always renders
    // every currently-loaded GLO12 frame in a single pass -- so a cache is
    // either complete (frame count matches glo12_num_frames exactly, so a
    // GLO12 frame index doubles as the cache's own frame index) or absent;
    // see WindInterpolator's temperature cache comment for the full
    // silent-miss-vs-throw policy this follows.
public:
    std::string CurrentCacheFilename() const;
private:
    void EnsureCurrentCacheOpen();
    bool TryGetCachedCurrentFrame(int frameIdx, float* out_vx, float* out_vy, float* out_thick);

    std::unique_ptr<H5::H5File> current_cache_file;
    int current_cache_num_frames = 0;
    bool current_cache_open_attempted = false;
    std::mutex current_cache_mutex;

    // Projection math (ProjectPixel/ComputeRotation) now lives in the
    // single shared Projection class -- see prms.proj (parameters_sim.h /
    // projection.h). Use prms.proj.ProjectPixel(...)/ComputeRotation(...)
    // directly instead of a private copy here.
};

#endif // CURRENTINTERPOLATOR_H
