#include "currentinterpolator.h"
#include <H5Cpp.h>
#include <netcdf>
#include <spdlog/spdlog.h>
#include <cmath>
#include <filesystem>
#include <fmt/format.h>
#include <algorithm> // for upper_bound
#include <utility>
#include <limits>
#include <omp.h>

CurrentInterpolator::CurrentInterpolator(SimParams& params, DirectoryManager& dm) : prms(params), dirs(dm)
{
    num_frames = 0;
//    gx = 0;
//    gy = 0;
    time_interval = 0.0;
    loop_mode = 0;

    // Check HDF5 Version
    unsigned maj, min, rel;
    H5get_libversion(&maj, &min, &rel);
    LOGR("HDF5 Version [Compile-Time]: {}", H5_VERS_INFO);
    LOGR("HDF5 Version [Runtime]:      {}.{}.{}" , maj, min, rel);
}

CurrentInterpolator::~CurrentInterpolator() = default;




void CurrentInterpolator::SetGLO12Path(const std::string& filePath)
{
    if (!glo12_path.empty()) {
        throw std::runtime_error(fmt::format(
            "SetGLO12Path called more than once (already set to '{}', now called with '{}')",
            glo12_path, filePath));
    }

    // Check file exists before attempting to open
    if (!std::filesystem::exists(filePath)) {
        throw std::runtime_error(fmt::format("GLO12 file not found: {}", filePath));
    }
    
    glo12_path = filePath;
    LoadGLO12Metadata();
}

void CurrentInterpolator::SetGLO12TidesPath(const std::string& filePath)
{
    if (!glo12_tides_path.empty()) {
        throw std::runtime_error(fmt::format(
            "SetGLO12TidesPath called more than once (already set to '{}', now called with '{}')",
            glo12_tides_path, filePath));
    }

    // Check file exists before attempting to open
    if (!std::filesystem::exists(filePath)) {
        throw std::runtime_error(fmt::format("GLO12 Tides file not found: {}", filePath));
    }
    
    glo12_tides_path = filePath;
    LOGR("Loading GLO12 Tides Metadata from {}", glo12_tides_path);
    file_glo12_tides = std::make_unique<H5::H5File>(glo12_tides_path, H5F_ACC_RDONLY);

    // We assume the grid and time structure matches existing GLO12 file for simplicity.
    // If we wanted to be robust we would verify dimensions here.
}

void CurrentInterpolator::SetGLO12ThicknessPath(const std::string& filePath)
{
    if (!glo12_thickness_path.empty()) {
        throw std::runtime_error(fmt::format(
            "SetGLO12ThicknessPath called more than once (already set to '{}', now called with '{}')",
            glo12_thickness_path, filePath));
    }

    // Check file exists before attempting to open
    if (!std::filesystem::exists(filePath)) {
        throw std::runtime_error(fmt::format("GLO12 Thickness file not found: {}", filePath));
    }

    glo12_thickness_path = filePath;
    LOGR("Loading GLO12 Thickness file from {}", glo12_thickness_path);
    file_glo12_thickness = std::make_unique<H5::H5File>(glo12_thickness_path, H5F_ACC_RDONLY);

    // We assume the LAT/LON GRID matches the primary GLO12 (currents) file
    // for simplicity -- SetGLO12Path() must be called (its LoadGLO12Metadata()
    // populates glo12_lats/lons, which this file's bilinear interpolation
    // reuses). The TIME axis, however, is loaded independently: the ice-
    // thickness product is a much coarser cadence than currents (e.g. daily
    // vs. hourly) and covers a different date range, so reusing the primary
    // file's frame indices would read out of bounds.
    try {
        glo12_thickness_times = LoadGLO12TimeAxis(file_glo12_thickness.get());
        LOGR("GLO12 Thickness file has {} time frame(s), range [{} .. {}]",
             glo12_thickness_times.size(),
             glo12_thickness_times.empty() ? 0 : glo12_thickness_times.front(),
             glo12_thickness_times.empty() ? 0 : glo12_thickness_times.back());
    } catch (const H5::Exception& e) {
        LOGR("GLO12 Thickness Error loading time: {}", e.getDetailMsg());
        throw;
    }
}

std::vector<long long> CurrentInterpolator::LoadGLO12TimeAxis(H5::H5File* file) const
{
    std::string time_name = "time";
    H5::Group root = file->openGroup("/");
    if (!root.nameExists(time_name)) {
        if (root.nameExists("time_counter")) {
            time_name = "time_counter";
        } else {
            throw std::runtime_error("GLO12: Neither 'time' nor 'time_counter' found");
        }
    }

    H5::DataSet ds_time = file->openDataSet(time_name);
    H5::DataSpace space = ds_time.getSpace();
    hsize_t dims[1];
    space.getSimpleExtentDims(dims, NULL);
    int n = static_cast<int>(dims[0]);

    std::vector<double> buf(n);
    ds_time.read(buf.data(), H5::PredType::NATIVE_DOUBLE);

    // Explicitly handle "Hours since 1950-01-01 00:00:00"
    // 1950-01-01 to 1970-01-01 is 7305 days (including leap years 1952, 56, 60, 64, 68)
    // 7305 days * 24 * 3600 = 631,152,000 seconds
    const long long OFFSET_1950_TO_1970 = 631152000;
    std::vector<long long> times(n);
    for (int i = 0; i < n; ++i) {
        long long seconds_since_1950 = (long long)(buf[i] * 3600.0);
        times[i] = seconds_since_1950 - OFFSET_1950_TO_1970;
    }
    return times;
}
void CurrentInterpolator::LoadGLO12Metadata()
{
    if (glo12_path.empty()) return;

    LOGR("Loading GLO12 Metadata from {}", glo12_path);
    file_glo12 = std::make_unique<H5::H5File>(glo12_path, H5F_ACC_RDONLY);

    // Load Time (try 'time', then 'time_counter')
    try {
        glo12_times = LoadGLO12TimeAxis(file_glo12.get());
        glo12_num_frames = (int)glo12_times.size();
        glo12_start_time = glo12_times.empty() ? 0 : glo12_times[0];
        LOGR("GLO12 Start Time: {} (seconds-ish)", glo12_start_time);
    } catch (const H5::Exception& e) {
        LOGR("GLO12 Error loading time: {}", e.getDetailMsg());
        throw;
    }

    // Load Latitude/Longitude (Strict names)
    // We confirmed via check_nc that 'latitude' and 'longitude' exist (1D).
    auto load_coord_strict = [&](const std::string& name, std::vector<double>& out_vec, std::string label) {
        if (!file_glo12->nameExists(name)) {
             throw std::runtime_error(fmt::format("GLO12: Could not find {}", label));
        }
        H5::DataSet ds = file_glo12->openDataSet(name);
        H5::DataSpace space = ds.getSpace();
        hsize_t dims[1];
        space.getSimpleExtentDims(dims, NULL);
        out_vec.resize(dims[0]);
        ds.read(out_vec.data(), H5::PredType::NATIVE_DOUBLE);
        LOGR("GLO12 Loaded {} from '{}', size {}", label, name, dims[0]);
    };

    load_coord_strict("latitude", glo12_lats, "Latitude");
    load_coord_strict("longitude", glo12_lons, "Longitude");

    LOGR("GLO12 Grid: {}x{} (Lat/Lon)", glo12_lats.size(), glo12_lons.size());
}

void CurrentInterpolator::LoadGLO12Frame(int frameIdx, int bufferSlot)
{
    if (!file_glo12) return;

    const int& gx = prms.GridXTotal;
    const int& gy = prms.GridYTotal;
    size_t gridSize = (size_t)gx * (size_t)gy;
    ocean_vx_frame_buffer[bufferSlot].resize(gridSize);
    ocean_vy_frame_buffer[bufferSlot].resize(gridSize);
    ocean_thickness_frame_buffer[bufferSlot].resize(gridSize);

    RenderGLO12FrameFields(frameIdx,
                            ocean_vx_frame_buffer[bufferSlot].data(),
                            ocean_vy_frame_buffer[bufferSlot].data(),
                            ocean_thickness_frame_buffer[bufferSlot].data(),
                            /*skipCache=*/false);
}

void CurrentInterpolator::RenderCurrentFrameUncached(int frameIdx, std::vector<float>& out_vx,
                                                       std::vector<float>& out_vy, std::vector<float>& out_thickness)
{
    size_t gridSize = (size_t)prms.GridXTotal * (size_t)prms.GridYTotal;
    out_vx.resize(gridSize);
    out_vy.resize(gridSize);
    out_thickness.resize(gridSize);
    RenderGLO12FrameFields(frameIdx, out_vx.data(), out_vy.data(), out_thickness.data(), /*skipCache=*/true);
}

void CurrentInterpolator::RenderGLO12FrameFields(int frameIdx, float* out_vx, float* out_vy, float* out_thick, bool skipCache)
{
    if (!file_glo12) return;

    if (!skipCache) {
        if (TryGetCachedCurrentFrame(frameIdx, out_vx, out_vy, out_thick)) {
            if (on_loading_progress) on_loading_progress();
            return;
        }
    }

    // Reuse ERA5 interpolation logic structure:
    // Read uo, vo at frameIdx (slice).
    // Interpolate.

    int n_lat = glo12_lats.size();
    int n_lon = glo12_lons.size();
    size_t n_elem = n_lat * n_lon;

    raw_u.resize(n_elem);
    raw_v.resize(n_elem);
    raw_thick.resize(n_elem);

    // Explicitly read 'uo' and 'vo'
    // Explicitly read 'uo' and 'vo'
    auto read_strict = [&](H5::H5File* src, const std::string& name, std::vector<float>& buf, int useFrameIdx) {
        if (!src->nameExists(name)) {
             throw std::runtime_error(fmt::format("GLO12: Variable '{}' not found in file", name));
        }

        H5::DataSet ds = src->openDataSet(name);
        H5::DataSpace space = ds.getSpace();
        int ndims = space.getSimpleExtentNdims();
        std::vector<hsize_t> dims(ndims);
        space.getSimpleExtentDims(dims.data(), NULL);

        std::vector<hsize_t> start(ndims, 0);
        std::vector<hsize_t> count(ndims, 1);

        // Assuming time is 0-th dim, lat/lon last two
        start[0] = useFrameIdx;
        count[ndims-2] = n_lat;
        count[ndims-1] = n_lon;
        
        // Select hyperslab
        space.selectHyperslab(H5S_SELECT_SET, count.data(), start.data());
        
        hsize_t mem_dims[2] = {(hsize_t)n_lat, (hsize_t)n_lon};
        H5::DataSpace mem_space(2, mem_dims);
        
        ds.read(buf.data(), H5::PredType::NATIVE_FLOAT, mem_space, space);
        
        // Attributes for unpacking
        double fill_value = -32767.0; // Default
        bool has_fill = false;

        if (ds.attrExists("_FillValue")) {
            H5::Attribute att = ds.openAttribute("_FillValue");
            att.read(H5::PredType::NATIVE_DOUBLE, &fill_value);
            has_fill = true;
        }

        // Some GLO12 products (e.g. the GLORYS12V1 "my" reanalysis used for
        // pre-2022 dates) store variables as packed int16 with scale_factor/
        // add_offset (real = raw*scale_factor + add_offset), instead of
        // unpacked float32 like the ANFC product used for 2022-onward dates.
        // The ds.read() above only does HDF5's own numeric type conversion
        // (int16 -> float), which is a plain cast, not this CF-convention
        // unpacking -- so without this, packed files silently read as raw
        // integers (e.g. an ocean current of 350 instead of ~0.35 m/s).
        // Defaults to a no-op (1.0/0.0) for files that don't pack their data.
        double scale_factor = 1.0, add_offset = 0.0;
        if (ds.attrExists("scale_factor")) {
            H5::Attribute att = ds.openAttribute("scale_factor");
            att.read(H5::PredType::NATIVE_DOUBLE, &scale_factor);
        }
        if (ds.attrExists("add_offset")) {
            H5::Attribute att = ds.openAttribute("add_offset");
            att.read(H5::PredType::NATIVE_DOUBLE, &add_offset);
        }

        // Apply filtering (Fill Value check) and unpacking
        float min_val = 1e9, max_val = -1e9;
        int valid_count = 0;
        for(auto& v : buf) {
                // Check for fill value (or extremely large values typical of NetCDF fill)
                // -- compared against the raw (still-packed) value, same convention as _FillValue itself.
                bool is_fill = false;
                if (has_fill && std::abs(v - fill_value) < 1e-5) is_fill = true;
                if (std::abs(v) > 1e30) is_fill = true; // Safety check for 1e37

                if (is_fill) {
                    v = 0.0f;
                } else {
                    v = (float)(v * scale_factor + add_offset);
                    if(v < min_val) min_val = v;
                    if(v > max_val) max_val = v;
                    valid_count++;
                }
        }
        LOGR("GLO12: Read '{}' frame {}, valid pts: {}/{}, Range: [{}, {}]", name, useFrameIdx, valid_count, buf.size(), min_val, max_val);
    };

    // uo/vo (ocean current) come from the primary GLO12 file; sithick (ice
    // thickness) comes from its own file if SetGLO12ThicknessPath() was
    // called (the real Kane Basin deployment splits currents/thickness into
    // separate .nc files sharing the same lat/lon/time grid), falling back
    // to the primary file for anyone who has both variables in one file.
    // Whatever's missing defaults to 0 rather than throwing, so any single
    // file alone still works for what it does have.
    bool has_uv = file_glo12->nameExists("uo") && file_glo12->nameExists("vo");
    if (has_uv) {
        read_strict(file_glo12.get(), "uo", raw_u, frameIdx);
        read_strict(file_glo12.get(), "vo", raw_v, frameIdx);
    } else {
        std::fill(raw_u.begin(), raw_u.end(), 0.0f);
        std::fill(raw_v.begin(), raw_v.end(), 0.0f);
        LOGR("GLO12: 'uo'/'vo' not found in file -- ocean current will be 0 (file likely has only ice-thickness data)");
    }

    // sithick's own time axis (glo12_thickness_times, if a separate
    // thickness file was set) is independent of the primary file's -- the
    // ice-thickness product is a much coarser cadence (e.g. daily vs.
    // hourly) and covers a different date range, so frameIdx (an index into
    // glo12_times) cannot be reused directly. Find the nearest thickness
    // frame to this primary frame's own absolute time instead.
    H5::H5File* thickness_source = file_glo12_thickness ? file_glo12_thickness.get() : file_glo12.get();
    const std::vector<long long>& thickness_time_axis = glo12_thickness_times.empty() ? glo12_times : glo12_thickness_times;
    bool has_sithick = thickness_source->nameExists("sithick");
    if (has_sithick && !thickness_time_axis.empty()) {
        long long target_ts = glo12_times[frameIdx];
        auto tit = std::upper_bound(thickness_time_axis.begin(), thickness_time_axis.end(), target_ts);
        int tidx = (tit == thickness_time_axis.begin()) ? 0 : (int)std::distance(thickness_time_axis.begin(), tit) - 1;
        if (tidx < 0) tidx = 0;
        if (tidx >= (int)thickness_time_axis.size()) tidx = (int)thickness_time_axis.size() - 1;
        int nearestIdx = tidx;
        if (tidx + 1 < (int)thickness_time_axis.size()) {
            if (std::llabs(target_ts - thickness_time_axis[tidx + 1]) < std::llabs(target_ts - thickness_time_axis[tidx])) nearestIdx = tidx + 1;
        }
        read_strict(thickness_source, "sithick", raw_thick, nearestIdx);
    } else {
        std::fill(raw_thick.begin(), raw_thick.end(), 0.0f);
    }

    // --- INTEGRATE TIDES IF AVAILABLE ---
    if (prms.UseGLO12Tides && file_glo12_tides) {
         try {
             std::vector<float> raw_utide(n_elem), raw_vtide(n_elem);
             
             // Helper for reading from tidal file (similar to read_strict but using file_glo12_tides)
             auto read_tide = [&](const std::string& name, std::vector<float>& buf) {
                if (!file_glo12_tides->nameExists(name)) {
                     throw std::runtime_error(fmt::format("GLO12 Tides: Variable '{}' not found", name));
                }

                H5::DataSet ds = file_glo12_tides->openDataSet(name);
                H5::DataSpace space = ds.getSpace();
                int ndims = space.getSimpleExtentNdims();
                std::vector<hsize_t> dims(ndims);
                space.getSimpleExtentDims(dims.data(), NULL); // e.g. [time, lat, lon]

                std::vector<hsize_t> start(ndims, 0);
                std::vector<hsize_t> count(ndims, 1);
                
                // Use same frameIdx (assuming time axes are aligned/congruent)
                // If dimensions mismatch, HDF5 might throw or selection fails
                start[0] = frameIdx;
                count[ndims-2] = n_lat;
                count[ndims-1] = n_lon;
                
                space.selectHyperslab(H5S_SELECT_SET, count.data(), start.data());
                
                hsize_t mem_dims[2] = {(hsize_t)n_lat, (hsize_t)n_lon};
                H5::DataSpace mem_space(2, mem_dims);
                
                ds.read(buf.data(), H5::PredType::NATIVE_FLOAT, mem_space, space);
                
                // Handle Fill Values
                double fill_value = -32767.0;
                bool has_fill = false;
                if (ds.attrExists("_FillValue")) {
                    H5::Attribute att = ds.openAttribute("_FillValue");
                    att.read(H5::PredType::NATIVE_DOUBLE, &fill_value);
                    has_fill = true;
                }

                // See read_strict()'s identical comment: packed int16 GLO12
                // products need scale_factor/add_offset unpacking; defaults
                // to a no-op for unpacked-float files.
                double scale_factor = 1.0, add_offset = 0.0;
                if (ds.attrExists("scale_factor")) {
                    H5::Attribute att = ds.openAttribute("scale_factor");
                    att.read(H5::PredType::NATIVE_DOUBLE, &scale_factor);
                }
                if (ds.attrExists("add_offset")) {
                    H5::Attribute att = ds.openAttribute("add_offset");
                    att.read(H5::PredType::NATIVE_DOUBLE, &add_offset);
                }

                // Filter
                for(auto& v : buf) {
                    bool is_fill = false;
                    if (has_fill && std::abs(v - fill_value) < 1e-5) is_fill = true;
                    if (std::abs(v) > 1e30) is_fill = true;
                    if (is_fill) v = 0.0f;
                    else v = (float)(v * scale_factor + add_offset);
                }
                LOGR("GLO12 Tides: Read '{}' frame {}", name, frameIdx);
             };

             read_tide("utide", raw_utide);
             read_tide("vtide", raw_vtide);
             
             // Add Tides to Total Current
             #pragma omp parallel for
             for(size_t k=0; k<n_elem; ++k) {
                 raw_u[k] += raw_utide[k];
                 raw_v[k] += raw_vtide[k];
             }
             LOGR("GLO12: Added Tidal Currents to Frame {}", frameIdx);

         } catch (const std::exception& e) {
             LOGR("GLO12 Tides Error (Frame {}): {}", frameIdx, e.what());
             // Fallback? Assuming we want to proceed with non-tidal current if tides fail?
             // Or throw? User wants integration, so erroring is probably safer to alert them.
             throw;
         }
    }

    // Substantial blur to drop GLO12's local node-to-node gradient (an
    // artifact of data not intended for local ice-fracture forcing) while
    // keeping the large-scale drift. Box-blur is linear, so blurring the
    // already-combined mean+tide field here is equivalent to blurring uo/vo
    // and utide/vtide separately before summing them.
    if (enable_current_blur) {
        BoxBlur(n_lon, n_lat, raw_u, current_blur_radius);
        BoxBlur(n_lon, n_lat, raw_v, current_blur_radius);
    }

    // Interpolate to Grid
    const int& gx = prms.GridXTotal;
    const int& gy = prms.GridYTotal;
    size_t gridSize = (size_t)gx * (size_t)gy;

    bool lat_descending = n_lat > 1 && glo12_lats[0] > glo12_lats[1];
    
    // Log GLO12 Bounds
    if (n_lat > 0 && n_lon > 0) {
        LOGR("GLO12 Bounds: Lat[{} .. {}], Lon[{} .. {}]",
            glo12_lats.front(), glo12_lats.back(), glo12_lons.front(), glo12_lons.back());
    }

    std::atomic<int> valid_interp_count = 0;
    
    #pragma omp parallel for
    for (int j = 0; j < gy; ++j) {
        for (int i = 0; i < gx; ++i) {
            size_t grid_idx = (size_t)j + (size_t)i * (size_t)gy;
            
            int global_x = i + prms.ModeledRegionOffsetX;
            int global_y_grid = j + prms.ModeledRegionOffsetY;
            int global_y = prms.InitializationImageSizeY - 1 - global_y_grid;
            
            Projection::LatLon ll = prms.proj.ProjectPixel(global_x, global_y);
            if (!ll.valid) {
                 out_vx[grid_idx] = 0.0f;
                 out_vy[grid_idx] = 0.0f;
                 out_thick[grid_idx] = 0.0f;
                 continue;
            }
            
            // Calc r_idx, c_idx
            double r_idx = 0;
            double dlat = glo12_lats[1] - glo12_lats[0];
            r_idx = (ll.lat_deg - glo12_lats[0]) / dlat;

            double target_lon = ll.lon_deg;
            // GLO12 usually -180..180 or 0..360?
            // HDF5 output check didn't show values.
            // Safe bet: if grid is 0..360 and we are negative, add 360.
            bool is_0_360 = (glo12_lons.back() > 180.0);
            if (is_0_360 && target_lon < 0) target_lon += 360.0;
            
            double dlon = glo12_lons[1] - glo12_lons[0];
            double c_idx = (target_lon - glo12_lons[0]) / dlon;
            
            
            int r0 = (int)std::floor(r_idx);
            int c0 = (int)std::floor(c_idx);
            int r1 = r0 + 1;
            int c1 = c0 + 1;
            
            double dr = r_idx - r0;
            double dc = c_idx - c0;
            
            // Check bounds strictly?
            // If projected point is OUTSIDE GLO12 grid, we should probably set 0
            if (r0 < 0 || r0 >= n_lat - 1 || c0 < 0 || c0 >= n_lon -1) {
                // Out of bounds
                out_vx[grid_idx] = 0.0f;
                out_vy[grid_idx] = 0.0f;
                out_thick[grid_idx] = 0.0f;
                continue;
            }

            valid_interp_count++;

            // Use clamped indices from before just in case
            if (r0 < 0) r0 = 0; if (r1 >= n_lat) r1 = n_lat-1;
            if (r0 >= n_lat) r0 = n_lat-1; // clamp
            
            c0 = (c0 % n_lon + n_lon) % n_lon;
            c1 = (c1 % n_lon + n_lon) % n_lon;
            
            auto getVal = [&](const std::vector<float>& src) {
                float v00 = src[c0 + r0*n_lon];
                float v01 = src[c1 + r0*n_lon];
                float v10 = src[c0 + r1*n_lon];
                float v11 = src[c1 + r1*n_lon];
                double top = v00 * (1.0 - dc) + v01 * dc;
                double bot = v10 * (1.0 - dc) + v11 * dc;
                return top * (1.0 - dr) + bot * dr;
            };
            
            double u_val = getVal(raw_u);
            double v_val = getVal(raw_v);

            // Rotate
            Projection::RotMat rot = prms.proj.ComputeRotation(ll.lat_deg * (M_PI/180.0), ll.lon_deg * (M_PI/180.0));
            double vx_grid = u_val * rot.ex + v_val * rot.nx;
            double vy_grid = u_val * rot.ey + v_val * rot.ny;

            out_vx[grid_idx] = (float)vx_grid;
            out_vy[grid_idx] = (float)vy_grid;
            // Scalar, no rotation needed -- same bilinear (r/c/dr/dc) weights as u/v.
            out_thick[grid_idx] = has_sithick ? (float)getVal(raw_thick) : 0.0f;
        }
    }

    // Summary stats
    float min_v = 1e9, max_v = -1e9;
    int nonzero = 0;
    for (size_t k = 0; k < gridSize; ++k) {
        float f = out_vx[k];
        if(f != 0.0f) {
            nonzero++;
            if(f < min_v) min_v = f;
            if(f > max_v) max_v = f;
        }
    }
    LOGR("GLO12: Frame {} INTERPOLATED Grid Stats: NonZero {}/{}, Range [{}, {}]",
        frameIdx, nonzero, gridSize, (max_v < min_v ? 0.0f : min_v), (max_v < min_v ? 0.0f : max_v));
    spdlog::default_logger()->flush();

    if (on_loading_progress) on_loading_progress();
}

long long CurrentInterpolator::GetGLO12Timestamp(int frameIdx) const
{
    if (frameIdx < 0 || frameIdx >= (int)glo12_times.size()) {
        throw std::out_of_range(fmt::format(
            "GetGLO12Timestamp: frame index {} out of range (glo12_num_frames={})",
            frameIdx, glo12_num_frames));
    }
    return glo12_times[frameIdx];
}

// --- HDF5 current-frame cache -------------------------------------------
//
// Layout (written by preparer's "Precompute Current Cache" tool):
//   /vx, /vy, /thickness : float[num_frames][GridXTotal][GridYTotal], chunked (1,gx,gy)
//   /timestamps          : int64[num_frames], Unix epoch seconds, same frame order
//                           (kept for external inspection only -- not read back here)
// All extendible along the frame dimension; /vx carries GridXTotal/
// GridYTotal attributes, checked on open.
//
// Cache generation always renders every currently-loaded GLO12 frame in a
// single pass (0..glo12_num_frames-1, in order), so a cache is either
// complete -- in which case a GLO12 frame index is also a valid index into
// /vx, /vy, /thickness directly -- or it doesn't exist yet. Same silent-
// miss-vs-throw policy as WindInterpolator's temperature cache (see its
// comment): a mismatched grid size or frame count means the cache is
// stale/incomplete and throws instead of being used partially.

std::string CurrentInterpolator::CurrentCacheFilename() const
{
    return fmt::format("{}/glo12_current_cache/current_{}x{}.h5", dirs.DataCacheRoot(), prms.GridXTotal, prms.GridYTotal);
}

void CurrentInterpolator::EnsureCurrentCacheOpen()
{
    if (current_cache_open_attempted) return;
    current_cache_open_attempted = true;

    std::string path = CurrentCacheFilename();
    if (!std::filesystem::exists(path)) return; // no cache yet -- not an error

    auto file = std::make_unique<H5::H5File>(path, H5F_ACC_RDONLY);
    H5::DataSet vxDs = file->openDataSet("vx");

    int gx_attr = 0, gy_attr = 0;
    vxDs.openAttribute("GridXTotal").read(H5::PredType::NATIVE_INT, &gx_attr);
    vxDs.openAttribute("GridYTotal").read(H5::PredType::NATIVE_INT, &gy_attr);
    if (gx_attr != prms.GridXTotal || gy_attr != prms.GridYTotal) {
        throw std::runtime_error(fmt::format(
            "Current cache {} has grid attributes {}x{} that don't match its own filename ({}x{})",
            path, gx_attr, gy_attr, prms.GridXTotal, prms.GridYTotal));
    }

    hsize_t dims[3];
    vxDs.getSpace().getSimpleExtentDims(dims, nullptr);
    if ((int)dims[0] != glo12_num_frames) {
        throw std::runtime_error(fmt::format(
            "Current cache {} has {} frame(s) but {} GLO12 frame(s) are currently loaded -- "
            "regenerate it via Tools > Precompute Current Cache",
            path, dims[0], glo12_num_frames));
    }

    current_cache_num_frames = (int)dims[0];
    current_cache_file = std::move(file);
    LOGR("Current cache opened: {} ({} frames)", path, current_cache_num_frames);
}

void CurrentInterpolator::InvalidateCurrentCache()
{
    std::lock_guard<std::mutex> lock(current_cache_mutex);
    current_cache_file.reset();
    current_cache_num_frames = 0;
    current_cache_open_attempted = false;
}

bool CurrentInterpolator::HasCurrentCache()
{
    std::lock_guard<std::mutex> lock(current_cache_mutex);
    EnsureCurrentCacheOpen();
    return current_cache_file != nullptr;
}

bool CurrentInterpolator::TryGetCachedCurrentFrame(int frameIdx, float* out_vx, float* out_vy, float* out_thick)
{
    std::lock_guard<std::mutex> lock(current_cache_mutex);

    EnsureCurrentCacheOpen();
    if (!current_cache_file) return false;

    auto readComponent = [&](const char* name, float* out) {
        H5::DataSet ds = current_cache_file->openDataSet(name);
        hsize_t dims[3];
        ds.getSpace().getSimpleExtentDims(dims, nullptr);

        hsize_t offset[3] = {(hsize_t)frameIdx, 0, 0};
        hsize_t count[3]  = {1, dims[1], dims[2]};
        H5::DataSpace filespace = ds.getSpace();
        filespace.selectHyperslab(H5S_SELECT_SET, count, offset);

        hsize_t mem_dims[2] = {dims[1], dims[2]};
        H5::DataSpace memspace(2, mem_dims);

        ds.read(out, H5::PredType::NATIVE_FLOAT, memspace, filespace);
    };
    readComponent("vx", out_vx);
    readComponent("vy", out_vy);
    readComponent("thickness", out_thick);
    return true;
}

// Projection & rotation math now lives in the single shared Projection
// class (prms.proj) -- see projection.h/.cpp. Previously this file had its
// own private copy (byte-for-byte identical ProjectPixel to
// WindInterpolator's) implementing an orthographic projection; both have
// been replaced by prms.proj's stereographic implementation.







bool CurrentInterpolator::SetTime(double t)
{
    if (!interpolation_enabled) return false;
    if (test_mode_enabled) return ProcessTestMode(t);
    return ProcessGLO12(t);
}

Eigen::Vector2d CurrentInterpolator::ComputeDefaultTestDirection() const
{
    constexpr double kTestDirLat1 = 80.738, kTestDirLon1 = -66.900;
    constexpr double kTestDirLat2 = 78.654, kTestDirLon2 = -73.357;
    constexpr double deg2rad = M_PI / 180.0;

    double phi1 = kTestDirLat1 * deg2rad, lam1 = kTestDirLon1 * deg2rad;
    double phi2 = kTestDirLat2 * deg2rad, lam2 = kTestDirLon2 * deg2rad;
    double dlon = lam2 - lam1;
    // Standard initial-bearing formula (radians, clockwise from north).
    double bearing = std::atan2(std::sin(dlon) * std::cos(phi2),
                                 std::cos(phi1) * std::sin(phi2) - std::sin(phi1) * std::cos(phi2) * std::cos(dlon));
    double u = std::sin(bearing);   // geographic east component
    double v = std::cos(bearing);   // geographic north component
    Projection::RotMat rot = prms.proj.ComputeRotation(phi1, lam1);
    return Eigen::Vector2d(u * rot.ex + v * rot.nx, u * rot.ey + v * rot.ny);
}

bool CurrentInterpolator::ProcessTestMode(double t)
{
    const int& gx = prms.GridXTotal;
    const int& gy = prms.GridYTotal;
    size_t gridSize = (size_t)gx * (size_t)gy;

    if (!test_dir_initialized) {
        Eigen::Vector2d dir = ComputeDefaultTestDirection();
        test_dir_x = dir.x();
        test_dir_y = dir.y();
        test_dir_initialized = true;
    }

    // Two FIXED reference instants, exactly one day apart. Frame indices 0/1
    // never change again after the first call, so FrameRingBuffer finds them
    // already resident on every subsequent call and this is a no-op -- no
    // repeated GPU upload. magnitude(t) is globally linear in elapsed time,
    // so blending between these two fixed frames via an UNCLAMPED alpha
    // reproduces the exact value at any t, forever, regardless of run length.
    constexpr double kRefIntervalSeconds = 86400.0;
    auto load_func = [&](int frameIdx, int slot) {
        double ref_elapsed = (frameIdx == 0) ? 0.0 : kRefIntervalSeconds;
        double magnitude = test_initial_magnitude
                          + (test_growth_rate_per_day / 86400.0) * ref_elapsed;
        float vx = (float)(test_dir_x * magnitude);
        float vy = (float)(test_dir_y * magnitude);
        ocean_vx_frame_buffer[slot].assign(gridSize, vx);
        ocean_vy_frame_buffer[slot].assign(gridSize, vy);
        ocean_thickness_frame_buffer[slot].assign(gridSize, 0.0f);
    };

    bool changed = ocean_ring.Update(0, 1, -1, load_func);

    double elapsed = t - prms.SimulationStartTime;
    current_ocean_alpha = elapsed / kRefIntervalSeconds;   // unclamped by design
    return changed;
}

void CurrentInterpolator::BoxBlur(int width, int height, std::vector<float>& field, int radius)
{
    if (radius <= 0) return;
    const size_t n = (size_t)width * height;
    std::vector<float> tmp(n);

    // Horizontal pass
    #pragma omp parallel for
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int x0 = std::max(0, x - radius);
            int x1 = std::min(width - 1, x + radius);
            double sum = 0.0;
            for (int xx = x0; xx <= x1; ++xx) sum += field[(size_t)y*width + xx];
            tmp[(size_t)y*width + x] = (float)(sum / (x1 - x0 + 1));
        }
    }

    // Vertical pass
    #pragma omp parallel for
    for (int x = 0; x < width; ++x) {
        for (int y = 0; y < height; ++y) {
            int y0 = std::max(0, y - radius);
            int y1 = std::min(height - 1, y + radius);
            double sum = 0.0;
            for (int yy = y0; yy <= y1; ++yy) sum += tmp[(size_t)yy*width + x];
            field[(size_t)y*width + x] = (float)(sum / (y1 - y0 + 1));
        }
    }
}


std::pair<double, double> CurrentInterpolator::GetOceanValue(int i, int j) const
{
    if (!interpolation_enabled) return {0.0, 0.0};

    // WindInterpolator's TestWindSpeed synthetic-wind test mode: ocean current
    // speed goes to zero (water treated as standing still), regardless of
    // whether GLO12Data was configured. Water drag itself is untouched --
    // kernels.cu's water-drag term isn't gated by a Use*Data flag the way
    // wind drag is, so a zero v_w here just makes it act as damping against
    // stationary water rather than being disabled.
    if (prms.TestWindSpeed != 0.0) return {0.0, 0.0};

    const int& gx = prms.GridXTotal;
    const int& gy = prms.GridYTotal;

    if (num_frames == 0 && !test_mode_enabled) {
        // Check if GLO12 is available
        bool has_glo12 = prms.UseGLO12Data && glo12_num_frames > 0;
        if (!has_glo12) return {0.0, 0.0};
        
        // If we are here, we use GLO12 data (which populates ocean_vx_frame_buffer)
    }
    if (i < 0 || i >= gx || j < 0 || j >= gy) return {0.0, 0.0};

    size_t idx = j + static_cast<size_t>(i) * gy;

    if (!ocean_ring.IsReady()) return {0.0, 0.0};
    int s0 = ocean_ring.ActiveSlot(0);
    int s1 = ocean_ring.ActiveSlot(1);

    if (num_frames == 1) {
        return {(double)ocean_vx_frame_buffer[s0][idx], (double)ocean_vy_frame_buffer[s0][idx]};
    }

    double vx_first = ocean_vx_frame_buffer[s0][idx];
    double vx_second = ocean_vx_frame_buffer[s1][idx];
    double vy_first = ocean_vy_frame_buffer[s0][idx];
    double vy_second = ocean_vy_frame_buffer[s1][idx];

    double vx = (1.0 - current_ocean_alpha) * vx_first + current_ocean_alpha * vx_second;
    double vy = (1.0 - current_ocean_alpha) * vy_first + current_ocean_alpha * vy_second;
    
    return {vx, vy};
}

std::pair<double, double> CurrentInterpolator::GetOceanValueGeographic(int i, int j) const
{
    const auto [vx, vy] = GetOceanValue(i, j);

    // Same pixel mapping RenderGLO12FrameFields uses to get this cell's
    // lat/lon (including its Y-flip) -- see the comment there.
    const int global_x = i + prms.ModeledRegionOffsetX;
    const int global_y_grid = j + prms.ModeledRegionOffsetY;
    const int global_y = prms.InitializationImageSizeY - 1 - global_y_grid;

    Projection::LatLon proj_ll = prms.proj.ProjectPixel(global_x, global_y);
    if (!proj_ll.valid) {
        const double nan = std::numeric_limits<double>::quiet_NaN();
        return {nan, nan};
    }

    const double lat_rad = proj_ll.lat_deg * (M_PI / 180.0);
    const double lon_rad = proj_ll.lon_deg * (M_PI / 180.0);
    Projection::RotMat r = prms.proj.ComputeRotation(lat_rad, lon_rad);

    // Inverse of the forward rotation (rot.ex/ey/nx/ny applied as
    // vx_grid=u*ex+v*nx, vy_grid=u*ey+v*ny in RenderGLO12FrameFields) --
    // its transpose, since it's an orthonormal rotation (the stereographic
    // projection is conformal, so the local east/north basis it's built
    // from is orthogonal).
    const double u_geo = r.ex * vx + r.ey * vy;
    const double v_geo = r.nx * vx + r.ny * vy;
    return {u_geo, v_geo};
}

double CurrentInterpolator::GetThicknessValue(int i, int j) const
{
    if (!interpolation_enabled) return 0.0;
    if (!prms.UseGLO12Data || glo12_num_frames == 0) return 0.0;
    if (!glo12_thickness_time_covered) return 0.0;

    const int& gx = prms.GridXTotal;
    const int& gy = prms.GridYTotal;
    if (i < 0 || i >= gx || j < 0 || j >= gy) return 0.0;

    size_t idx = j + static_cast<size_t>(i) * gy;

    if (!ocean_ring.IsReady()) return 0.0;
    int s0 = ocean_ring.ActiveSlot(0);
    int s1 = ocean_ring.ActiveSlot(1);
    if (ocean_thickness_frame_buffer[s0].empty()) return 0.0;

    double t0 = ocean_thickness_frame_buffer[s0][idx];
    double t1 = ocean_thickness_frame_buffer[s1][idx];
    return (1.0 - current_ocean_alpha) * t0 + current_ocean_alpha * t1;
}

const float* CurrentInterpolator::GetOceanDataPointer(int logicalFrame, int component) const
{
    if (!interpolation_enabled) return nullptr;
    // logicalFrame: 0 or 1
    if (logicalFrame < 0 || logicalFrame > 1) return nullptr;
    if (!ocean_ring.IsReady()) return nullptr;
    int slot = ocean_ring.ActiveSlot(logicalFrame);

    if (component == 0) return ocean_vx_frame_buffer[slot].data();
    else return ocean_vy_frame_buffer[slot].data();
}

std::pair<double, double> CurrentInterpolator::GetLatLon(int i, int j) const
{
    // Global pixel coordinates
    int global_x = i + prms.ModeledRegionOffsetX;
    
    // Invert Y because PROJ coeffs are based on top-left origin image, 
    // while grid is bottom-left origin.
    int global_y_grid = j + prms.ModeledRegionOffsetY;
    int global_y = prms.InitializationImageSizeY - 1 - global_y_grid;

    Projection::LatLon ll = prms.proj.ProjectPixel(global_x, global_y);

    if (!ll.valid) {
        return {0.0, 0.0};
    }
    return {ll.lat_deg, ll.lon_deg};
}


void CurrentInterpolator::RenderThicknessAtTime(double t, int width, int height,
                                                 std::vector<float>& out_thickness,
                                                 std::vector<uint8_t>& out_valid)
{
    if (!file_glo12) {
        throw std::runtime_error("CurrentInterpolator::RenderThicknessAtTime: no GLO12 file loaded (call SetGLO12Path first)");
    }
    if (glo12_num_frames == 0) {
        throw std::runtime_error("CurrentInterpolator::RenderThicknessAtTime: GLO12 file has no time frames");
    }

    // Pick the single nearest-in-time frame (this is a one-shot initial
    // condition, not a continuously blended forcing). sithick has its own
    // (independent, possibly coarser) time axis if a separate thickness
    // file was set -- see LoadGLO12Frame for the same reasoning.
    H5::H5File* thickness_source = file_glo12_thickness ? file_glo12_thickness.get() : file_glo12.get();
    const std::vector<long long>& thickness_time_axis = glo12_thickness_times.empty() ? glo12_times : glo12_thickness_times;
    if (thickness_time_axis.empty()) {
        throw std::runtime_error("CurrentInterpolator::RenderThicknessAtTime: no time frames available for the thickness source");
    }

    long long target_ts = (long long)t;
    auto it = std::upper_bound(thickness_time_axis.begin(), thickness_time_axis.end(), target_ts);
    int idx = (it == thickness_time_axis.begin()) ? 0 : (int)std::distance(thickness_time_axis.begin(), it) - 1;
    if (idx < 0) idx = 0;
    if (idx >= (int)thickness_time_axis.size() - 1 && (int)thickness_time_axis.size() > 0) idx = (int)thickness_time_axis.size() - 1;
    int frameIdx = idx;
    if (idx + 1 < (int)thickness_time_axis.size()) {
        long long t0 = thickness_time_axis[idx];
        long long t1 = thickness_time_axis[idx + 1];
        if (std::llabs(target_ts - t1) < std::llabs(target_ts - t0)) frameIdx = idx + 1;
    }
    LOGR("CurrentInterpolator::RenderThicknessAtTime: using GLO12 (thickness) frame {} (t={}, frame_time={})",
         frameIdx, target_ts, thickness_time_axis[frameIdx]);

    const int n_lat = (int)glo12_lats.size();
    const int n_lon = (int)glo12_lons.size();
    const size_t n_elem = (size_t)n_lat * n_lon;

    std::vector<float> raw_thick_render(n_elem);
    std::vector<uint8_t> raw_valid(n_elem, 1);

    if (!thickness_source->nameExists("sithick")) {
        throw std::runtime_error("CurrentInterpolator::RenderThicknessAtTime: 'sithick' variable not found in GLO12 (thickness) file");
    }

    H5::DataSet ds = thickness_source->openDataSet("sithick");
    H5::DataSpace space = ds.getSpace();
    int ndims = space.getSimpleExtentNdims();
    std::vector<hsize_t> dims(ndims);
    space.getSimpleExtentDims(dims.data(), NULL);

    std::vector<hsize_t> start(ndims, 0);
    std::vector<hsize_t> count(ndims, 1);
    start[0] = frameIdx;
    count[ndims - 2] = n_lat;
    count[ndims - 1] = n_lon;
    space.selectHyperslab(H5S_SELECT_SET, count.data(), start.data());

    hsize_t mem_dims[2] = {(hsize_t)n_lat, (hsize_t)n_lon};
    H5::DataSpace mem_space(2, mem_dims);
    ds.read(raw_thick_render.data(), H5::PredType::NATIVE_FLOAT, mem_space, space);

    double fill_value = 9.96921e+36;
    bool has_fill = false;
    if (ds.attrExists("_FillValue")) {
        H5::Attribute att = ds.openAttribute("_FillValue");
        att.read(H5::PredType::NATIVE_DOUBLE, &fill_value);
        has_fill = true;
    }

    // See read_strict()'s identical comment: packed int16 GLO12 products
    // (e.g. GLORYS12V1) need scale_factor/add_offset unpacking; defaults to
    // a no-op for unpacked-float files.
    double scale_factor = 1.0, add_offset = 0.0;
    if (ds.attrExists("scale_factor")) {
        H5::Attribute att = ds.openAttribute("scale_factor");
        att.read(H5::PredType::NATIVE_DOUBLE, &scale_factor);
    }
    if (ds.attrExists("add_offset")) {
        H5::Attribute att = ds.openAttribute("add_offset");
        att.read(H5::PredType::NATIVE_DOUBLE, &add_offset);
    }

    int valid_count = 0;
    float min_val = 1e9f, max_val = -1e9f;
    for (size_t k = 0; k < n_elem; ++k) {
        bool is_fill = false;
        if (has_fill && std::abs((double)raw_thick_render[k] - fill_value) < std::abs(fill_value) * 1e-4) is_fill = true;
        if (std::abs(raw_thick_render[k]) > 1e30f) is_fill = true;
        if (is_fill) {
            raw_valid[k] = 0;
            raw_thick_render[k] = 0.0f;
        } else {
            raw_thick_render[k] = (float)(raw_thick_render[k] * scale_factor + add_offset);
            valid_count++;
            if (raw_thick_render[k] < min_val) min_val = raw_thick_render[k];
            if (raw_thick_render[k] > max_val) max_val = raw_thick_render[k];
        }
    }
    LOGR("CurrentInterpolator: GLO12 sithick frame {} valid pts: {}/{}, Range: [{}, {}]",
         frameIdx, valid_count, n_elem, min_val, max_val);

    out_thickness.assign((size_t)width * height, 0.0f);
    out_valid.assign((size_t)width * height, 0);

    bool is_0_360 = (glo12_lons.back() > 180.0);
    double dlat = (n_lat > 1) ? (glo12_lats[1] - glo12_lats[0]) : 1.0;
    double dlon = (n_lon > 1) ? (glo12_lons[1] - glo12_lons[0]) : 1.0;

    int confident = 0;
    #pragma omp parallel for reduction(+:confident)
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            size_t out_idx = (size_t)y * width + x;

            // Same top-left-origin-image convention as GetLatLon/LoadGLO12Frame.
            int global_y = height - 1 - y;
            Projection::LatLon ll = prms.proj.ProjectPixel(x, global_y);
            if (!ll.valid) continue;

            double r_idx = (ll.lat_deg - glo12_lats[0]) / dlat;
            double target_lon = ll.lon_deg;
            if (is_0_360 && target_lon < 0) target_lon += 360.0;
            double c_idx = (target_lon - glo12_lons[0]) / dlon;

            int r0 = (int)std::floor(r_idx);
            int c0 = (int)std::floor(c_idx);
            int r1 = r0 + 1;
            int c1 = c0 + 1;
            if (r0 < 0 || r1 >= n_lat || c0 < 0 || c1 >= n_lon) continue;

            double dr = r_idx - r0;
            double dc = c_idx - c0;

            size_t i00 = (size_t)c0 + (size_t)r0 * n_lon;
            size_t i01 = (size_t)c1 + (size_t)r0 * n_lon;
            size_t i10 = (size_t)c0 + (size_t)r1 * n_lon;
            size_t i11 = (size_t)c1 + (size_t)r1 * n_lon;

            if (!raw_valid[i00] || !raw_valid[i01] || !raw_valid[i10] || !raw_valid[i11]) {
                // Any fill/land corner -- don't blend it in as if it were
                // real ice thickness. Leave invalid for nearest-fill+blur.
                continue;
            }

            double top = raw_thick_render[i00] * (1.0 - dc) + raw_thick_render[i01] * dc;
            double bot = raw_thick_render[i10] * (1.0 - dc) + raw_thick_render[i11] * dc;
            out_thickness[out_idx] = (float)(top * (1.0 - dr) + bot * dr);
            out_valid[out_idx] = 1;
            confident++;
        }
    }
    LOGR("CurrentInterpolator: RenderThicknessAtTime confident (4-corner-valid) pixels: {}/{}",
         confident, (size_t)width * height);
}

void CurrentInterpolator::SetEnabled(bool enabled)
{
    if (interpolation_enabled == enabled) return;
    interpolation_enabled = enabled;
    if (!enabled) {
        // Reset() blocks on any in-flight preload first -- otherwise the
        // background thread could still be writing into a buffer we're
        // about to swap out from under it.
        ocean_ring.Reset();
        for(int i=0; i<NUM_SLOTS; ++i) {
            std::vector<float>().swap(ocean_vx_frame_buffer[i]);
            std::vector<float>().swap(ocean_vy_frame_buffer[i]);
            std::vector<float>().swap(ocean_thickness_frame_buffer[i]);
        }
    }
}

bool CurrentInterpolator::ProcessGLO12(double t)
{
    bool ocean_frames_changed = false;
    // --- OCEAN CURRENT (GLO12 - Alternative) ---
    // we use GLO12 to populate the ocean buffers.
    if (prms.UseGLO12Data && glo12_num_frames > 0 && num_frames == 0) {
        // Use logic similar to ERA5 (Time Series)
        // t is already an absolute Unix epoch (SimulationTime + SimulationStartTime,
        // see model.cpp), matching glo12_times' units directly -- no offset needed.
        long long target_ts = (long long)t;

        // Recorded before any clamping below -- see GetThicknessValue(), which
        // returns 0 rather than a stale/clamped frame for out-of-coverage
        // dates. Uses sithick's own (independent, possibly coarser) time
        // axis if a separate thickness file was set, else the primary one.
        const std::vector<long long>& thickness_time_axis = glo12_thickness_times.empty() ? glo12_times : glo12_thickness_times;
        glo12_thickness_time_covered = !thickness_time_axis.empty() &&
                              target_ts >= thickness_time_axis.front() &&
                              target_ts <= thickness_time_axis.back();

        auto it = std::upper_bound(glo12_times.begin(), glo12_times.end(), target_ts);
        int idx = (it == glo12_times.begin()) ? 0 : std::distance(glo12_times.begin(), it) - 1;
        
        if (idx >= glo12_num_frames - 1) idx = glo12_num_frames - 2;
        if (idx < 0) idx = 0;
         
        int f_first = idx;
        int f_second = idx + 1;
         
        long long t0 = glo12_times[f_first];
        long long t1 = glo12_times[f_second];
        double dt = (double)(t1 - t0);
        double glo_alpha = (dt > 1e-3) ? (double)(target_ts - t0) / dt : 0.0;
        if (glo_alpha < 0) glo_alpha = 0; if (glo_alpha > 1) glo_alpha = 1;
        
        int next_preload_idx;
        if (loop_mode == 0) next_preload_idx = (f_second + 1) % glo12_num_frames; // periodic
        else next_preload_idx = (f_second < glo12_num_frames - 1) ? f_second + 1 : -1; // hold last

        ocean_frames_changed = ocean_ring.Update(f_first, f_second, next_preload_idx,
            [this](int f, int s){ LoadGLO12Frame(f, s); });
        if (ocean_frames_changed) {
            LOGR("Loading GLO12 Frames {} and {}", f_first, f_second);
        }
        current_ocean_alpha = glo_alpha;
    }
    return ocean_frames_changed;
}


