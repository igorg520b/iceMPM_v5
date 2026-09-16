#include "windinterpolator.h"
#include <iostream>
#include <spdlog/spdlog.h>
#include <netcdf>
#include <cmath>
#include <chrono>
#include <filesystem>
#include <algorithm>
#include <typeinfo>
#include <fstream>
#include <limits>
#include <ctime>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <H5Cpp.h>

using namespace netCDF;


WindInterpolator::WindInterpolator(SimParams& params, DirectoryManager& dm) : prms(params), dirs(dm) {
    carra1_num_frames = 0;
}

WindInterpolator::~WindInterpolator() = default;


void WindInterpolator::SetCARRA1Path(const std::string& filePath)
{
    // Check file exists before attempting to open
    if (!std::filesystem::exists(filePath)) {
        throw std::runtime_error(fmt::format("CARRA1 file not found: {}", filePath));
    }
    for (const auto& f : carra1_files) {
        if (f.path == filePath) {
            throw std::runtime_error(fmt::format("CARRA1 file already added: {}", filePath));
        }
    }

    carra1_files.emplace_back();
    CARRA1File& f = carra1_files.back();
    f.path = filePath;

    if (prms.UseWindData) {
        LoadCARRA1FileMetadata(f);
    }
}

void WindInterpolator::LoadCARRA1FileMetadata(CARRA1File& f)
{
    LOGR("Loading CARRA1 Metadata from {}", f.path);
    const bool isFirstFile = (carra1_files.size() == 1);
    try {
        f.file = std::make_unique<netCDF::NcFile>(f.path, netCDF::NcFile::read);

        // --- 1. Latitude / Longitude dimensions ---
        netCDF::NcVar latVar = f.file->getVar("latitude");
        netCDF::NcVar lonVar = f.file->getVar("longitude");

        if (latVar.isNull() || lonVar.isNull()) {
            throw std::runtime_error("latitude/longitude variables not found in CARRA1");
        }

        auto latDims = latVar.getDims();
        int this_ny = latDims[0].getSize();
        int this_nx = latDims[1].getSize();

        if (isFirstFile) {
            carra1_ny = this_ny;
            carra1_nx = this_nx;

            carra1_lats.resize((size_t)carra1_ny * carra1_nx);
            carra1_lons.resize((size_t)carra1_ny * carra1_nx);

            latVar.getVar(carra1_lats.data());
            lonVar.getVar(carra1_lons.data());
        } else {
            // Additional files must share the exact same grid as the first
            // file -- that grid is what PrecomputeGridMapping() already
            // built the (i,j)->(lat_idx,lon_idx) mapping from.
            if (this_nx != carra1_nx || this_ny != carra1_ny) {
                throw std::runtime_error(fmt::format(
                    "CARRA1 file {} has a different grid ({}x{}) than the first file ({}x{}) -- "
                    "all CARRA1 files added to the same WindInterpolator must share one grid",
                    f.path, this_nx, this_ny, carra1_nx, carra1_ny));
            }
        }

        // --- 2. Time variable ---
        netCDF::NcVar timeVar = f.file->getVar("valid_time");
        if (timeVar.isNull()) {
            throw std::runtime_error("Time variable 'valid_time' not found in CARRA1");
        }
        auto timeDims = timeVar.getDims();
        f.num_frames = timeDims[0].getSize();

        std::vector<double> timeData(f.num_frames);
        timeVar.getVar(timeData.data());

        f.times.resize(f.num_frames);
        for (int i = 0; i < f.num_frames; ++i) {
            f.times[i] = (long long)timeData[i];
        }

        if (f.num_frames == 0) {
            throw std::runtime_error("CARRA1 time dataset empty: " + f.path);
        }

        // Append this file's times onto the global timeline. Files are
        // expected to be added in chronological order (e.g. May then
        // June) -- warn (rather than fail) if that's violated, since a
        // small overlap/gap is recoverable via ProcessCARRA1's binary
        // search, but a badly out-of-order file would silently produce
        // wrong results.
        if (!carra1_times.empty() && f.times.front() <= carra1_times.back()) {
            LOGR("WARNING: CARRA1 file {} starts at or before the end of the previously added file(s) "
                 "(new file starts {}, previous timeline ends {}) -- files should be added in "
                 "chronological order", f.path, f.times.front(), carra1_times.back());
        }
        f.global_offset = (int)carra1_times.size();
        carra1_times.insert(carra1_times.end(), f.times.begin(), f.times.end());
        carra1_num_frames = (int)carra1_times.size();

        if (carra1_start_time == 0 && !carra1_times.empty()) {
            carra1_start_time = carra1_times[0];
            LOGR("CARRA1 Start Time: {}", carra1_start_time);
        }

        LOGR("CARRA1 file {} loaded: {} frames (global offset {}); total frames now {}",
             f.path, f.num_frames, f.global_offset, carra1_num_frames);

        if (isFirstFile) {
            LOGR("CARRA1 Grid: {}x{} (Lat/Lon)", carra1_ny, carra1_nx);
            // Pre-compute the rasterized grid mappings (grid is shared by all files)
            PrecomputeGridMapping();
        }

    } catch (netCDF::exceptions::NcException& e) {
        LOGR("NetCDF error in LoadCARRA1FileMetadata: {}", e.what());
        throw std::runtime_error(std::string("CARRA1 netCDF Error: ") + e.what());
    }
}

std::pair<int, int> WindInterpolator::ResolveGlobalFrame(int globalIdx) const
{
    if (globalIdx < 0) return {-1, -1};
    for (size_t k = 0; k < carra1_files.size(); ++k) {
        const CARRA1File& f = carra1_files[k];
        if (globalIdx < f.global_offset + f.num_frames) {
            return {(int)k, globalIdx - f.global_offset};
        }
    }
    return {-1, -1};
}

std::pair<int, int> WindInterpolator::GetMappedIndices(int i, int j) const {
    if (i < 0 || i >= prms.GridXTotal || j < 0 || j >= prms.GridYTotal) return {-1, -1};
    int idx = j * prms.GridXTotal + i;
    return {mapped_lat_idx[idx], mapped_lon_idx[idx]};
}

std::pair<double, double> WindInterpolator::GridInverseProjectPixel(double lat_deg, double lon_deg) const {
    // Pure projection math (lat/lon -> raw image pixel) delegates to the
    // single shared Projection class; only the grid sub-region offset/Y-flip
    // below is specific to WindInterpolator's curvilinear grid mapping.
    auto raw = prms.proj.InverseProjectPixel(lat_deg, lon_deg);
    double global_x = raw.first;
    double global_y = raw.second;

    double i_idx = global_x - prms.ModeledRegionOffsetX;
    double global_y_grid = (prms.InitializationImageSizeY - 1) - global_y;
    double j_idx = global_y_grid - prms.ModeledRegionOffsetY;

    return {i_idx, j_idx};
}

void WindInterpolator::PrecomputeGridMapping() {
    LOGR("Precomputing (i,j) -> (lat_idx, lon_idx) CARRA1 mapping...");
    
    int gx = prms.GridXTotal;
    int gy = prms.GridYTotal;

    mapped_lat_idx.assign(gx * gy, -1);
    mapped_lon_idx.assign(gx * gy, -1);

    auto cross = [](const Eigen::Vector2d& p0, const Eigen::Vector2d& p1, const Eigen::Vector2d& p2) {
        double v1x = p1.x() - p0.x();
        double v1y = p1.y() - p0.y();
        double v2x = p2.x() - p0.x();
        double v2y = p2.y() - p0.y();
        return v1x * v2y - v1y * v2x;
    };

    // Iterate over CARRA1 cells
    for (int r = 0; r < carra1_ny - 1; ++r) {
        for (int c = 0; c < carra1_nx - 1; ++c) {
            int idx00 = r * carra1_nx + c;
            int idx01 = r * carra1_nx + (c + 1);
            int idx10 = (r + 1) * carra1_nx + c;
            int idx11 = (r + 1) * carra1_nx + (c + 1);

            double lat00 = carra1_lats[idx00], lon00 = carra1_lons[idx00];
            double lat01 = carra1_lats[idx01], lon01 = carra1_lons[idx01];
            double lat10 = carra1_lats[idx10], lon10 = carra1_lons[idx10];
            double lat11 = carra1_lats[idx11], lon11 = carra1_lons[idx11];

            auto unwrap_lon = [&](double& tgt, double ref) {
                while (tgt - ref > 180.0) tgt -= 360.0;
                while (ref - tgt > 180.0) tgt += 360.0;
            };

            unwrap_lon(lon01, lon00);
            unwrap_lon(lon10, lon00);
            unwrap_lon(lon11, lon00);

            auto p00_idx = GridInverseProjectPixel(lat00, lon00);
            auto p01_idx = GridInverseProjectPixel(lat01, lon01);
            auto p10_idx = GridInverseProjectPixel(lat10, lon10);
            auto p11_idx = GridInverseProjectPixel(lat11, lon11);

            Eigen::Vector2d p00(p00_idx.first, p00_idx.second);
            Eigen::Vector2d p01(p01_idx.first, p01_idx.second);
            Eigen::Vector2d p10(p10_idx.first, p10_idx.second);
            Eigen::Vector2d p11(p11_idx.first, p11_idx.second);

            double min_i = std::min({p00.x(), p01.x(), p10.x(), p11.x()});
            double max_i = std::max({p00.x(), p01.x(), p10.x(), p11.x()});
            double min_j = std::min({p00.y(), p01.y(), p10.y(), p11.y()});
            double max_j = std::max({p00.y(), p01.y(), p10.y(), p11.y()});

            int start_i = std::max(0, (int)std::floor(min_i));
            int end_i   = std::min(gx - 1, (int)std::ceil(max_i));
            int start_j = std::max(0, (int)std::floor(min_j));
            int end_j   = std::min(gy - 1, (int)std::ceil(max_j));

            if (start_i > end_i || start_j > end_j) continue;

            bool is_ccw = cross(p00, p01, p11) > 0;
            
            for (int j = start_j; j <= end_j; ++j) {
                for (int i = start_i; i <= end_i; ++i) {
                    Eigen::Vector2d pt(i, j);
                    
                    double c1 = cross(p00, p01, pt);
                    double c2 = cross(p01, p11, pt);
                    double c3 = cross(p11, p10, pt);
                    double c4 = cross(p10, p00, pt);

                    bool inside = is_ccw ? (c1>=0 && c2>=0 && c3>=0 && c4>=0) 
                                         : (c1<=0 && c2<=0 && c3<=0 && c4<=0);

                    if (inside) {
                        mapped_lat_idx[j * gx + i] = r;
                        mapped_lon_idx[j * gx + i] = c;
                    }
                }
            }
        }
    }

    int unmapped_count = 0;
    for (size_t k = 0; k < mapped_lat_idx.size(); ++k) {
        if (mapped_lat_idx[k] == -1 || mapped_lon_idx[k] == -1) {
            unmapped_count++;
        }
    }
    LOGR("Precomputing mapping completed. Unmapped pixels: {} out of {}", unmapped_count, gx * gy);
    //if(unmapped_count > 0) throw std::runtime_error("void WindInterpolator::PrecomputeGridMapping() unmapped cells");
}


void WindInterpolator::LoadWindFrame(int frameIdx, int bufferSlot)
{
    if (carra1_files.empty() || !prms.UseWindData) return;

    size_t gridSize = (size_t)prms.GridXTotal * (size_t)prms.GridYTotal;
    wind_vx_frame_buffer[bufferSlot].resize(gridSize);
    wind_vy_frame_buffer[bufferSlot].resize(gridSize);
    wind_temp_frame_buffer[bufferSlot].resize(gridSize);

    RenderFrameFields(frameIdx, /*computeWind=*/true, /*computeTemp=*/true,
                       wind_vx_frame_buffer[bufferSlot].data(),
                       wind_vy_frame_buffer[bufferSlot].data(),
                       wind_temp_frame_buffer[bufferSlot].data());
}

void WindInterpolator::RenderFrameFields(int frameIdx, bool computeWind, bool computeTemp,
                                          float* out_vx, float* out_vy, float* out_temp,
                                          bool skipCache)
{
    if (carra1_files.empty()) {
        throw std::runtime_error("RenderFrameFields: no CARRA1 data loaded");
    }

    auto [fileIdx, localIdx] = ResolveGlobalFrame(frameIdx);
    if (fileIdx < 0) {
        throw std::runtime_error(fmt::format("RenderFrameFields: global frame {} is out of range "
                                              "(carra1_num_frames={})", frameIdx, carra1_num_frames));
    }

    const int& gx = prms.GridXTotal;
    const int& gy = prms.GridYTotal;
    size_t gridSize = (size_t)gx * (size_t)gy;

    // Try the HDF5 caches first -- a hit on both means we can skip the
    // netCDF read and the per-cell solve entirely.
    bool wind_from_cache = false;
    if (computeWind) {
        if (!skipCache) {
            wind_from_cache = TryGetCachedWindFrame(frameIdx, out_vx, out_vy);
        }
        if (!wind_from_cache) {
            std::fill(out_vx, out_vx + gridSize, 0.0f);
            std::fill(out_vy, out_vy + gridSize, 0.0f);
        }
    }
    bool needWindCompute = computeWind && !wind_from_cache;

    bool temp_from_cache = false;
    if (computeTemp) {
        if (!skipCache) {
            temp_from_cache = TryGetCachedTemperatureFrame(frameIdx, out_temp);
        }
        if (!temp_from_cache) {
            std::fill(out_temp, out_temp + gridSize, 0.0f);
        }
    }
    bool needTempCompute = computeTemp && !temp_from_cache;

    if (!needWindCompute && !needTempCompute) {
        // Everything requested was satisfied by the cache(s) -- no netCDF
        // I/O, no per-cell solve.
        if (on_loading_progress) on_loading_progress();
        return;
    }

    netCDF::NcFile* file = carra1_files[fileIdx].file.get();

    auto tLoadStart = std::chrono::high_resolution_clock::now();

    // Retrieve the unrotated wind/temperature data fields directly from the
    // CARRA1 NetCDF frame buffer. We preallocate these buffers at the class
    // level to avoid recreating them on every single simulation frame.
    size_t n_elem = (size_t)carra1_ny * carra1_nx;
    if (needWindCompute) {
        raw_u.resize(n_elem);
        raw_v.resize(n_elem);
    }
    if (needTempCompute) raw_skt.resize(n_elem);

    auto vecToStr = [](const std::vector<size_t>& v) {
        std::string s = "[";
        for (size_t i = 0; i < v.size(); ++i) { if (i) s += ","; s += std::to_string(v[i]); }
        s += "]";
        return s;
    };
    std::string current_step = "opening variables";

    try {
        if (needWindCompute) {
            netCDF::NcVar uVar = file->getVar("u10");
            netCDF::NcVar vVar = file->getVar("v10");
            if (uVar.isNull() || vVar.isNull()) {
                throw std::runtime_error("u10/v10 variables not found in CARRA1 data");
            }

            auto uDims = uVar.getDims();
            bool hasLevel = (uDims.size() == 4);

            std::vector<size_t> start, count;
            if (!hasLevel) {
                start = {(size_t)localIdx, 0, 0};
                count = {1, (size_t)carra1_ny, (size_t)carra1_nx};
            } else {
                start = {(size_t)localIdx, 0, 0, 0};
                count = {1, 1, (size_t)carra1_ny, (size_t)carra1_nx};
            }

            current_step = fmt::format("u10 read (start={}, count={})", vecToStr(start), vecToStr(count));
            uVar.getVar(start, count, raw_u.data());
            current_step = fmt::format("v10 read (start={}, count={})", vecToStr(start), vecToStr(count));
            vVar.getVar(start, count, raw_v.data());
        }

        if (needTempCompute) {
            netCDF::NcVar sktVar = file->getVar("skt");
            if (sktVar.isNull()) {
                throw std::runtime_error("skt (surface temperature) variable not found in CARRA1 data: " + carra1_files[fileIdx].path);
            }

            // skt is a surface (not height-above-ground) variable, so it's
            // typically 3D [time,y,x] even when u10/v10 are 4D [time,level,y,x]
            // -- use its own dimensionality rather than assuming it matches uDims.
            auto sktDims = sktVar.getDims();
            bool sktHasLevel = (sktDims.size() == 4);
            std::vector<size_t> sktStart, sktCount;
            if (!sktHasLevel) {
                sktStart = {(size_t)localIdx, 0, 0};
                sktCount = {1, (size_t)carra1_ny, (size_t)carra1_nx};
            } else {
                sktStart = {(size_t)localIdx, 0, 0, 0};
                sktCount = {1, 1, (size_t)carra1_ny, (size_t)carra1_nx};
            }
            current_step = fmt::format("skt read (start={}, count={})", vecToStr(sktStart), vecToStr(sktCount));
            sktVar.getVar(sktStart, sktCount, raw_skt.data());
        }

    } catch (const netCDF::exceptions::NcException& e) {
        LOGR("Error reading CARRA1 frame {} (file {}, local frame {}) during {}: [{}] {}",
             frameIdx, carra1_files[fileIdx].path, localIdx, current_step, typeid(e).name(), e.what());
        throw std::runtime_error(fmt::format("CARRA1 netCDF error during {} (frame {}, file {}, local frame {}): {}",
                                              current_step, frameIdx, carra1_files[fileIdx].path, localIdx, e.what()));
    }

    auto tLoadEnd = std::chrono::high_resolution_clock::now();

    auto tInterpStart = std::chrono::high_resolution_clock::now();

    // Iterate over the simulation grid
#pragma omp parallel for
    for(int j = 0; j < gy; ++j) {
        for(int i = 0; i < gx; ++i) {
            const int map_idx = j * gx + i;
            const int dest_idx = j + i * gy;
            int r = mapped_lat_idx[map_idx];
            int c = mapped_lon_idx[map_idx];

            if (r == -1 || c == -1) continue; // already zero-filled above

            // CARRA1 quads are defined by (r, c), (r, c+1), (r+1, c), (r+1, c+1)
            int q00 = r * carra1_nx + c;
            int q01 = r * carra1_nx + (c + 1);
            int q10 = (r + 1) * carra1_nx + c;
            int q11 = (r + 1) * carra1_nx + (c + 1);

            // 1) Get lat/lon coordinates of (i,j) and quad corners, and store
            //    them in Eigen::Vector2d. We assume the quad is not curved
            //    in the local region.
            int global_x = i + prms.ModeledRegionOffsetX;
            int global_y_grid = j + prms.ModeledRegionOffsetY;
            int global_y = prms.InitializationImageSizeY - 1 - global_y_grid;

            Projection::LatLon proj_ll = prms.proj.ProjectPixel(global_x, global_y);
            if (!proj_ll.valid) continue; // already zero-filled above

            Eigen::Vector2d pt(proj_ll.lon_deg, proj_ll.lat_deg);

            Eigen::Vector2d p00(carra1_lons[q00], carra1_lats[q00]);
            Eigen::Vector2d p01(carra1_lons[q01], carra1_lats[q01]);
            Eigen::Vector2d p10(carra1_lons[q10], carra1_lats[q10]);
            Eigen::Vector2d p11(carra1_lons[q11], carra1_lats[q11]);

            // Normalize longitude wrapping relative to p00
            auto unwrap_lon = [&](double& tgt, double ref) {
                while (tgt - ref > 180.0) tgt -= 360.0;
                while (ref - tgt > 180.0) tgt += 360.0;
            };
            unwrap_lon(p01.x(), p00.x());
            unwrap_lon(p10.x(), p00.x());
            unwrap_lon(p11.x(), p00.x());
            unwrap_lon(pt.x(), p00.x());

            // -------------------------------------------------------------
            // 2) Iterative Newton-Raphson solver for Bilinear Coordinates
            //    We want to find an Eigen::Vector2d st = (s, t) such that:
            //    P(s, t) = p00(1-s)(1-t) + p01(s)(1-t) + p10(1-s)(t) + p11(s)(t)
            //    matches our target point 'pt'. This is the dominant per-cell
            //    cost, and is shared by wind and temperature alike -- it only
            //    runs at all when at least one of them needs computing (see
            //    the early-return above for the fully-cached case).
            // -------------------------------------------------------------
            // Start with an initial guess perfectly in the middle of the quad
            Eigen::Vector2d st(0.5, 0.5);

            const int max_iters = 8;
            for (int iter = 0; iter < max_iters; ++iter) {
                double s = st.x();
                double t = st.y();

                // Compute the current interpolated point at (s, t)
                Eigen::Vector2d p_est = p00*(1.0-s)*(1.0-t) + p01*s*(1.0-t) + p10*(1.0-s)*t + p11*s*t;

                // Compute the residual (difference from target)
                Eigen::Vector2d r = pt - p_est;

                // If the error is extremely small, we have converged
                if (r.squaredNorm() < 1e-10) {
                    break;
                }

                // Compute the Jacobian matrix: J = [dp/ds, dp/dt]
                // dp/ds = derivative of P(s, t) with respect to s
                Eigen::Vector2d dp_ds = -p00*(1.0-t) + p01*(1.0-t) - p10*t + p11*t;
                // dp/dt = derivative of P(s, t) with respect to t
                Eigen::Vector2d dp_dt = -p00*(1.0-s) - p01*s + p10*(1.0-s) + p11*s;

                Eigen::Matrix2d J;
                J.col(0) = dp_ds;
                J.col(1) = dp_dt;

                // Update (s, t) using Newton-Raphson: st_{n+1} = st_{n} + J^{-1} * r
                st += J.inverse() * r;
            }

            // Clamp the bilinear coordinates strictly to [0, 1]
            double s = std::max(0.0, std::min(1.0, st.x()));
            double t = std::max(0.0, std::min(1.0, st.y()));

            if (needWindCompute) {
                double u00 = raw_u[q00], v00 = raw_v[q00];
                double u01 = raw_u[q01], v01 = raw_v[q01];
                double u10 = raw_u[q10], v10 = raw_v[q10];
                double u11 = raw_u[q11], v11 = raw_v[q11];

                // Use bilinear coordinates to interpolate raw_u, raw_v
                double u_local = u00*(1.0-s)*(1.0-t) + u01*s*(1.0-t) + u10*(1.0-s)*t + u11*s*t;
                double v_local = v00*(1.0-s)*(1.0-t) + v01*s*(1.0-t) + v10*(1.0-s)*t + v11*s*t;

                // Transform local interpolated wind to grid components in two formal steps
                Eigen::Vector2d geo_wind = TransformWind_LocalToGeographic(u_local, v_local, p00, p01, p10);

                double lat_rad = pt.y() * (M_PI/180.0);
                double lon_rad = pt.x() * (M_PI/180.0);
                Eigen::Vector2d grid_wind = TransformWind_GeographicToGrid(geo_wind, lat_rad, lon_rad);

                out_vx[dest_idx] = (float)grid_wind.x();
                out_vy[dest_idx] = (float)grid_wind.y();
            }

            if (needTempCompute) {
                double t00 = raw_skt[q00], t01 = raw_skt[q01], t10 = raw_skt[q10], t11 = raw_skt[q11];
                // Temperature is a scalar (no direction), so it just needs
                // the same bilinear interpolation -- no local->geographic->
                // grid rotation step, unlike wind.
                double temp_local = t00*(1.0-s)*(1.0-t) + t01*s*(1.0-t) + t10*(1.0-s)*t + t11*s*t;
                out_temp[dest_idx] = (float)temp_local;
            }
        }
    }

    auto tInterpEnd = std::chrono::high_resolution_clock::now();
    double load_ms = std::chrono::duration<double, std::milli>(tLoadEnd - tLoadStart).count();
    double interp_ms = std::chrono::duration<double, std::milli>(tInterpEnd - tInterpStart).count();
    LOGR("CARRA1: Frame {} loaded in {:.1f} ms | Interpolated (gx/gy loop) in {:.1f} ms", frameIdx, load_ms, interp_ms);

    if (on_loading_progress) on_loading_progress();
}


bool WindInterpolator::ProcessCARRA1(double t)
{
    // WIND (CARRA1) ---
    bool wind_frames_changed = false;
    if (prms.UseWindData && carra1_num_frames > 0) {
         // t is already an absolute Unix epoch (SimulationTime + SimulationStartTime,
         // see model.cpp), matching carra1_times' units directly -- no offset needed.
         long long target_ts = (long long)t;
         auto it = std::upper_bound(carra1_times.begin(), carra1_times.end(), target_ts);
         int idx = (it == carra1_times.begin()) ? 0 : std::distance(carra1_times.begin(), it) - 1;
         
         if (idx >= carra1_num_frames - 1) idx = carra1_num_frames - 2;
         if (idx < 0) idx = 0;
         
         int w_first = idx;
         int w_second = idx + 1;
         
         long long t0 = carra1_times[w_first];
         long long t1 = carra1_times[w_second];
         double dt = (double)(t1 - t0);
         double w_alpha = (dt > 1e-3) ? (double)(target_ts - t0) / dt : 0.0;
         if (w_alpha < 0) w_alpha = 0; if (w_alpha > 1) w_alpha = 1;

         // CARRA1 is typically non-periodic in this context (linear time series).
         int next_preload_idx = (w_second < carra1_num_frames - 1) ? w_second + 1 : -1;

         wind_frames_changed = wind_ring.Update(w_first, w_second, next_preload_idx,
             [this](int f, int s){ LoadWindFrame(f, s); });
         if (wind_frames_changed) {
             LOGR("Loading Wind Frames {} and {}; using Ring Buffer logic", w_first, w_second);
         }
         current_wind_alpha = w_alpha;
    }
    return wind_frames_changed;
}

std::pair<double, double> WindInterpolator::GetWindValue(int i, int j) const
{
    if (!interpolation_enabled) return {0.0, 0.0};

    if (!prms.UseWindData) return {0.0, 0.0};
    
    const int& gx = prms.GridXTotal;
    const int& gy = prms.GridYTotal;

    if (!wind_ring.IsReady()) return {0.0, 0.0};
    int s0 = wind_ring.ActiveSlot(0);
    int s1 = wind_ring.ActiveSlot(1);

    if (wind_vx_frame_buffer[s0].empty()) return {0.0, 0.0};

    size_t idx = (size_t)i * gy + j;
    if (idx >= wind_vx_frame_buffer[s0].size()) return {0.0, 0.0};

    double vx0 = wind_vx_frame_buffer[s0][idx];
    double vx1 = wind_vx_frame_buffer[s1][idx];
    double vy0 = wind_vy_frame_buffer[s0][idx];
    double vy1 = wind_vy_frame_buffer[s1][idx];

    double vx = vx0 * (1.0 - current_wind_alpha) + vx1 * current_wind_alpha;
    double vy = vy0 * (1.0 - current_wind_alpha) + vy1 * current_wind_alpha;

    return {vx, vy};
}

std::pair<double, double> WindInterpolator::GetWindValueGeographic(int i, int j) const
{
    const auto [vx, vy] = GetWindValue(i, j);

    // Same pixel mapping RenderFrameFields uses to get this cell's lat/lon
    // (including its Y-flip) -- see the comment there.
    const int global_x = i + prms.ModeledRegionOffsetX;
    const int global_y_grid = j + prms.ModeledRegionOffsetY;
    const int global_y = prms.InitializationImageSizeY - 1 - global_y_grid;

    Projection::LatLon proj_ll = prms.proj.ProjectPixel(global_x, global_y);
    // NaN, not {0,0} -- {0,0} is a legitimate return for genuinely calm
    // wind, and callers that need to tell "no data here" apart from "real
    // zero wind here" (e.g. a wind rose tallying how many of its sample
    // cells actually have data) need those to be distinguishable.
    if (!proj_ll.valid) {
        const double nan = std::numeric_limits<double>::quiet_NaN();
        return {nan, nan};
    }

    const double lat_rad = proj_ll.lat_deg * (M_PI / 180.0);
    const double lon_rad = proj_ll.lon_deg * (M_PI / 180.0);
    Projection::RotMat r = prms.proj.ComputeRotation(lat_rad, lon_rad);

    // Inverse of TransformWind_GeographicToGrid's rotation matrix
    // [[ex,nx],[ey,ny]] -- its transpose, since it's an orthonormal
    // rotation (the stereographic projection is conformal, so the local
    // east/north basis it's built from is orthogonal).
    const double u_geo = r.ex * vx + r.ey * vy;
    const double v_geo = r.nx * vx + r.ny * vy;
    return {u_geo, v_geo};
}

double WindInterpolator::GetTemperatureValue(int i, int j) const
{
    if (!interpolation_enabled) return 0.0;
    if (!prms.UseWindData) return 0.0;

    const int& gy = prms.GridYTotal;

    if (!wind_ring.IsReady()) return 0.0;
    int s0 = wind_ring.ActiveSlot(0);
    int s1 = wind_ring.ActiveSlot(1);

    if (wind_temp_frame_buffer[s0].empty()) return 0.0;

    size_t idx = (size_t)i * gy + j;
    if (idx >= wind_temp_frame_buffer[s0].size()) return 0.0;

    double t0 = wind_temp_frame_buffer[s0][idx];
    double t1 = wind_temp_frame_buffer[s1][idx];

    return t0 * (1.0 - current_wind_alpha) + t1 * current_wind_alpha;
}

std::pair<int, int> WindInterpolator::GetFrameRangeForTimeSpan(long long t_start, long long t_end) const
{
    if (carra1_times.empty() || t_end < t_start)
        throw std::runtime_error("GetFrameRangeForTimeSpan: no CARRA1 data or invalid span");

    auto itFirst = std::lower_bound(carra1_times.begin(), carra1_times.end(), t_start);
    auto itLast  = std::upper_bound(carra1_times.begin(), carra1_times.end(), t_end);

    if (itFirst == carra1_times.end() || itFirst == itLast)
        throw std::runtime_error("GetFrameRangeForTimeSpan: span not covered by loaded CARRA1 data");

    int firstIdx = (int)std::distance(carra1_times.begin(), itFirst);
    int lastIdx  = (int)std::distance(carra1_times.begin(), itLast) - 1;

    if (carra1_times[firstIdx] != t_start || carra1_times[lastIdx] != t_end) {
        throw std::runtime_error(fmt::format(
            "GetFrameRangeForTimeSpan: [{}, {}] must exactly match CARRA1 frame timestamps; "
            "nearest available are [{}, {}]", t_start, t_end, carra1_times[firstIdx], carra1_times[lastIdx]));
    }

    return {firstIdx, lastIdx};
}

long long WindInterpolator::GetCARRA1Timestamp(int globalFrameIdx) const
{
    if (globalFrameIdx < 0 || globalFrameIdx >= (int)carra1_times.size()) {
        throw std::out_of_range(fmt::format(
            "GetCARRA1Timestamp: frame index {} out of range (carra1_num_frames={})",
            globalFrameIdx, carra1_num_frames));
    }
    return carra1_times[globalFrameIdx];
}

void WindInterpolator::GetTemperatureFrameExact(int globalFrameIdx, std::vector<float>& out)
{
    size_t gridSize = (size_t)prms.GridXTotal * (size_t)prms.GridYTotal;
    out.resize(gridSize);
    RenderFrameFields(globalFrameIdx, /*computeWind=*/false, /*computeTemp=*/true,
                       nullptr, nullptr, out.data());
}

void WindInterpolator::RenderTempAndWindFrameUncached(int globalFrameIdx, std::vector<float>& out_vx,
                                                       std::vector<float>& out_vy, std::vector<float>& out_temp)
{
    size_t gridSize = (size_t)prms.GridXTotal * (size_t)prms.GridYTotal;
    out_vx.resize(gridSize);
    out_vy.resize(gridSize);
    out_temp.resize(gridSize);
    RenderFrameFields(globalFrameIdx, /*computeWind=*/true, /*computeTemp=*/true,
                       out_vx.data(), out_vy.data(), out_temp.data(), /*skipCache=*/true);
}

void WindInterpolator::ExportTemperatureTimeSeries(double lat_deg, double lon_deg, const std::string& csvPath)
{
    if (carra1_files.empty()) {
        throw std::runtime_error("ExportTemperatureTimeSeries: no CARRA1 data loaded");
    }

    // Nearest-neighbor search over the raw CARRA1 lat/lon raster -- not the
    // sim grid, so this works independently of any project having been
    // prepared. A simple planar (unweighted) distance is fine at CARRA1's
    // ~2.5km regional grid spacing -- but CARRA1 stores longitude in
    // [0, 360), while callers pass lon_deg in [-180, 180] (e.g. the
    // preparer dialog defaults to -69), so the raw difference must be
    // wrapped into [-180, 180] before squaring (same fix as RenderFrameFields'
    // unwrap_lon) -- otherwise a domain near lon=-66 compares against raw
    // CARRA1 values near 294, and the "nearest" point found is nonsense
    // (near the Greenwich meridian, off by the full 360-degree offset).
    size_t n_elem = (size_t)carra1_ny * carra1_nx;
    double best_d2 = std::numeric_limits<double>::max();
    long long best_idx = -1;
    for (size_t k = 0; k < n_elem; ++k) {
        double dlat = carra1_lats[k] - lat_deg;
        double dlon = carra1_lons[k] - lon_deg;
        dlon -= 360.0 * std::round(dlon / 360.0);   // wrap to [-180, 180]
        double d2 = dlat * dlat + dlon * dlon;
        if (d2 < best_d2) { best_d2 = d2; best_idx = (long long)k; }
    }
    if (best_idx < 0) {
        throw std::runtime_error("ExportTemperatureTimeSeries: empty CARRA1 grid");
    }
    int r = (int)(best_idx / carra1_nx);
    int c = (int)(best_idx % carra1_nx);
    LOGR("ExportTemperatureTimeSeries: requested ({}, {}), nearest CARRA1 cell (r={}, c={}) at ({}, {})",
         lat_deg, lon_deg, r, c, carra1_lats[best_idx], carra1_lons[best_idx]);

    std::ofstream out(csvPath);
    if (!out) {
        throw std::runtime_error("ExportTemperatureTimeSeries: cannot open " + csvPath + " for writing");
    }
    out << "timestamp_utc,epoch_unix,temperature_C\n";

    for (int frameIdx = 0; frameIdx < carra1_num_frames; ++frameIdx) {
        auto [fileIdx, localIdx] = ResolveGlobalFrame(frameIdx);
        netCDF::NcFile* file = carra1_files[fileIdx].file.get();
        netCDF::NcVar sktVar = file->getVar("skt");
        if (sktVar.isNull()) {
            throw std::runtime_error("ExportTemperatureTimeSeries: 'skt' variable not found in " + carra1_files[fileIdx].path);
        }

        auto dims = sktVar.getDims();
        bool hasLevel = (dims.size() == 4);
        std::vector<size_t> start, count;
        if (!hasLevel) {
            start = {(size_t)localIdx, (size_t)r, (size_t)c};
            count = {1, 1, 1};
        } else {
            start = {(size_t)localIdx, 0, (size_t)r, (size_t)c};
            count = {1, 1, 1, 1};
        }
        float kelvin = 0.0f;
        sktVar.getVar(start, count, &kelvin);
        double celsius = (double)kelvin - 273.15;

        long long ts = carra1_times[frameIdx];
        std::time_t tt = (std::time_t)ts;
        std::tm tm_utc;
        gmtime_r(&tt, &tm_utc);
        char buf[32];
        std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%SZ", &tm_utc);

        out << buf << "," << ts << "," << celsius << "\n";
    }
    out.close();
    LOGR("ExportTemperatureTimeSeries: wrote {} frames to {}", carra1_num_frames, csvPath);
}

// --- HDF5 temperature-frame cache -------------------------------------------
//
// Layout (written by preparer's "Precompute Temperature Cache" tool):
//   /temperature : float[num_frames][GridXTotal][GridYTotal], chunked (1,gx,gy)
//   /timestamps  : int64[num_frames], Unix epoch seconds, same frame order
//                  (kept for external inspection only -- not read back here)
// Both extendible along the frame dimension. /temperature carries
// GridXTotal/GridYTotal attributes, checked on open.
//
// Cache generation always renders every currently-loaded CARRA1 frame in a
// single pass (0..carra1_num_frames-1, in order), so a cache is either
// complete -- in which case a CARRA1 global frame index is also a valid
// index into /temperature directly, no per-frame timestamp lookup needed --
// or it doesn't exist yet. "The file doesn't exist" is the only condition
// treated as a normal, silent miss (checked via std::filesystem::exists,
// never via catching an exception); a mismatched grid size or frame count
// means the cache is stale/incomplete and throws instead of being used
// partially.

std::string WindInterpolator::TemperatureCacheFilename() const
{
    return fmt::format("{}/carra1_temp_cache/temp_{}x{}.h5", dirs.DataCacheRoot(), prms.GridXTotal, prms.GridYTotal);
}

void WindInterpolator::EnsureTemperatureCacheOpen()
{
    if (temp_cache_open_attempted) return;
    temp_cache_open_attempted = true;

    std::string path = TemperatureCacheFilename();
    if (!std::filesystem::exists(path)) return; // no cache yet -- not an error

    auto file = std::make_unique<H5::H5File>(path, H5F_ACC_RDONLY);
    H5::DataSet tempDs = file->openDataSet("temperature");

    int gx_attr = 0, gy_attr = 0;
    tempDs.openAttribute("GridXTotal").read(H5::PredType::NATIVE_INT, &gx_attr);
    tempDs.openAttribute("GridYTotal").read(H5::PredType::NATIVE_INT, &gy_attr);
    if (gx_attr != prms.GridXTotal || gy_attr != prms.GridYTotal) {
        throw std::runtime_error(fmt::format(
            "Temperature cache {} has grid attributes {}x{} that don't match its own filename ({}x{})",
            path, gx_attr, gy_attr, prms.GridXTotal, prms.GridYTotal));
    }

    hsize_t dims[3];
    tempDs.getSpace().getSimpleExtentDims(dims, nullptr);
    if ((int)dims[0] != carra1_num_frames) {
        throw std::runtime_error(fmt::format(
            "Temperature cache {} has {} frame(s) but {} CARRA1 frame(s) are currently loaded -- "
            "regenerate it via Tools > Precompute Temperature Cache",
            path, dims[0], carra1_num_frames));
    }

    temp_cache_num_frames = (int)dims[0];
    temp_cache_file = std::move(file);
    LOGR("Temperature cache opened: {} ({} frames)", path, temp_cache_num_frames);
}

void WindInterpolator::InvalidateTemperatureCache()
{
    std::lock_guard<std::mutex> lock(temp_cache_mutex);
    temp_cache_file.reset();
    temp_cache_num_frames = 0;
    temp_cache_open_attempted = false;
}

bool WindInterpolator::HasTemperatureCache()
{
    std::lock_guard<std::mutex> lock(temp_cache_mutex);
    EnsureTemperatureCacheOpen();
    return temp_cache_file != nullptr;
}

bool WindInterpolator::TryGetCachedTemperatureFrame(int frameIdx, float* out_temp)
{
    std::lock_guard<std::mutex> lock(temp_cache_mutex);

    EnsureTemperatureCacheOpen();
    if (!temp_cache_file) return false;

    H5::DataSet tempDs = temp_cache_file->openDataSet("temperature");
    hsize_t dims[3];
    tempDs.getSpace().getSimpleExtentDims(dims, nullptr);

    hsize_t offset[3] = {(hsize_t)frameIdx, 0, 0};
    hsize_t count[3]  = {1, dims[1], dims[2]};
    H5::DataSpace filespace = tempDs.getSpace();
    filespace.selectHyperslab(H5S_SELECT_SET, count, offset);

    hsize_t mem_dims[2] = {dims[1], dims[2]};
    H5::DataSpace memspace(2, mem_dims);

    tempDs.read(out_temp, H5::PredType::NATIVE_FLOAT, memspace, filespace);
    return true;
}

// --- HDF5 wind-frame cache ---------------------------------------------------
//
// Layout (written by preparer's "Precompute Wind Cache" tool):
//   /vx, /vy    : float[num_frames][GridXTotal][GridYTotal], chunked (1,gx,gy)
//   /timestamps : int64[num_frames], Unix epoch seconds, same frame order
// Both /vx and /vy are extendible along the frame dimension; /vx carries
// GridXTotal/GridYTotal attributes, checked on open. Otherwise structurally
// identical to the temperature cache above -- see its comment for the
// silent-miss-vs-throw policy, which applies here unchanged.

std::string WindInterpolator::WindCacheFilename() const
{
    return fmt::format("{}/carra1_wind_cache/wind_{}x{}.h5", dirs.DataCacheRoot(), prms.GridXTotal, prms.GridYTotal);
}

void WindInterpolator::EnsureWindCacheOpen()
{
    if (wind_cache_open_attempted) return;
    wind_cache_open_attempted = true;

    std::string path = WindCacheFilename();
    if (!std::filesystem::exists(path)) return; // no cache yet -- not an error

    auto file = std::make_unique<H5::H5File>(path, H5F_ACC_RDONLY);
    H5::DataSet vxDs = file->openDataSet("vx");

    int gx_attr = 0, gy_attr = 0;
    vxDs.openAttribute("GridXTotal").read(H5::PredType::NATIVE_INT, &gx_attr);
    vxDs.openAttribute("GridYTotal").read(H5::PredType::NATIVE_INT, &gy_attr);
    if (gx_attr != prms.GridXTotal || gy_attr != prms.GridYTotal) {
        throw std::runtime_error(fmt::format(
            "Wind cache {} has grid attributes {}x{} that don't match its own filename ({}x{})",
            path, gx_attr, gy_attr, prms.GridXTotal, prms.GridYTotal));
    }

    hsize_t dims[3];
    vxDs.getSpace().getSimpleExtentDims(dims, nullptr);
    if ((int)dims[0] != carra1_num_frames) {
        throw std::runtime_error(fmt::format(
            "Wind cache {} has {} frame(s) but {} CARRA1 frame(s) are currently loaded -- "
            "regenerate it via Tools > Precompute Wind Cache",
            path, dims[0], carra1_num_frames));
    }

    wind_cache_num_frames = (int)dims[0];
    wind_cache_file = std::move(file);
    LOGR("Wind cache opened: {} ({} frames)", path, wind_cache_num_frames);
}

void WindInterpolator::InvalidateWindCache()
{
    std::lock_guard<std::mutex> lock(wind_cache_mutex);
    wind_cache_file.reset();
    wind_cache_num_frames = 0;
    wind_cache_open_attempted = false;
}

bool WindInterpolator::HasWindCache()
{
    std::lock_guard<std::mutex> lock(wind_cache_mutex);
    EnsureWindCacheOpen();
    return wind_cache_file != nullptr;
}

bool WindInterpolator::TryGetCachedWindFrame(int frameIdx, float* out_vx, float* out_vy)
{
    std::lock_guard<std::mutex> lock(wind_cache_mutex);

    EnsureWindCacheOpen();
    if (!wind_cache_file) return false;

    auto readComponent = [&](const char* name, float* out) {
        H5::DataSet ds = wind_cache_file->openDataSet(name);
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
    return true;
}

// ProjectPixel/ComputeRotation now live in the single shared Projection
// class (prms.proj) -- see projection.h/.cpp. ComputeRotationMatrix below
// is a thin adapter converting prms.proj.ComputeRotation's flat RotMat into
// the Eigen::Matrix2d shape TransformWind_GeographicToGrid expects.
Eigen::Matrix2d WindInterpolator::ComputeRotationMatrix(double phi, double lam) const
{
    Projection::RotMat r = prms.proj.ComputeRotation(phi, lam);
    Eigen::Matrix2d rot;
    rot(0,0) = r.ex; // East -> X
    rot(1,0) = r.ey; // East -> Y
    rot(0,1) = r.nx; // North -> X
    rot(1,1) = r.ny; // North -> Y
    return rot;
}

void WindInterpolator::SetEnabled(bool enabled)
{
    if (interpolation_enabled == enabled) return;
    interpolation_enabled = enabled;
    if (!enabled) {
        // Reset() blocks on any in-flight preload first -- otherwise the
        // background thread could still be writing into a buffer we're
        // about to swap out from under it.
        wind_ring.Reset();
        for (int i=0; i<NUM_SLOTS; ++i) {
            std::vector<float>().swap(wind_vx_frame_buffer[i]);
            std::vector<float>().swap(wind_vy_frame_buffer[i]);
            std::vector<float>().swap(wind_temp_frame_buffer[i]);
        }
    }
}

// We need SetTime
bool WindInterpolator::SetTime(double t) {
    if (!interpolation_enabled) return false;

    // TestWindSpeed (SimParams, JSON-configurable): a nonzero value switches
    // this class into its own synthetic-wind test mode -- constant magnitude,
    // no growth, fixed Nares Strait direction (see
    // ComputeDefaultTestDirection below). Checked lazily here, not at
    // construction, since prms.TestWindSpeed isn't parsed from JSON yet at
    // construction time.
    if (prms.TestWindSpeed != 0.0 && !test_mode_enabled) {
        test_mode_enabled = true;
        test_initial_magnitude = prms.TestWindSpeed;
        test_growth_rate_per_day = 0.0;
    }

    bool changed = ProcessCARRA1(t);
    // Only re-apply when the real CARRA1 frame just changed (i.e. only as
    // often as ProcessCARRA1 itself re-uploads -- e.g. hourly, NOT every
    // step). current_wind_alpha (computed by ProcessCARRA1 for the exact
    // same bracket, shared with temperature, left untouched here) then
    // smoothly blends the two endpoint values every step with zero further
    // GPU upload -- see ApplyWindTestMode for why this reproduces the
    // globally-linear magnitude(t) exactly, with no approximation error.
    if (test_mode_enabled && wind_ring.IsReady() && changed) {
        ApplyWindTestMode(t);
    }
    return changed;
}

Eigen::Vector2d WindInterpolator::ComputeDefaultTestDirection() const
{
    // Bearing across Nares Strait: 81 12'22.8"N 64 44'09.8"W -> 80 21'29.0"N 68 38'15.3"W.
    constexpr double kTestDirLat1 = 81.206333, kTestDirLon1 = -64.736056;
    constexpr double kTestDirLat2 = 80.358056, kTestDirLon2 = -68.637583;
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

void WindInterpolator::ApplyWindTestMode(double t)
{
    if (!test_dir_initialized) {
        Eigen::Vector2d dir = ComputeDefaultTestDirection();
        test_dir_x = dir.x();
        test_dir_y = dir.y();
        test_dir_initialized = true;
    }

    // Re-resolve the same [t0_real, t1_real] bracket ProcessCARRA1 just used
    // (mirrors its own upper_bound lookup) so the synthetic wind is evaluated
    // at the EXACT two timestamps current_wind_alpha already blends between.
    // Since magnitude(t) is globally linear in elapsed time, writing the true
    // magnitude at each real endpoint here means the untouched, already-
    // correct current_wind_alpha reproduces the exact value at any t in
    // between every step, with no further GPU upload needed until the next
    // real CARRA1 frame boundary.
    long long target_ts = (long long)t;
    auto it = std::upper_bound(carra1_times.begin(), carra1_times.end(), target_ts);
    int idx = (it == carra1_times.begin()) ? 0 : (int)std::distance(carra1_times.begin(), it) - 1;
    if (idx >= carra1_num_frames - 1) idx = carra1_num_frames - 2;
    if (idx < 0) idx = 0;
    long long t0_real = carra1_times[idx];
    long long t1_real = carra1_times[idx + 1];

    auto magnitude_at = [&](long long ts) {
        double elapsed = (double)ts - prms.SimulationStartTime;
        return test_initial_magnitude + (test_growth_rate_per_day / 86400.0) * std::max(0.0, elapsed);
    };
    double mag0 = magnitude_at(t0_real);
    double mag1 = magnitude_at(t1_real);
    float vx0 = (float)(test_dir_x * mag0), vy0 = (float)(test_dir_y * mag0);
    float vx1 = (float)(test_dir_x * mag1), vy1 = (float)(test_dir_y * mag1);

    // Overwrite ONLY wind vx/vy in the active slots -- wind_temp_frame_buffer
    // (surface temperature) is left exactly as ProcessCARRA1 loaded it, and
    // current_wind_alpha is left untouched (it's shared with temperature
    // blending -- see the comment on test_mode_enabled in windinterpolator.h).
    int slot0 = wind_ring.ActiveSlot(0);
    int slot1 = wind_ring.ActiveSlot(1);
    if (slot0 >= 0) {
        std::fill(wind_vx_frame_buffer[slot0].begin(), wind_vx_frame_buffer[slot0].end(), vx0);
        std::fill(wind_vy_frame_buffer[slot0].begin(), wind_vy_frame_buffer[slot0].end(), vy0);
    }
    if (slot1 >= 0) {
        std::fill(wind_vx_frame_buffer[slot1].begin(), wind_vx_frame_buffer[slot1].end(), vx1);
        std::fill(wind_vy_frame_buffer[slot1].begin(), wind_vy_frame_buffer[slot1].end(), vy1);
    }
}

const float* WindInterpolator::GetWindDataPointer(int logicalFrame, int component) const
{
    if (!wind_ring.IsReady()) return nullptr;
    int slot = wind_ring.ActiveSlot(logicalFrame);
    if (component == 0) return wind_vx_frame_buffer[slot].data();
    if (component == 1) return wind_vy_frame_buffer[slot].data();
    return nullptr;
}

const float* WindInterpolator::GetTempDataPointer(int logicalFrame) const
{
    if (!wind_ring.IsReady()) return nullptr;
    int slot = wind_ring.ActiveSlot(logicalFrame);
    return wind_temp_frame_buffer[slot].data();
}

Eigen::Vector2d WindInterpolator::TransformWind_LocalToGeographic(double u_local, double v_local, const Eigen::Vector2d& p00, const Eigen::Vector2d& p01, const Eigen::Vector2d& p10) const {
    double phi00_rad = p00.y() * (M_PI / 180.0);
    double cos_lat = std::cos(phi00_rad);
    
    // X-axis geographic direction (p00 -> p01)
    double x_dlat = (p01.y() - p00.y());
    double x_dlon = (p01.x() - p00.x()) * cos_lat;
    double x_len = std::sqrt(x_dlat * x_dlat + x_dlon * x_dlon);
    if (x_len < 1e-9) x_len = 1.0;
    
    Eigen::Vector2d eX(x_dlon / x_len, x_dlat / x_len); // (Eastward, Northward)
    
    // Y-axis geographic direction (p00 -> p10)
    double y_dlat = (p10.y() - p00.y());
    double y_dlon = (p10.x() - p00.x()) * cos_lat;
    double y_len = std::sqrt(y_dlat * y_dlat + y_dlon * y_dlon);
    if (y_len < 1e-9) y_len = 1.0;
    
    Eigen::Vector2d eY(y_dlon / y_len, y_dlat / y_len);
    
    // Combine local vector components with geographic unit vectors
    Eigen::Vector2d geo_wind = u_local * eX + v_local * eY;
    return geo_wind;
}

Eigen::Vector2d WindInterpolator::TransformWind_GeographicToGrid(const Eigen::Vector2d& geo_wind, double lat_rad, double lon_rad) const {
    Eigen::Matrix2d rot = ComputeRotationMatrix(lat_rad, lon_rad);
    return rot * geo_wind;
}
