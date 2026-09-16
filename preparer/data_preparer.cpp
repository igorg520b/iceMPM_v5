#include "data_preparer.h"
#include "poisson_disk_sampling.h"
#include "optimized_poisson_disk_sampling.h"
#include <spdlog/spdlog.h>
#include <H5Cpp.h>
#include <iostream>
#include <fstream>
#include <random>
#include <algorithm>
#include <fmt/format.h>
#include <fmt/std.h>
#include <cstring>
#include <queue>
#include <cmath>

// Helper includes
#include <png.h>


// Static helper for PNG loading
bool DataPreparer::LoadPng(const std::string& filename, int& w, int& h, int& channels, std::vector<uint8_t>& data)
{
    FILE* fp = fopen(filename.c_str(), "rb");
    if (!fp) {
        LOGR("LoadPng: Failed to open file {}", filename);
        return false;
    }

    // verify signature
    png_byte header[8];
    if (fread(header, 1, 8, fp) != 8) {
        LOGR("LoadPng: Failed to read signature from {}", filename);
        fclose(fp);
        return false;
    }
    if (png_sig_cmp(header, 0, 8)) {
        LOGR("LoadPng: File {} is not a PNG", filename);
        fclose(fp);
        return false;
    }

    png_structp png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (!png_ptr) {
        LOGR("LoadPng: png_create_read_struct failed");
        fclose(fp);
        return false;
    }

    png_infop info_ptr = png_create_info_struct(png_ptr);
    if (!info_ptr) {
        LOGR("LoadPng: png_create_info_struct failed");
        png_destroy_read_struct(&png_ptr, (png_infopp)NULL, (png_infopp)NULL);
        fclose(fp);
        return false;
    }

    if (setjmp(png_jmpbuf(png_ptr))) {
        LOGR("LoadPng: Error during init_io");
        png_destroy_read_struct(&png_ptr, &info_ptr, NULL);
        fclose(fp);
        return false;
    }

    png_init_io(png_ptr, fp);
    png_set_sig_bytes(png_ptr, 8);

    png_read_info(png_ptr, info_ptr);

    w = png_get_image_width(png_ptr, info_ptr);
    h = png_get_image_height(png_ptr, info_ptr);
    auto color_type = png_get_color_type(png_ptr, info_ptr);
    auto bit_depth = png_get_bit_depth(png_ptr, info_ptr);

    // Standardize to RGBA 8-bit
    if (bit_depth == 16)
        png_set_strip_16(png_ptr);

    if (color_type == PNG_COLOR_TYPE_PALETTE)
        png_set_palette_to_rgb(png_ptr);

    if (color_type == PNG_COLOR_TYPE_GRAY && bit_depth < 8)
        png_set_expand_gray_1_2_4_to_8(png_ptr);

    if (png_get_valid(png_ptr, info_ptr, PNG_INFO_tRNS))
        png_set_tRNS_to_alpha(png_ptr);

    // Expand gray to RGB
    if (color_type == PNG_COLOR_TYPE_GRAY || color_type == PNG_COLOR_TYPE_GRAY_ALPHA)
        png_set_gray_to_rgb(png_ptr);

    // Add alpha if missing
    if (!(color_type & PNG_COLOR_MASK_ALPHA))
         png_set_add_alpha(png_ptr, 0xff, PNG_FILLER_AFTER);

    png_read_update_info(png_ptr, info_ptr);

    // We effectively forced it to RGBA 8-bit, so channels = 4
    channels = 4;
    size_t row_bytes = png_get_rowbytes(png_ptr, info_ptr);
    
    // row_bytes should be w * 4
    if (row_bytes != w * 4) {
        LOGR("LoadPng: Unexpected row_bytes {} for width {}", row_bytes, w);
    }
    
    data.resize((size_t)h * row_bytes);

    std::vector<png_bytep> row_pointers(h);
    for (int y = 0; y < h; y++) {
        row_pointers[y] = &data[y * row_bytes];
    }

    if (setjmp(png_jmpbuf(png_ptr))) {
        LOGR("LoadPng: Error during read_image");
        png_destroy_read_struct(&png_ptr, &info_ptr, NULL);
        fclose(fp);
        return false;
    }

    png_read_image(png_ptr, row_pointers.data());

    png_destroy_read_struct(&png_ptr, &info_ptr, NULL);
    fclose(fp);
    return true;
}




namespace fs = std::filesystem;

DataPreparer::DataPreparer(HostSideData& hsd) : hsd(hsd)
{
}

std::string DataPreparer::prepare_cache_filename(int gx, int gy, int ppc)
{
    return fmt::format("{}/poisson_cache/points_{}x{}_{:d}.h5", hsd.dirs.DataCacheRoot(), gx, gy, ppc);
}

bool DataPreparer::attempt_to_fill_from_cache(int gx, int gy, int ppc, std::vector<std::array<float, 2>> &buffer)
{
    std::string cache_file = prepare_cache_filename(gx, gy, ppc);
    std::filesystem::path cache_path(cache_file);

    if (!std::filesystem::exists(cache_path)) {
        LOGR("Poisson cache file does not exist");
        return false;
    }

    // The file existing is the only condition treated as a normal "no cache
    // yet" case. From here on, anything that goes wrong (can't open it, the
    // dataset has the wrong shape, a read fails) is a real problem -- not a
    // cache miss to quietly fall back from -- so it throws rather than being
    // caught and swallowed.
    LOGR("Attempting to load Poisson points from cache: {}", cache_file);
    H5::H5File cache_hfile(cache_file, H5F_ACC_RDONLY);
    H5::DataSet cache_dataset = cache_hfile.openDataSet("coords");

    hsize_t dims[2];
    cache_dataset.getSpace().getSimpleExtentDims(dims, NULL);
    if (dims[1] != 2) {
        throw std::runtime_error(fmt::format(
            "Poisson cache file {} has invalid format (dims[1]={}, expected 2)", cache_file, dims[1]));
    }

    buffer.resize(dims[0]);
    cache_dataset.read(buffer.data(), H5::PredType::NATIVE_FLOAT);
    cache_hfile.close();

    LOGR("Loaded {} points from cache", buffer.size());
    return true;
}



void DataPreparer::generate_and_save_poisson(int gx, int gy, float points_per_cell, std::vector<std::array<float, 2>> &buffer)
{
    // Use (gy-1)/(gx-1) to ensure that isotropic scaling (by gx-1) maps exactly to [0, gy-1]
    const float dy = (float)(gy - 1) / (gx - 1);

    LOGR("Generating Poisson points with radius-based parameters (OPTIMIZED)");

    const std::array<float, 2> x_min{0, 0};
    const std::array<float, 2> x_max{1, dy};
    constexpr float magic_constant = 0.6;
    const float radius = std::sqrt(magic_constant / (points_per_cell * gx * gx));

    LOGR("Poisson parameters: grid={}x{}, dy={}, radius={}, target_ppc={}", gx, gy, dy, radius, points_per_cell);

    // MASK LAMBDA for optimization
    // Maps local point p [0, 1]x[0, dy] to global image coordinates and checks ICE mask.
    auto mask_func = [&](const std::array<float, 2>& pt) -> bool {
        // Map to integer grid coordinates relative to Ice Region
        // pt[0] is in [0, 1], pt[1] is in [0, dy]
        // gx is IceRegionWidth, gy is IceRegionHeight
        
        // Map pt[0] from [0, 1] to [0, gx-1]
        int i = (int)(pt[0] * (gx - 1) + 0.5f);
        // Map pt[1] from [0, dy] to [0, gy-1]. 
        // Since dy = (gy-1)/(gx-1), pt[1] * (gx-1) maps [0, dy] -> [0, gy-1]
        int j = (int)(pt[1] * (gx - 1) + 0.5f); 
        
        // Bounds check (local to IceRegion)
        if (i < 0 || i >= gx || j < 0 || j >= gy) return false;

        // Global check
        int img_x = i + this->IceRegionOffsetX;
        int img_y = j + this->IceRegionOffsetY;
        
        // Safety check for image bounds
        if (img_x < 0 || img_x >= m_width || img_y < 0 || img_y >= m_height) return false;
        
        uint8_t flags = m_flags[(size_t)img_y * m_width + img_x];
        bool is_ice = (flags & FLAG_ICE) && (flags & FLAG_WATER) && !(flags & FLAG_OPEN_BOUNDARY);
        return is_ice;
    };

    // Use the optimized version with Mask
    std::random_device rd;
    uint32_t poisson_seed = rd();
    LOGR("Poisson disk sampling seed: {}", poisson_seed);
    buffer = thinks::PoissonDiskSampling(radius, x_min, x_max, mask_func, 30, poisson_seed);
    LOGR("Generated {} raw Poisson points", buffer.size());

    // Calculate actual Ice Area (count of valid cells) for correct PPC calculation
    long long count_ice_cells = 0;
    for(int j=0; j<gy; ++j) {
        for(int i=0; i<gx; ++i) {
            int img_x = i + this->IceRegionOffsetX;
            int img_y = j + this->IceRegionOffsetY;
            
            if (img_x >= 0 && img_x < m_width && img_y >= 0 && img_y < m_height) {
                uint8_t flags = m_flags[(size_t)img_y * m_width + img_x];
                if ((flags & FLAG_ICE) && (flags & FLAG_WATER) && !(flags & FLAG_OPEN_BOUNDARY)) {
                    count_ice_cells++;
                }
            }
        }
    }

    // Log the actual points per cell achieved based on ACTIVE area
    double active_area_fraction = (double)count_ice_cells / (double)(gx * gy);
    float raw_ppc = 0.0f;
    if (count_ice_cells > 0) {
        raw_ppc = (float)buffer.size() / (float)count_ice_cells;
    }

    LOGR("Raw achieved ppc: {:.4f} (target {:.4f}) | IceCells: {}/{} ({:.1f}%)", 
         raw_ppc, points_per_cell, count_ice_cells, gx*gy, active_area_fraction*100.0);

    // Note: We deliberately DO NOT scale the points. 
    // Scaling shifts coordinates relative to the underlying Mask (IceRegion), causing data misalignment.
    // Small variations in PPC are handled by the ParticleArea normalization in PopulatePoints.
    
    // Log final ppc
    LOGR("grid: {}x{}; final pts {:>8}; final_ppc {:.4f}", gx, gy, buffer.size(), raw_ppc);

    // Save to cache. A failure here (disk full, permissions, etc.) throws
    // rather than being logged as a warning and continued past -- silently
    // "succeeding" would leave the user believing a cache was written when
    // it wasn't, discovered only when the next run mysteriously regenerates
    // points from scratch too.
    std::string cache_file = prepare_cache_filename(gx, gy, (int)points_per_cell);
    std::filesystem::create_directories(std::filesystem::path(cache_file).parent_path());
    H5::H5File cache_hfile(cache_file, H5F_ACC_TRUNC);

    hsize_t dims[2] = {buffer.size(), 2};
    H5::DataSpace cache_space(2, dims);

    H5::DataSet cache_dataset = cache_hfile.createDataSet("coords", H5::PredType::NATIVE_FLOAT, cache_space);
    cache_dataset.write(buffer.data(), H5::PredType::NATIVE_FLOAT);

    H5::DataSpace att_space(H5S_SCALAR);
    int gx_attr = gx, gy_attr = gy, ppc_attr = (int)points_per_cell;
    cache_dataset.createAttribute("GridXTotal", H5::PredType::NATIVE_INT, att_space).write(H5::PredType::NATIVE_INT, &gx_attr);
    cache_dataset.createAttribute("GridYTotal", H5::PredType::NATIVE_INT, att_space).write(H5::PredType::NATIVE_INT, &gy_attr);
    cache_dataset.createAttribute("PointsPerCell", H5::PredType::NATIVE_INT, att_space).write(H5::PredType::NATIVE_INT, &ppc_attr);

    cache_hfile.close();
    LOGR("Saved Poisson points to cache: {}", cache_file);
}



bool DataPreparer::ProcessMaskLayer(const std::string& filename, uint8_t flag, bool invert, int threshold)
{
    if (filename.empty()) return false;

    int w, h, c;
    std::vector<uint8_t> raw;
    if (!LoadPng(filename, w, h, c, raw)) {
        throw std::runtime_error("Failed to load mask: " + filename);
    }

    if (w != m_width || h != m_height) {
        LOGR("Mask {} is {}x{}, resizing to {}x{} (working resolution)",
             filename, w, h, m_width, m_height);
        std::vector<uint8_t> resized;
        ResizeMaskNearest(raw, w, h, resized, m_width, m_height);
        raw = std::move(resized);
        w = m_width;
        h = m_height;
    }

    // raw is RGBA (4 bytes per pixel)
    for (int y = 0; y < h; ++y) {
        // LoadPng returns top-left origin, same as stbi.
        // We flip Y to match simulation coordinates (bottom-left origin).
        int src_y = h - 1 - y;
        for (int x = 0; x < w; ++x) {
            size_t idx = ((size_t)src_y * w + x) * 4;
            // Use Red channel (index 0) as the value
            uint8_t val = raw[idx];
            bool condition = (val < threshold); // Default: black (<128) matches condition
            if (invert) condition = !condition;

            if (condition) {
                m_flags[(size_t)y * w + x] |= flag;
            }
        }
    }

    LOGR("Loaded mask {} -> flag {}", filename, flag);
    return true;
}

void DataPreparer::ResizeMaskNearest(const std::vector<uint8_t>& src, int srcW, int srcH,
                                      std::vector<uint8_t>& dst, int dstW, int dstH)
{
    dst.resize((size_t)dstW * dstH * 4);
    const double scaleX = (double)srcW / dstW;
    const double scaleY = (double)srcH / dstH;

    #pragma omp parallel for
    for (int y = 0; y < dstH; ++y) {
        int sy = std::min(srcH - 1, (int)(y * scaleY));
        for (int x = 0; x < dstW; ++x) {
            int sx = std::min(srcW - 1, (int)(x * scaleX));
            size_t srcIdx = ((size_t)sy * srcW + sx) * 4;
            size_t dstIdx = ((size_t)y * dstW + x) * 4;
            dst[dstIdx + 0] = src[srcIdx + 0];
            dst[dstIdx + 1] = src[srcIdx + 1];
            dst[dstIdx + 2] = src[srcIdx + 2];
            dst[dstIdx + 3] = src[srcIdx + 3];
        }
    }
}

bool DataPreparer::ProcessFootprintMask(const std::string& filename)
{
    if (filename.empty()) return false;

    int w, h, c;
    std::vector<uint8_t> raw;
    if (!LoadPng(filename, w, h, c, raw)) {
        throw std::runtime_error("Failed to load footprint mask: " + filename);
    }

    if (w != m_width || h != m_height) {
        LOGR("Footprint mask {} is {}x{}, resizing to {}x{} (working resolution)",
             filename, w, h, m_width, m_height);
        std::vector<uint8_t> resized;
        ResizeMaskNearest(raw, w, h, resized, m_width, m_height);
        raw = std::move(resized);
        w = m_width;
        h = m_height;
    }

    // raw is RGBA (4 bytes per pixel); same top-left -> bottom-left Y-flip
    // convention as ProcessMaskLayer. Black (<128) = open boundary.
    for (int y = 0; y < h; ++y) {
        int src_y = h - 1 - y;
        for (int x = 0; x < w; ++x) {
            size_t idx = ((size_t)src_y * w + x) * 4;
            uint8_t val = raw[idx];
            if (val < 128) {
                m_flags[(size_t)y * w + x] |= FLAG_OPEN_BOUNDARY;
            }
        }
    }

    LOGR("Loaded footprint mask {} -> FLAG_OPEN_BOUNDARY", filename);
    return true;
}

void DataPreparer::PrepareGridAndPoints(std::string fileNameLandMask, std::string fileNameColor,
                                       std::string fileNameIceMask, std::string fileNameCrushedMask,
                                       std::string fileNameCrackedMask,
                                       std::string projectDirectory, int pointsPerCell,
                                       double thicknessFrom, double thicknessTo,
                                       double probCracked, double stdDevThickness,
                                       std::string fileNameThicknessMask,
                                       bool allocate_dense_grid,
                                       bool reload_after_save,
                                       bool generatePoints,
                                       std::string fileNameFootprintMask,
                                       std::string fileNameAnalysisMask,
                                       std::string fileNameSnapshotToLoad)
{
    LOGR("PrepareGridAndPoints: starting optimization run");

    hsd.data_directory = projectDirectory;

    // 1. Load Color Image (Master Dimensions)
    int c;
    std::vector<uint8_t> color_raw;
    if (!LoadPng(fileNameColor, m_width, m_height, c, color_raw)) {
        throw std::runtime_error("Failed to load ImageColor: " + fileNameColor);
    }
    
    // Store in m_color (RGB) and flip vertically
    m_color.resize((size_t)m_width * m_height * 3);
    
    for (int y = 0; y < m_height; ++y) {
        int src_y = m_height - 1 - y; // Flip Y
        for (int x = 0; x < m_width; ++x) {
            size_t src_idx = ((size_t)src_y * m_width + x) * 4; // RGBA
            size_t dst_idx = ((size_t)y * m_width + x) * 3;     // RGB
            m_color[dst_idx + 0] = color_raw[src_idx + 0];
            m_color[dst_idx + 1] = color_raw[src_idx + 1];
            m_color[dst_idx + 2] = color_raw[src_idx + 2];
        }
    }
    
    // Clear raw buffer
    color_raw.clear(); color_raw.shrink_to_fit();

    LOGR("[MEMORY] DataPreparer: m_color allocated: {:.3f} MB", (double)m_color.size() * sizeof(uint8_t) / 1.0e6);

    // Initialize buffers
    m_flags.assign((size_t)m_width * m_height, 0); // All zero initially
    m_thickness.resize((size_t)m_width * m_height);
    LOGR("[MEMORY] DataPreparer: m_flags allocated: {:.3f} MB", (double)m_flags.size() * sizeof(uint8_t) / 1.0e6);
    LOGR("[MEMORY] DataPreparer: m_thickness allocated: {:.3f} MB", (double)m_thickness.size() * sizeof(uint8_t) / 1.0e6);

    // 2. Load Masks and merge into m_flags
    // Landmask: Black (<128) = Water (Modeled Area), White = Land
    // So we set FLAG_WATER if pixel < 128
    if (!fileNameLandMask.empty()) {
        ProcessMaskLayer(fileNameLandMask, FLAG_WATER, false); 
    } else {
        // No landmask -> All water
        std::fill(m_flags.begin(), m_flags.end(), FLAG_WATER);
        LOGR("No ImageLandMask -> Interpreting entire domain as WATER");
    }

    // IceMask: White (>=128) = Ice, Black = No Ice
    // Default ProcessMask logic is "val < threshold" (Black).
    // condition = (val < 128).
    ProcessMaskLayer(fileNameIceMask, FLAG_ICE, true); 

    // CrushedMask: Black (<128) = Crushed.
    if (!fileNameCrushedMask.empty()) {
        ProcessMaskLayer(fileNameCrushedMask, FLAG_CRUSHED, false);
    }

    // FootprintMask: Black (<128) = Open Boundary. Overlaid with priority
    // over FLAG_WATER when landmask_buffer is built (see PrepareGrid).
    if (!fileNameFootprintMask.empty()) {
        ProcessFootprintMask(fileNameFootprintMask);
    }

    // CrackedMask: Black (<128) = Cracked.
    if (!fileNameCrackedMask.empty()) {
        ProcessMaskLayer(fileNameCrackedMask, FLAG_CRACKED, false);
    }

    // AnalysisRegionMask: White (>=128) = included in region diagnostics, Black = excluded.
    if (!fileNameAnalysisMask.empty()) {
        ProcessMaskLayer(fileNameAnalysisMask, FLAG_ANALYSIS_REGION, true);
    }

    // 3. Load Thickness from the image mask (or random, if none given).
    if (!fileNameThicknessMask.empty()) {
         int tw, th, tc;
         std::vector<uint8_t> traw;
         if (!LoadPng(fileNameThicknessMask, tw, th, tc, traw)) 
              throw std::runtime_error("Failed to load ThicknessMask");
         
         if (tw != m_width || th != m_height) {
             LOGR("ThicknessMask {} is {}x{}, resizing to {}x{} (working resolution)",
                  fileNameThicknessMask, tw, th, m_width, m_height);
             std::vector<uint8_t> resized;
             ResizeMaskNearest(traw, tw, th, resized, m_width, m_height);
             traw = std::move(resized);
             tw = m_width;
             th = m_height;
         }

         // Flip and store (traw is RGBA)
         for(int y=0; y<th; ++y) {
             int src_y = th - 1 - y;
             for (int x=0; x<tw; ++x) {
                  m_thickness[(size_t)y*tw + x] = traw[(src_y*(size_t)tw + x)*4]; // Red channel
             }
         }
    } else {
        // Generate random thickness
        std::random_device rd;
        uint32_t thickness_seed = rd();
        std::mt19937 gen(thickness_seed);
        std::normal_distribution<double> dist(127.5, 255.0 / 4.0);
        for (auto& val : m_thickness) {
            val = (uint8_t)std::clamp(dist(gen), 0.0, 255.0);
        }
        LOGR("Generated random thickness field (seed {})", thickness_seed);
    }

    DetermineExtents();

    // Recommended stable time step (CFL-like elastic wave-speed estimate),
    // using the thinnest ice actually present in the initialized thickness
    // field (m_thickness, just populated above) as the worst case -- not
    // the flat ThicknessFrom config value, which may not be reached
    // anywhere in this particular thickness mask.
    // Not written back into the config file -- see simulation.json's own
    // InitialTimeStep, which the user sets explicitly.
    {
        uint8_t minThicknessByte = *std::min_element(m_thickness.begin(), m_thickness.end());
        double minThickness = thicknessFrom + ((double)minThicknessByte / 255.0) * (thicknessTo - thicknessFrom);

        const double recommendedTimeStep = 0.8 * hsd.prms.cellsize *
            std::sqrt(hsd.prms.IceDensity * minThickness / hsd.prms.YoungsModulus);
        LOGR("Minimum ice thickness in field: {:.6g} m", minThickness);
        LOGR("Recommended time step (0.8 * cellsize * sqrt(density * min_thickness / YoungsModulus)): {:.6g}",
             recommendedTimeStep);
    }

    // 5. Populate Points (Using m_flags, m_thickness, and m_color)
    if (!fileNameSnapshotToLoad.empty()) {
        // Overrides generatePoints entirely -- load a previously generated
        // point set instead of sampling a new one, so callers (e.g. thermal
        // spin-up) can run repeatably against fixed geometry. GridXTotal/
        // GridYTotal are already set by DetermineExtents() above, so
        // ReadPointsFromSnapshot's own mismatch check (see host_side_data.cpp)
        // verifies the loaded points actually belong to this same grid.
        if (generatePoints)
            LOGR("PrepareGridAndPoints: fileNameSnapshotToLoad is set -- ignoring generatePoints=true");
        LOGR("PrepareGridAndPoints: loading points from snapshot {} instead of generating", fileNameSnapshotToLoad);
        hsd.ReadPointsFromSnapshot(fileNameSnapshotToLoad);
        LogTotalIceVolume();
    } else if (generatePoints) {
        DetermineIceRegion();
        PopulatePoints_RAM_Optimized(pointsPerCell, thicknessFrom, thicknessTo, probCracked, stdDevThickness);
    } else {
        LOGR("PrepareGridAndPoints: generatePoints=false -- skipping point generation (grid sizing/rendering only)");
    }

    // 4. Process Grid (Using m_flags and m_color)
    PrepareGrid(projectDirectory, allocate_dense_grid);

    // Cleanup member buffers to release memory
    m_color.clear(); m_color.shrink_to_fit();
    m_flags.clear(); m_flags.shrink_to_fit();
    m_thickness.clear(); m_thickness.shrink_to_fit();

    LOGR("PrepareGridAndPoints: completed");
}


void DataPreparer::InitThermalState()
{
    int firstIdx = hsd.windInterp.GetFrameRangeForTimeSpan(
        hsd.prms.PreSimulationStartTime, hsd.prms.SimulationStartTime).first;

    const int gy = hsd.prms.GridYTotal;
    std::vector<float> startFrame;
    hsd.windInterp.GetTemperatureFrameExact(firstIdx, startFrame);

    const uint64_t numPoints = hsd.hssoa.size;
    #pragma omp parallel for
    for (uint64_t k = 0; k < numPoints; ++k) {
        SOAIterator it = hsd.hssoa.begin() + k;
        long long cellIdx = it->getCellIndex(gy);
        it->setValue(SimParams::PtArrIdx::idx_Ts_old, (double)startFrame[cellIdx] - 273.15);
    }
    LOGR("InitThermalState: {} points, Ts_old seeded from CARRA1 frame {}", numPoints, firstIdx);
}

void DataPreparer::ThermalSpinUpStep(int frameIdx)
{
    const double dt = (double)(hsd.windInterp.GetCARRA1Timestamp(frameIdx) -
                                hsd.windInterp.GetCARRA1Timestamp(frameIdx - 1));
    const int gy = hsd.prms.GridYTotal;

    hsd.windInterp.GetTemperatureFrameExact(frameIdx, m_spinup_temp_scratch);

    static constexpr SimParams::PtArrIdx modeIdx[3] = {
        SimParams::PtArrIdx::idx_A1, SimParams::PtArrIdx::idx_A2, SimParams::PtArrIdx::idx_A3};

    const uint64_t numPoints = hsd.hssoa.size;
    double temp_sum = 0.0;
    #pragma omp parallel for reduction(+:temp_sum)
    for (uint64_t k = 0; k < numPoints; ++k) {
        SOAIterator it = hsd.hssoa.begin() + k;
        long long cellIdx = it->getCellIndex(gy);
        double H = it->getValue(SimParams::PtArrIdx::idx_thickness);
        double Ts_old = it->getValue(SimParams::PtArrIdx::idx_Ts_old);
        double Ts_new = (double)m_spinup_temp_scratch[cellIdx] - 273.15;
        double r = (Ts_new - Ts_old) / dt;

        double A_new[3];
        for (int n = 1; n <= 3; ++n) {
            double tau_n = (H * H) / (SimParams::kappa_ice * SimParams::pi * SimParams::pi * (double)(n * n));
            double b_n = 2.0 / ((double)n * SimParams::pi);
            double A_old = it->getValue(modeIdx[n - 1]);
            A_new[n - 1] = (A_old + r * b_n * tau_n) * std::exp(-dt / tau_n) - r * b_n * tau_n;
            it->setValue(modeIdx[n - 1], A_new[n - 1]);
        }
        it->setValue(SimParams::PtArrIdx::idx_Ts_old, Ts_new);

        // Exact thickness-average of the sinusoidal vertical profile -- the
        // A2 (even-harmonic) term integrates to zero over the full thickness,
        // only the odd harmonics (A1, A3) contribute.
        temp_sum += (Ts_new + SimParams::T_bottom) * 0.5
                  + (2.0 / SimParams::pi) * A_new[0] + (2.0 / (3.0 * SimParams::pi)) * A_new[2];
    }

    hsd.AppendAverageIceTemperature(hsd.windInterp.GetCARRA1Timestamp(frameIdx),
        numPoints ? temp_sum / numPoints : 0.0, (unsigned)numPoints, "spinup");
}

namespace {
// CPU mirror of simulation/kernels.cu's reconstruct_T / brine_volume_frankenstein_garner /
// flexural_strength_timco_obrien -- keep in sync if the strength formula changes there.
double reconstruct_T_cpu(double z, double H, double Ts, double A1, double A2, double A3)
{
    return Ts + (SimParams::T_bottom - Ts) * (z / H)
         + A1*std::sin(SimParams::pi*z/H) + A2*std::sin(2.0*SimParams::pi*z/H) + A3*std::sin(3.0*SimParams::pi*z/H);
}

double brine_volume_frankenstein_garner_cpu(double T, double S)
{
    // Macaulay bracket <-T> = max(-T, eps), NOT |T| -- see kernels.cu's
    // brine_volume_frankenstein_garner for why.
    double Tabs = std::max(-T, 1e-6);
    return std::min(1.0, std::max(0.0, S * (49.185/Tabs + 0.532) * 1e-3));
}

double flexural_strength_timco_obrien_cpu(double vb)
{
    return 1.76e6 * std::exp(-5.88 * std::sqrt(vb));
}
} // namespace

void DataPreparer::ComputeIceStrengthCPU()
{
    const uint64_t numPoints = hsd.hssoa.size;
    constexpr int N_DEPTH = 8;

    #pragma omp parallel for
    for (uint64_t k = 0; k < numPoints; ++k) {
        SOAIterator it = hsd.hssoa.begin() + k;
        double H = it->getValue(SimParams::PtArrIdx::idx_thickness);
        double Ts = it->getValue(SimParams::PtArrIdx::idx_Ts_old);
        double A1 = it->getValue(SimParams::PtArrIdx::idx_A1);
        double A2 = it->getValue(SimParams::PtArrIdx::idx_A2);
        double A3 = it->getValue(SimParams::PtArrIdx::idx_A3);

        double sigma_sum = 0.0;
        for (int d = 0; d < N_DEPTH; ++d) {
            double z = (d + 0.5) * H / N_DEPTH;
            double T = reconstruct_T_cpu(z, H, Ts, A1, A2, A3);
            double vb = brine_volume_frankenstein_garner_cpu(T, hsd.prms.IceSalinity);
            sigma_sum += flexural_strength_timco_obrien_cpu(vb);
        }
        it->setValue(SimParams::PtArrIdx::idx_xi, sigma_sum / N_DEPTH);
    }
}

void DataPreparer::ThermalSpinUpSave()
{
    // Always (re)compute idx_xi from the final thermal state before saving --
    // previously this only happened when "Render Spin-Up Frames" was on (for
    // frame-capture purposes), so idx_xi silently stayed at its
    // zero-initialized placeholder in the common case where that option was
    // off. HostSideData::SaveSnapshot serializes the whole point SOA
    // generically, so idx_xi rides along automatically once populated.
    ComputeIceStrengthCPU();

    double xi_sum = 0.0;
    const uint64_t numPoints = hsd.hssoa.size;
    #pragma omp parallel for reduction(+:xi_sum)
    for (uint64_t k = 0; k < numPoints; ++k)
        xi_sum += (hsd.hssoa.begin() + k)->getValue(SimParams::PtArrIdx::idx_xi);
    LOGR("ThermalSpinUpSave: average idx_xi (flexural strength) over {} points = {:.6g} Pa",
         numPoints, numPoints ? xi_sum / numPoints : 0.0);

    std::string snapDir = hsd.dirs.SnapshotsDirectory();
    hsd.SaveSnapshot(0, 0.0, true, snapDir);
    LOGR("ThermalSpinUpSave: saved to {}", snapDir);

    // Debug: print (A1, A2, A3, Ts_old, thickness, lat, lon, pixel_x, pixel_y) for a random
    // sample of points, Mathematica list format. pixel_x/pixel_y are in the same top-down
    // convention as the original ImageColor PNG (row 0 = top), for overlaying on that image.
    const int sampleSize = (int)std::min<uint64_t>(10, numPoints);
    std::mt19937 rng(std::random_device{}());
    std::uniform_int_distribution<uint64_t> dist(0, numPoints - 1);

    std::cout << "{";
    for (int s = 0; s < sampleSize; ++s) {
        SOAIterator it = hsd.hssoa.begin() + dist(rng);
        double A1 = it->getValue(SimParams::PtArrIdx::idx_A1);
        double A2 = it->getValue(SimParams::PtArrIdx::idx_A2);
        double A3 = it->getValue(SimParams::PtArrIdx::idx_A3);
        double Ts = it->getValue(SimParams::PtArrIdx::idx_Ts_old);
        double H = it->getValue(SimParams::PtArrIdx::idx_thickness);

        auto [i_global, j_global] = it->getCellXY();
        int img_x = (int)i_global + hsd.prms.ModeledRegionOffsetX;
        int img_y = (int)j_global + hsd.prms.ModeledRegionOffsetY;
        int raw_y = hsd.prms.InitializationImageSizeY - 1 - img_y;
        Projection::LatLon ll = hsd.prms.proj.ProjectPixel(img_x, raw_y);

        std::cout << "{" << A1 << "," << A2 << "," << A3 << "," << Ts << "," << H << ","
                   << ll.lat_deg << "," << ll.lon_deg << "," << img_x << "," << raw_y << "}";
        if (s + 1 < sampleSize) std::cout << ",";
    }
    std::cout << "}" << std::endl;
}

void DataPreparer::GenerateTempWindCache(const std::string& tempCacheFile,
                                          const std::string& windCacheFile,
                                          const std::function<bool(int,int)>& progressFn)
{
    const int gx = hsd.prms.GridXTotal;
    const int gy = hsd.prms.GridYTotal;
    const int numFrames = hsd.windInterp.GetCARRA1NumFrames();

    // Create the extendible cache datasets up front (0 frames) in both
    // files, then extend and write one frame at a time below -- an aborted
    // or failed run still leaves everything written so far as a valid,
    // usable partial cache in each file.
    H5::H5File tempHfile(tempCacheFile, H5F_ACC_TRUNC);
    H5::H5File windHfile(windCacheFile, H5F_ACC_TRUNC);

    hsize_t initDims[3] = {0, (hsize_t)gx, (hsize_t)gy};
    hsize_t maxDims[3] = {H5S_UNLIMITED, (hsize_t)gx, (hsize_t)gy};
    H5::DSetCreatPropList props;
    hsize_t chunk[3] = {1, (hsize_t)gx, (hsize_t)gy};
    props.setChunk(3, chunk);
    props.setDeflate(1);

    H5::DataSpace tempSpace(3, initDims, maxDims);
    H5::DataSet tempDs = tempHfile.createDataSet("temperature", H5::PredType::NATIVE_FLOAT, tempSpace, props);
    H5::DataSpace vxSpace(3, initDims, maxDims);
    H5::DataSet vxDs = windHfile.createDataSet("vx", H5::PredType::NATIVE_FLOAT, vxSpace, props);
    H5::DataSpace vySpace(3, initDims, maxDims);
    H5::DataSet vyDs = windHfile.createDataSet("vy", H5::PredType::NATIVE_FLOAT, vySpace, props);

    H5::DataSpace attSpace(H5S_SCALAR);
    int gxAttr = gx, gyAttr = gy;
    tempDs.createAttribute("GridXTotal", H5::PredType::NATIVE_INT, attSpace).write(H5::PredType::NATIVE_INT, &gxAttr);
    tempDs.createAttribute("GridYTotal", H5::PredType::NATIVE_INT, attSpace).write(H5::PredType::NATIVE_INT, &gyAttr);
    vxDs.createAttribute("GridXTotal", H5::PredType::NATIVE_INT, attSpace).write(H5::PredType::NATIVE_INT, &gxAttr);
    vxDs.createAttribute("GridYTotal", H5::PredType::NATIVE_INT, attSpace).write(H5::PredType::NATIVE_INT, &gyAttr);

    hsize_t timeInitDims[1] = {0};
    hsize_t timeMaxDims[1] = {H5S_UNLIMITED};
    H5::DSetCreatPropList timeProps;
    hsize_t timeChunk[1] = {256};
    timeProps.setChunk(1, timeChunk);
    H5::DataSpace tempTimeSpace(1, timeInitDims, timeMaxDims);
    H5::DataSet tempTimeDs = tempHfile.createDataSet("timestamps", H5::PredType::NATIVE_INT64, tempTimeSpace, timeProps);
    H5::DataSpace windTimeSpace(1, timeInitDims, timeMaxDims);
    H5::DataSet windTimeDs = windHfile.createDataSet("timestamps", H5::PredType::NATIVE_INT64, windTimeSpace, timeProps);

    std::vector<float> scratch_vx, scratch_vy, scratch_temp;
    int framesWritten = 0;
    bool aborted = false;

    auto writeComponent = [&](H5::DataSet& ds, const std::vector<float>& data, hsize_t newFrameCount) {
        hsize_t newDims[3] = {newFrameCount, (hsize_t)gx, (hsize_t)gy};
        ds.extend(newDims);
        H5::DataSpace filespace = ds.getSpace();
        hsize_t offset[3] = {(hsize_t)framesWritten, 0, 0};
        hsize_t count[3] = {1, (hsize_t)gx, (hsize_t)gy};
        filespace.selectHyperslab(H5S_SELECT_SET, count, offset);
        hsize_t memDims[2] = {(hsize_t)gx, (hsize_t)gy};
        H5::DataSpace memspace(2, memDims);
        ds.write(data.data(), H5::PredType::NATIVE_FLOAT, memspace, filespace);
    };
    auto writeTimestamp = [&](H5::DataSet& ds, long long ts, hsize_t newFrameCount) {
        hsize_t newDims[1] = {newFrameCount};
        ds.extend(newDims);
        H5::DataSpace filespace = ds.getSpace();
        hsize_t offset[1] = {(hsize_t)framesWritten};
        hsize_t count[1] = {1};
        filespace.selectHyperslab(H5S_SELECT_SET, count, offset);
        H5::DataSpace memspace(1, count);
        int64_t ts64 = ts;
        ds.write(&ts64, H5::PredType::NATIVE_INT64, memspace, filespace);
    };

    for (int frameIdx = 0; frameIdx < numFrames; ++frameIdx) {
        if (progressFn && !progressFn(frameIdx, numFrames)) { aborted = true; break; }

        long long ts;
        try {
            // One combined render call per frame: does the shared grid<->CARRA1
            // bilinear-quad solve once (instead of twice) and reads u10/v10/skt
            // back-to-back in a single netCDF pass (instead of two separate
            // full end-to-end passes over the file) -- see RenderFrameFields.
            hsd.windInterp.RenderTempAndWindFrameUncached(frameIdx, scratch_vx, scratch_vy, scratch_temp);
            ts = hsd.windInterp.GetCARRA1Timestamp(frameIdx);
        } catch (const std::exception& e) {
            throw std::runtime_error(fmt::format(
                "Temperature+wind cache generation failed at frame {}/{} ({} frame(s) already written to {} / {}): {}",
                frameIdx, numFrames, framesWritten, tempCacheFile, windCacheFile, e.what()));
        }

        const hsize_t newFrameCount = (hsize_t)framesWritten + 1;
        writeComponent(tempDs, scratch_temp, newFrameCount);
        writeComponent(vxDs, scratch_vx, newFrameCount);
        writeComponent(vyDs, scratch_vy, newFrameCount);
        writeTimestamp(tempTimeDs, ts, newFrameCount);
        writeTimestamp(windTimeDs, ts, newFrameCount);

        framesWritten++;
    }

    tempHfile.close();
    windHfile.close();

    // Whatever we just wrote (whole, partial, or aborted) supersedes any
    // "no cache" or stale-frame-count result this session may have already
    // cached from earlier in the run.
    hsd.windInterp.InvalidateTemperatureCache();
    hsd.windInterp.InvalidateWindCache();

    LOGR("GenerateTempWindCache: {} {}/{} frames to {} and {}",
         aborted ? "aborted, wrote" : "wrote", framesWritten, numFrames, tempCacheFile, windCacheFile);
}


void DataPreparer::DetermineExtents()
{
    hsd.prms.InitializationImageSizeX = m_width;
    hsd.prms.InitializationImageSizeY = m_height;

    // Find cropped area (bounding box of WATER pixels). Deliberately reads
    // only FLAG_WATER and FLAG_OPEN_BOUNDARY -- never FLAG_ICE or any other
    // flag -- so the simulation domain itself can never depend on the ice
    // mask, even though m_flags already holds FLAG_ICE bits by the time
    // this runs (see DetermineIceRegion() for the separate, ice-aware
    // bounding box, computed later, only once point generation needs it).
    int xmin = m_width, xmax = -1;
    int ymin = m_height, ymax = -1;

    bool found_water = false;
    for (int j = 0; j < m_height; j++) {
        for (int i = 0; i < m_width; i++) {
            uint8_t flags = m_flags[(size_t)j * m_width + i];
            // Open-boundary pixels (footprint mask -- where the map
            // projection is undefined/outside the region of interest) never
            // hold points (see the generation filter below) and should not
            // widen the crop -- otherwise the rectangular bbox degenerates
            // to the full image whenever open boundary reaches the edges.
            bool is_open_boundary = flags & FLAG_OPEN_BOUNDARY;
            if ((flags & FLAG_WATER) && !is_open_boundary) {
                xmin = std::min(xmin, i);
                xmax = std::max(xmax, i);
                ymin = std::min(ymin, j);
                ymax = std::max(ymax, j);
                found_water = true;
            }
        }
    }

    if (!found_water) {
        throw std::runtime_error("No modeled area (WATER) found in landmask!");
    }

    // Pad by ±2 cells and clamp
    xmin = std::max(0, xmin - 2);
    xmax = std::min(m_width - 1, xmax + 2);
    ymin = std::max(0, ymin - 2);
    ymax = std::min(m_height - 1, ymax + 2);

    hsd.prms.ModeledRegionOffsetX = xmin;
    hsd.prms.ModeledRegionOffsetY = ymin;
    hsd.prms.GridXTotal = xmax - xmin + 1;
    hsd.prms.GridYTotal = ymax - ymin + 1;

    LOGR("Simulation grid: GridXTotal={}, GridYTotal={}, ModeledRegionOffsetX={}, ModeledRegionOffsetY={}",
         hsd.prms.GridXTotal, hsd.prms.GridYTotal, hsd.prms.ModeledRegionOffsetX, hsd.prms.ModeledRegionOffsetY);

    // Calculate physical parameters -- cellsize (meters/pixel) comes
    // directly from the projection's affine transform, not a separately
    // specified physical dimension. Callers must have already populated
    // hsd.prms.proj (e.g. via hsd.prms.ParseFile(...)) before this runs.
    hsd.prms.cellsize = hsd.prms.proj.MetersPerPixel();
    if (hsd.prms.cellsize <= 0.0) {
        throw std::runtime_error("DetermineExtents: projection is not configured (cellsize/MetersPerPixel is <= 0) -- "
                                  "was hsd.prms.ParseFile() called with PROJ_* fields before PrepareGridAndPoints?");
    }
    hsd.prms.cellsize_inv = 1.0 / hsd.prms.cellsize;
}


void DataPreparer::DetermineIceRegion()
{
    // Bounding box of ICE pixels, only counted where they're also genuinely
    // WATER and not open-boundary -- a pixel where ImageIceMask and
    // ImageLandMask disagree (ice-flagged but not water-flagged, e.g. from
    // the two masks not aligning exactly at a coastline) is not valid ice
    // and must not widen this box. This intentionally mirrors
    // DetermineExtents()'s own FLAG_WATER && !FLAG_OPEN_BOUNDARY condition,
    // with FLAG_ICE added on top, rather than substituted for it.
    int xmin_ice = m_width, xmax_ice = -1;
    int ymin_ice = m_height, ymax_ice = -1;
    bool found_ice = false;

    for (int j = 0; j < m_height; j++) {
        for (int i = 0; i < m_width; i++) {
            uint8_t flags = m_flags[(size_t)j * m_width + i];
            if ((flags & FLAG_ICE) && (flags & FLAG_WATER) && !(flags & FLAG_OPEN_BOUNDARY)) {
                xmin_ice = std::min(xmin_ice, i);
                xmax_ice = std::max(xmax_ice, i);
                ymin_ice = std::min(ymin_ice, j);
                ymax_ice = std::max(ymax_ice, j);
                found_ice = true;
            }
        }
    }

    if (found_ice) {
        // Pad by ±2 cells and clamp, same as DetermineExtents()'s water box.
        xmin_ice = std::max(0, xmin_ice - 2);
        xmax_ice = std::min(m_width - 1, xmax_ice + 2);
        ymin_ice = std::max(0, ymin_ice - 2);
        ymax_ice = std::min(m_height - 1, ymax_ice + 2);

        IceRegionOffsetX = xmin_ice;
        IceRegionOffsetY = ymin_ice;
        IceRegionWidth = xmax_ice - xmin_ice + 1;
        IceRegionHeight = ymax_ice - ymin_ice + 1;
        LOGR("Ice Region (point-generation scan area only): offset({}, {}), size {}x{}", xmin_ice, ymin_ice, IceRegionWidth, IceRegionHeight);
    } else {
        // No ice found? (Shouldn't happen if simulation expects ice, but handle gracefully)
        LOGR("No ICE found!");
        IceRegionOffsetX = 0;
        IceRegionOffsetY = 0;
        IceRegionWidth = 0;
        IceRegionHeight = 0;
    }
}


void DataPreparer::PrepareGrid(std::string projectDirectory, bool allocate_dense_grid)
{
    LOGR("PrepareGrid: starting");
    // Allocate grid arrays
    hsd.AllocateGridArrays(allocate_dense_grid);

    // (4) Build landmask_buffer and analysis_mask_buffer
    LOGR("PrepareGrid: building landmask_buffer...");
    for (int i = 0; i < hsd.prms.GridXTotal; i++) {
        for (int j = 0; j < hsd.prms.GridYTotal; j++) {
            int img_x = i + hsd.prms.ModeledRegionOffsetX;
            int img_y = j + hsd.prms.ModeledRegionOffsetY;

            uint8_t flags = m_flags[(size_t)img_y * m_width + img_x];
            uint8_t status;
            if (flags & FLAG_OPEN_BOUNDARY) {
                // Open boundary takes priority over any other landmask value.
                status = SimParams::OpenBoundaryIndicator;
            } else {
                bool is_water = (flags & FLAG_WATER);
                status = is_water ? SimParams::ModelledAreaIndicator : 0;
            }

            size_t idx = j + (size_t)i * hsd.prms.GridYTotal;
            hsd.landmask_buffer[idx] = status;
            hsd.analysis_mask_buffer[idx] = (flags & FLAG_ANALYSIS_REGION) ? 1 : 0;
        }
    }

    // (5) Store original colors (copy from m_color)
    LOGR("PrepareGrid: copying color buffer...");
    hsd.original_image_colors_rgb = m_color; 
    // This is the COPY the user might be worried about, but HSD needs it for visualization.
    // m_color will be freed after this whole function ends.

    // (6) Fill water areas with blue color for visualization in HSD
    LOGR("PrepareGrid: filling modelled area with blue...");
    hsd.FillModelledAreaWithBlueColor();

    // (7) Save grid HDF5 file
    LOGR("PrepareGrid: saving grid.h5...");
    std::string gridFilePath = projectDirectory + "/grid.h5";
    H5::H5File file(gridFilePath, H5F_ACC_TRUNC);

    // Save landmask
    hsize_t landmask_dims[2] = {static_cast<hsize_t>(hsd.prms.GridXTotal), static_cast<hsize_t>(hsd.prms.GridYTotal)};
    H5::DataSpace landmask_space(2, landmask_dims);
    H5::DSetCreatPropList landmask_props;
    hsize_t chunks[2] = {std::min<hsize_t>(hsd.prms.GridXTotal, 64), std::min<hsize_t>(hsd.prms.GridYTotal, 64)};
    landmask_props.setChunk(2, chunks);
    landmask_props.setDeflate(1);
    file.createDataSet("landmask", H5::PredType::NATIVE_UINT8, landmask_space, landmask_props)
        .write(hsd.landmask_buffer.data(), H5::PredType::NATIVE_UINT8);

    // Save analysis_mask (region for post-simulation diagnostics -- see visualizer)
    H5::DSetCreatPropList analysis_mask_props;
    analysis_mask_props.setChunk(2, chunks);
    analysis_mask_props.setDeflate(1);
    file.createDataSet("analysis_mask", H5::PredType::NATIVE_UINT8, landmask_space, analysis_mask_props)
        .write(hsd.analysis_mask_buffer.data(), H5::PredType::NATIVE_UINT8);

    // Save color
    hsize_t color_dims[3] = {static_cast<hsize_t>(m_height), static_cast<hsize_t>(m_width), 3};
    H5::DataSpace color_space(3, color_dims);
    H5::DSetCreatPropList color_props;
    hsize_t cchunks[3] = {std::min<hsize_t>(m_height, 64), std::min<hsize_t>(m_width, 64), 3};
    color_props.setChunk(3, cchunks);
    color_props.setDeflate(6);
    file.createDataSet("color_grid", H5::PredType::NATIVE_UINT8, color_space, color_props)
        .write(hsd.original_image_colors_rgb.data(), H5::PredType::NATIVE_UINT8);

    // Attributes
    int gx = hsd.prms.GridXTotal, gy = hsd.prms.GridYTotal;
    int ox = hsd.prms.ModeledRegionOffsetX, oy = hsd.prms.ModeledRegionOffsetY;
    H5::DataSpace scalar(H5S_SCALAR);
    auto writeAttr = [&](auto& ds, const char* name, auto type, const void* val) {
        ds.createAttribute(name, type, scalar).write(type, val);
    };
    
    // We need a dataset to write attributes to. `landmask` works.
    auto ds = file.openDataSet("landmask");
    writeAttr(ds, "GridXTotal", H5::PredType::NATIVE_INT, &gx);
    writeAttr(ds, "GridYTotal", H5::PredType::NATIVE_INT, &gy);
    writeAttr(ds, "OffsetX", H5::PredType::NATIVE_INT, &ox);
    writeAttr(ds, "OffsetY", H5::PredType::NATIVE_INT, &oy);
    writeAttr(ds, "InitImageSizeX", H5::PredType::NATIVE_INT, &m_width);
    writeAttr(ds, "InitImageSizeY", H5::PredType::NATIVE_INT, &m_height);
    writeAttr(ds, "CellSize", H5::PredType::NATIVE_DOUBLE, &hsd.prms.cellsize);

    // Persist the full projection (not just the derived, tangent-point
    // CellSize) so grid.h5 is self-contained: anything reading it back can
    // reconstruct pixel<->lat/lon for any cell (via Projection::ProjectPixel,
    // using OffsetX/OffsetY above to go from grid-local (i,j) to raw image
    // (x,y)) and therefore the local stereographic scale/area-distortion
    // factor for that specific cell (Projection::ScaleFactor/AreaScaleFactor)
    // -- not just the nominal size at the projection's tangent point. This
    // also avoids relying on simulation.json still having matching PROJ_*
    // values whenever this grid.h5 is loaded later.
    writeAttr(ds, "PROJ_LAT_0", H5::PredType::NATIVE_DOUBLE, &hsd.prms.proj.LAT_0);
    writeAttr(ds, "PROJ_LON_0", H5::PredType::NATIVE_DOUBLE, &hsd.prms.proj.LON_0);
    writeAttr(ds, "PROJ_RESIZE_FACTOR", H5::PredType::NATIVE_DOUBLE, &hsd.prms.proj.RESIZE_FACTOR);
    {
        hsize_t coeff_dim[1] = {6};
        H5::DataSpace coeff_space(1, coeff_dim);
        ds.createAttribute("PROJ_TRANSFORM_COEFFS", H5::PredType::NATIVE_DOUBLE, coeff_space)
            .write(H5::PredType::NATIVE_DOUBLE, hsd.prms.proj.TRANSFORM_COEFFS);
    }

    file.close();
    LOGR("PrepareGrid completed");
}

void DataPreparer::LogTotalIceVolume()
{
    const uint64_t numPoints = hsd.hssoa.size;
    double total_ice_volume = 0.0;
    #pragma omp parallel for reduction(+:total_ice_volume)
    for (uint64_t k = 0; k < numPoints; ++k) {
        SOAIterator it = hsd.hssoa.begin() + k;
        double H = it->getValue(SimParams::PtArrIdx::idx_thickness);
        total_ice_volume += hsd.prms.ParticleArea * H;
    }
    LOGR("Total initial ice volume: {:.6e} m^3 ({:.4f} km^3)", total_ice_volume, total_ice_volume / 1.0e9);
}

void DataPreparer::PopulatePoints_RAM_Optimized(int pointsPerCell, double thicknessFrom, double thicknessTo,
                                     double probCracked, double stdDevThickness, bool compress)
{
    LOGR("PopulatePoints_RAM_Optimized (New): starting");

    // (1) Load points from cache
    LOGR("PopulatePoints_RAM_Optimized: preparing point buffer...");
    std::vector<std::array<float, 2>> pt_buffer;
    
    // START OPTIMIZATION: Use Ice Region for point generation
    if (this->IceRegionWidth <= 0 || this->IceRegionHeight <= 0) {
        LOGR("PopulatePoints: IceRegion is empty. Skipping point generation.");
        throw std::runtime_error("no ice");
        return; 
    }

    const int gx = this->IceRegionWidth;
    const int gy = this->IceRegionHeight;
    const int ox = this->IceRegionOffsetX;
    const int oy = this->IceRegionOffsetY;

    // Relative offset from Simulation Grid Origin (ModeledRegionOffset) to Ice Region Origin
    const int rel_ox = ox - hsd.prms.ModeledRegionOffsetX;
    const int rel_oy = oy - hsd.prms.ModeledRegionOffsetY;

    LOGR("PopulatePoints: Generating in Ice Region {}x{}, offset ({},{}), relative ({},{})", 
         gx, gy, ox, oy, rel_ox, rel_oy);

    // Try cache or generate
    if (!attempt_to_fill_from_cache(gx, gy, pointsPerCell, pt_buffer)) {
        generate_and_save_poisson(gx, gy, (float)pointsPerCell, pt_buffer);
    }

    if (pt_buffer.empty()) throw std::runtime_error("No Poisson points generated");

    // (3) Filter points 
    LOGR("PopulatePoints_RAM_Optimized: filtering points...");
    
    // Consistent dy definition
    const float dy = (float)(gy - 1) / (gx - 1);
    
    auto idxPt = [&](const std::array<float, 2> &pt) -> std::pair<int, int> {
        // Correct isotropic mapping matching generate_and_save_poisson mask logic
        int i = (int)(pt[0] * (gx - 1) + 0.5f);
        int j = (int)(pt[1] * (gx - 1) + 0.5f);
        return {i, j};
    };

    auto shouldRemove = [&](const std::array<float, 2> &pt) -> bool {
        auto [i, j] = idxPt(pt);
        if (i <= 1 || j <= 1 || i >= (gx - 2) || j >= (gy - 2)) return true;

        // ox/oy are IceRegionOffsets here
        int img_x = i + ox;
        int img_y = j + oy;
        
        uint8_t flags = m_flags[(size_t)img_y * m_width + img_x];
        if (!(flags & FLAG_WATER)) return true;
        if (!(flags & FLAG_ICE)) return true;
        if (flags & FLAG_OPEN_BOUNDARY) return true;
        return false;
    };

    std::erase_if(pt_buffer, shouldRemove);
    pt_buffer.shrink_to_fit();

    hsd.prms.nPtsInitial = pt_buffer.size();
    if (hsd.prms.nPtsInitial == 0) throw std::runtime_error("All points filtered out!");

    LOGR("PopulatePoints_RAM_Optimized: {} points remaining", hsd.prms.nPtsInitial);

    // Calculate ParticleArea using ICE AREA (not bounding box area)
    long long count_ice_cells = 0;
    for(int j=0; j<gy; ++j) {
        for(int i=0; i<gx; ++i) {
            int img_x = i + ox;
            int img_y = j + oy;
            if (img_x >= 0 && img_x < m_width && img_y >= 0 && img_y < m_height) {
                uint8_t f = m_flags[(size_t)img_y * m_width + img_x];
                if ((f & FLAG_ICE) && (f & FLAG_WATER) && !(f & FLAG_OPEN_BOUNDARY)) {
                    count_ice_cells++;
                }
            }
        }
    }

    const double h = hsd.prms.cellsize;
    double total_ice_area = (double)count_ice_cells * h * h;
    hsd.prms.ParticleArea = total_ice_area / (double)hsd.prms.nPtsInitial;
    
    LOGR("ParticleArea Calc: IceCells={}, TotalArea={:.4e}, nPts={}, PtArea={:.4e}", 
         count_ice_cells, total_ice_area, hsd.prms.nPtsInitial, hsd.prms.ParticleArea);

    // Initial random generators
    std::random_device rd;
    uint32_t points_seed = rd();
    std::mt19937 rng(points_seed);
    // Zero-mean perturbation added on top of the thicknessFrom/thicknessTo-
    // scaled value (then clamped back into [thicknessFrom, thicknessTo]).
    std::normal_distribution<double> thickness_dist(0.0, (double)stdDevThickness);
    // Initial cracked/crushed point generation is currently disabled -- see below.
    // std::bernoulli_distribution cracked_dist(probCracked);
    LOGR("PopulatePoints: random thickness-noise seed {}", points_seed);

    // Checks for generic badness (NaNs, Infs)
    for (size_t k = 0; k < pt_buffer.size(); ++k) {
        if (!std::isfinite(pt_buffer[k][0]) || !std::isfinite(pt_buffer[k][1])) {
             throw std::runtime_error(fmt::format("PopulatePoints: Point {} is NaN/Inf: ({}, {})", k, pt_buffer[k][0], pt_buffer[k][1]));
        }
    }

    // --- Transform and Sort pt_buffer ---
    LOGR("PopulatePoints: Transforming and Sorting points in RAM...");
    
    // Scale factor for transforming normalized coordinates to grid indices.
    // User requested equal scaling for both directions.
    // pt[0] is in [0, 1], mapped to [0, gx-1]
    const double scale = (double)(gx - 1);
    
    // Transform to Global Grid Continuous Coordinates
    for(size_t k = 0; k < pt_buffer.size(); ++k) {
        std::array<float, 2> &pt = pt_buffer[k];
        // pt[0] -> u_global
        pt[0] = (float)(pt[0] * scale + rel_ox);
        // pt[1] -> v_global
        pt[1] = (float)(pt[1] * scale + rel_oy);
    }
    
    // Sort by Cell Index (j + i * GridYTotal)
    // Note: Use a stable sort or regular sort? Regular is fine.
    const int GridYTotal = hsd.prms.GridYTotal;
    std::sort(pt_buffer.begin(), pt_buffer.end(), [GridYTotal](const std::array<float, 2>& a, const std::array<float, 2>& b){
        int ia = (int)(a[0] + 0.5f);
        int ja = (int)(a[1] + 0.5f);
        int ib = (int)(b[0] + 0.5f);
        int jb = (int)(b[1] + 0.5f);
        
        long long idx_a = ja + (long long)ia * GridYTotal;
        long long idx_b = jb + (long long)ib * GridYTotal;
        return idx_a < idx_b;
    });

    // VERIFY RAM SORT
    LOGR("PopulatePoints: Verifying RAM Sort...");
    long long ram_violations = 0;
    for(size_t k=1; k<pt_buffer.size(); ++k) {
         int ia = (int)(pt_buffer[k-1][0] + 0.5f);
         int ja = (int)(pt_buffer[k-1][1] + 0.5f);
         int ib = (int)(pt_buffer[k][0] + 0.5f);
         int jb = (int)(pt_buffer[k][1] + 0.5f);
         long long idx_a = ja + (long long)ia * GridYTotal;
         long long idx_b = jb + (long long)ib * GridYTotal;
         if (idx_a > idx_b) {
             LOGR("RAM Sort Violation at {}! {} > {}", k, idx_a, idx_b);
             ram_violations++;
             if (ram_violations > 5) break;
         }
    }
    if (ram_violations > 0) throw std::runtime_error("RAM Sort Failed");

    if (!pt_buffer.empty()) {
        int ia = (int)(pt_buffer[0][0] + 0.5f);
        int ja = (int)(pt_buffer[0][1] + 0.5f);
        LOGR("First Point in RAM Buffer: ({}, {}) -> idx {}", ia, ja, ja + (long long)ia * GridYTotal);
    }

    // --- Allocate HSSOA ---
    hsd.hssoa.Allocate(hsd.prms.nPtsInitial);
    hsd.hssoa.size = hsd.prms.nPtsInitial;

    LOGR("PopulatePoints: Filling HSSOA...");

    for (size_t k = 0; k < hsd.prms.nPtsInitial; ++k) {
        std::array<float, 2> &pt = pt_buffer[k];
        SOAIterator it = hsd.hssoa.begin() + k;

        // Points are already in Global Continuous Grid Coordinates
        double u_global = pt[0];
        double v_global = pt[1];

        // Global Integer Cell Index
        int i_global = (int)(u_global + 0.5f);
        int j_global = (int)(v_global + 0.5f);

        if (i_global < 0 || i_global >= hsd.prms.GridXTotal ||
            j_global < 0 || j_global >= hsd.prms.GridYTotal) {
            LOGR("FATAL: Point {} out of bounds! Global: ({:.3f}, {:.3f}) -> Int: ({}, {}) Grid: {}x{}",
                 k, u_global, v_global, i_global, j_global, hsd.prms.GridXTotal, hsd.prms.GridYTotal);
            throw std::runtime_error("Point Global Index Out of Bounds");
        }

        // Image Index for property lookup
        // i_global is relative to ModeledRegionOffset
        // img_x is relative to Image Origin (0,0)
        int img_x = i_global + hsd.prms.ModeledRegionOffsetX;
        int img_y = j_global + hsd.prms.ModeledRegionOffsetY;

        if (img_x < 0) img_x = 0; if (img_x >= m_width) img_x = m_width - 1;
        if (img_y < 0) img_y = 0; if (img_y >= m_height) img_y = m_height - 1;

        size_t img_idx = (size_t)img_y * m_width + img_x;

        // Utility Data (Color). Initial cracked/crushed point generation is
        // currently disabled -- see ProxyPoint::setColorRGB's doc comment.
        it->setColorRGB(m_color[img_idx * 3 + 0], m_color[img_idx * 3 + 1], m_color[img_idx * 3 + 2]);

        // Global Cell Indices
        it->setCellXY((unsigned)i_global, (unsigned)j_global);

        if (i_global < 1 || j_global < 1 || i_global > hsd.prms.GridXTotal-2 || j_global > hsd.prms.GridYTotal-2) {
            throw std::runtime_error(fmt::format("Point too close to grid boundary! ({}, {})",
                                                 i_global, j_global));
        }

        // Normalized Local Position [-0.5, 0.5)
        it->setValue(SimParams::PtArrIdx::posx, u_global - (double)i_global);
        it->setValue(SimParams::PtArrIdx::posy, v_global - (double)j_global);

        // Velocities
        it->setValue(SimParams::PtArrIdx::velx, 0.0);
        it->setValue(SimParams::PtArrIdx::vely, 0.0);

        // Thickness: m_thickness is a 0-255 map, linearly mixed into [thicknessFrom, thicknessTo].
        uint8_t t_val = m_thickness[img_idx];
        float t_norm = (float)t_val / 255.0f;
        double thick_val = thicknessFrom + t_norm * (thicknessTo - thicknessFrom);

        if (stdDevThickness > 0.0) {
            double noise = thickness_dist(rng);
            thick_val += noise;
            thick_val = std::clamp(thick_val, (double)thicknessFrom, (double)thicknessTo);
        }
        it->setValue(SimParams::PtArrIdx::idx_thickness, thick_val);

        // Jp_inv
        it->setValue(SimParams::PtArrIdx::idx_Jp_inv, 1.0);

        // Squared projection point-scale-factor (m^2) at this point's seeded
        // location, cached once here -- see SimParams::PtArrIdx::idx_m_squared.
        // Same img_x/img_y -> raw-pixel Y-flip convention as
        // GPU_Implementation5::update_scale_factor_field.
        {
            int raw_y = m_height - 1 - img_y;
            Projection::LatLon ll = hsd.prms.proj.ProjectPixel(img_x, raw_y);
            double m_squared = 1.0;
            if (ll.valid) {
                constexpr double deg2rad = 0.017453292519943295; // pi/180
                double lat_rad = ll.lat_deg * deg2rad;
                double lon_rad = ll.lon_deg * deg2rad;
                m_squared = hsd.prms.proj.AreaScaleFactor(lat_rad, lon_rad);
            }
            it->setValue(SimParams::PtArrIdx::idx_m_squared, m_squared);
        }

        // Fe identity. host_buffer is plain new[], not zero-initialized, so the
        // off-diagonal terms must be set explicitly -- kernels.cu reads all 4.
        it->setValue(SimParams::PtArrIdx::Fe00, 1.0);
        it->setValue((size_t)SimParams::PtArrIdx::Fe00 + 1, 0.0);
        it->setValue((size_t)SimParams::PtArrIdx::Fe00 + (size_t)SimParams::dim, 0.0);
        it->setValue((size_t)SimParams::PtArrIdx::Fe00 + (size_t)SimParams::dim + 1, 1.0);

        // Thermal spin-up state; Ts_old is a placeholder until InitThermalState() runs.
        it->setValue(SimParams::PtArrIdx::idx_A1, 0.0);
        it->setValue(SimParams::PtArrIdx::idx_A2, 0.0);
        it->setValue(SimParams::PtArrIdx::idx_A3, 0.0);
        it->setValue(SimParams::PtArrIdx::idx_xi, 0.0);
        it->setValue(SimParams::PtArrIdx::idx_Ts_old, 0.0);

        // p-q at fracture onset; filled in once by partition_kernel_g2p the
        // first time this point cracks (see kernels.cu).
        it->setValue(SimParams::PtArrIdx::idx_stored_P, 0.0);
        it->setValue(SimParams::PtArrIdx::idx_stored_Q, 0.0);

        // Accumulated plastic shear strain; incremented by PlasticProjection.
        it->setValue(SimParams::PtArrIdx::idx_gamma_p, 0.0);

        // Historical peak of Jp_inv; starts equal to Jp_inv's own initial value.
        it->setValue(SimParams::PtArrIdx::idx_Jp_inv_max, 1.0);
    }

    // Clear buffer!
    LOGR("PopulatePoints: Cleaning up RAM buffer...");
    pt_buffer.clear(); 
    pt_buffer.shrink_to_fit();

    // Sort
    LOGR("PopulatePoints: Sorting points...");
    hsd.hssoa.RemoveDisabledAndSort(hsd.prms.GridYTotal);

    // VERIFY SOA SORT
    LOGR("PopulatePoints: Verifying SOA Sort...");
    if (hsd.hssoa.size > 0) {
        long long soa_violations = 0;
        SOAIterator it = hsd.hssoa.begin();
        long long prev_idx = (*it).getCellIndex(hsd.prms.GridYTotal);
        ++it;
        int k=1;
        for (; it != hsd.hssoa.end(); ++it, ++k) {
             long long curr_idx = (*it).getCellIndex(hsd.prms.GridYTotal);
             if (curr_idx < prev_idx) {
                 LOGR("SOA Sort Violation at {}! {} < {}", k, curr_idx, prev_idx);
                 soa_violations++;
                 if (soa_violations > 5) break; 
             }
             prev_idx = curr_idx;
        }
        if (soa_violations > 0) throw std::runtime_error("SOA Sort Failed");
        
        // Log first point in SOA
        SOAIterator it0 = hsd.hssoa.begin();
        long long idx0 = (*it0).getCellIndex(hsd.prms.GridYTotal);
        unsigned x0 = (*it0).getCellX(); // helper method
        LOGR("First Point in SOA: idx {} (x={}, y={}) Hex 0x{:016x}", idx0, x0, idx0 - (long long)x0 * hsd.prms.GridYTotal, (*it0).getValueUInt64(SimParams::PtArrIdx::integer_cell_idx));
    }

    // Save
    LOGR("PopulatePoints: Saving s00000.h5 via HSSOA...");
    std::string snapDir = hsd.dirs.SnapshotsDirectory();
    hsd.SaveSnapshot(0, 0.0, true, snapDir);

    LogTotalIceVolume();

    LOGR("PopulatePoints_RAM_Optimized (New) completed");
}
