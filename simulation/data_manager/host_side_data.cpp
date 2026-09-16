// host_side_data.cpp

#include "host_side_data.h"
#include "poisson_disk_sampling.h"

#include <H5Cpp.h>
#include <spdlog/spdlog.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb_image_resize2.h"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <random>
#include <sstream>
#include <string>
#include <type_traits>
#include <utility>

#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <map>
#include <stdexcept>

#include <fmt/format.h>
#include <fmt/std.h>

namespace fs = std::filesystem;

HostSideData::HostSideData() : currentInterp(prms, dirs), windInterp(prms, dirs) {
  prms.Reset();
}

void HostSideData::FillModelledAreaWithBlueColor() {
  const int &width_ref = prms.InitializationImageSizeX;
  const int &height_ref = prms.InitializationImageSizeY;
  const int &ox_ref = prms.ModeledRegionOffsetX;
  const int &oy_ref = prms.ModeledRegionOffsetY;
  const int &gx_ref = prms.GridXTotal;
  const int &gy_ref = prms.GridYTotal;

  const size_t width = (size_t)width_ref;
  // const size_t height = (size_t)height_ref;
  const size_t ox = (size_t)ox_ref;
  const size_t oy = (size_t)oy_ref;
  const size_t gx = (size_t)gx_ref;
  const size_t gy = (size_t)gy_ref;

  for (size_t i = 0; i < gx; i++) {
    for (size_t j = 0; j < gy; j++) {
      uint8_t status = landmask_buffer[j + i * gy];
      if (status == SimParams::ModelledAreaIndicator) {
        for (int k = 0; k < 3; k++) {
          size_t idx = ((i + ox) + (j + oy) * width) * 3 + k;
          original_image_colors_rgb[idx] = ColorMap::rgb_water[k];
        }
      } else if (status == SimParams::OpenBoundaryIndicator) {
        // Overwrite the raw basemap pixel with black -- otherwise pt_*
        // views (which, unlike the 'regions' view, don't recolor
        // non-modeled cells per-frame) show whatever the source color
        // image happens to contain there, which can disagree with the
        // footprint-mask-derived open-boundary classification by a cell
        // or two at the mask's edges.
        for (int k = 0; k < 3; k++) {
          size_t idx = ((i + ox) + (j + oy) * width) * 3 + k;
          original_image_colors_rgb[idx] = 0;
        }
      }
    }
  }
}

// =============================  ALLOCATION FUNCTIONS

void HostSideData::AllocateGridArrays(bool allocate_dense_grid) {
  // Allocate grid buffers (for both preparer and simulation)
  const size_t modeled_grid_total = (size_t)prms.GridXTotal * prms.GridYTotal;
  const size_t initial_image_total =
      (size_t)prms.InitializationImageSizeX * prms.InitializationImageSizeY;

  allocated_bytes[0] = 0; // Reset grid allocation counter

  landmask_buffer.resize(modeled_grid_total);
  allocated_bytes[0] += modeled_grid_total * sizeof(uint8_t);

  analysis_mask_buffer.resize(modeled_grid_total);
  allocated_bytes[0] += modeled_grid_total * sizeof(uint8_t);

  // Allocate visualization/saving buffers
  size_t plane_size = (size_t)prms.GridXTotal * prms.GridYTotal;

  original_image_colors_rgb.resize(3 * initial_image_total);
  allocated_bytes[0] += 3 * initial_image_total * sizeof(uint8_t);

  // Resize the main container to hold all potential pointers/vectors
  host_grid_buffer.resize(SimParams::HostGridArrayIndex::nGridArraysHost);

  if (allocate_dense_grid) {
    // Simulation Mode: Allocate EVERYTHING
    for (int i = 0; i < SimParams::HostGridArrayIndex::nGridArraysHost; ++i) {
      host_grid_buffer[i].resize(modeled_grid_total, 0.0f);
      allocated_bytes[0] += modeled_grid_total * sizeof(float);
    }

    // Halo buffer is needed for halo exchange
    size_t halo_size = (size_t)prms.GridYTotal * prms.GridHaloSize;
    tmp_halo_buffer.resize(halo_size *
                           SimParams::HostGridArrayIndex::nGridArraysHost);
    allocated_bytes[0] += halo_size *
                          SimParams::HostGridArrayIndex::nGridArraysHost *
                          sizeof(float);

    save_buffer_float.resize(4 * plane_size);
    save_buffer_uint8.resize(4 * plane_size);

    // Track memory
    allocated_bytes[0] += (save_buffer_float.capacity() * sizeof(float));
    allocated_bytes[0] += (save_buffer_uint8.capacity() * sizeof(uint8_t));
  } else {
    // Visualization Mode (Sparse Allocation)
    std::vector<int> defaults = {
        SimParams::HostGridArrayIndex::grid_idx_vis_pts_density,
        SimParams::HostGridArrayIndex::grid_idx_fracture_tension,
        SimParams::HostGridArrayIndex::grid_idx_fracture_shear,
        SimParams::HostGridArrayIndex::grid_idx_fracture_crush,
        SimParams::HostGridArrayIndex::host_grid_idx_mass,
        SimParams::HostGridArrayIndex::grid_idx_vis_Jpinv,
        SimParams::HostGridArrayIndex::grid_idx_vis_P,
        SimParams::HostGridArrayIndex::grid_idx_vis_Q};

    for (int idx : defaults) {
      AllocateGridArray(idx);
    }

    // 'rgb' buffer is used for saving frames (simulation) but not needed for
    // preparer
    allocated_bytes[0] += 3 * initial_image_total * sizeof(uint8_t);
  }

  LOGR("[MEMORY] HostSideData Grid Arrays: {:.3f} MB",
       allocated_bytes[0] / 1.0e6);
  LOGR("    landmask_buffer: {:.3f} MB",
       (double)landmask_buffer.size() * sizeof(uint8_t) / 1.0e6);
  LOGR("    original_image_colors_rgb: {:.3f} MB",
       (double)original_image_colors_rgb.size() * sizeof(uint8_t) / 1.0e6);
  if (allocate_dense_grid)
    LOGR("    host_grid_buffer (ALL): {:.3f} MB",
         (double)(modeled_grid_total *
                  SimParams::HostGridArrayIndex::nGridArraysHost *
                  sizeof(float)) /
             1.0e6);
  else
    LOGR("    host_grid_buffer (SPARSE): {:.3f} MB",
         (double)allocated_bytes[0] / 1.0e6);
}

void HostSideData::AllocateGridArray(int arrayIndex) {
  if (arrayIndex < 0 ||
      arrayIndex >= SimParams::HostGridArrayIndex::nGridArraysHost)
    return;

  // Ensure main buffer is sized correctly
  if (host_grid_buffer.size() <
      SimParams::HostGridArrayIndex::nGridArraysHost) {
    host_grid_buffer.resize(SimParams::HostGridArrayIndex::nGridArraysHost);
  }

  if (host_grid_buffer[arrayIndex].empty()) {
    size_t size = (size_t)prms.GridXTotal * prms.GridYTotal;
    if (size == 0)
      return; // Don't allocate if dimensions are zero

    host_grid_buffer[arrayIndex].resize(size, 0.0f);
    allocated_bytes[0] += size * sizeof(float);
    LOGR("Allocated grid array index {} ({:.2f} MB)", arrayIndex,
         (double)size * sizeof(float) / 1.0e6);
  }
}

bool HostSideData::IsGridArrayAllocated(int arrayIndex) const {
  if (host_grid_buffer.empty())
    return false;
  if (arrayIndex < 0 ||
      arrayIndex >= SimParams::HostGridArrayIndex::nGridArraysHost)
    return false;
  return !host_grid_buffer[arrayIndex].empty();
}

float *HostSideData::GetGridBufferPointer(int arrayIndex) {
  if (host_grid_buffer.empty()) {
    return nullptr;
  }

  // Safety check for index
  if (arrayIndex < 0 ||
      arrayIndex >= SimParams::HostGridArrayIndex::nGridArraysHost) {
    LOGR("GetGridBufferPointer: Invalid array index {}", arrayIndex);
    return nullptr;
  }

  if (host_grid_buffer[arrayIndex].empty())
    return nullptr;
  return host_grid_buffer[arrayIndex].data();
}

// =============================  LOAD GRID DATA FROM FILE

void HostSideData::LoadGridDataFromFile(const std::string &gridFilePath) {
  LOGR("LoadGridDataFromFile: starting from {}", gridFilePath);

  try {
    // Open HDF5 file
    H5::H5File file(gridFilePath, H5F_ACC_RDONLY);

    // Open landmask dataset to read attributes
    H5::DataSet ds_landmask = file.openDataSet("landmask");
    H5::DataSpace space_landmask = ds_landmask.getSpace();

    // Read metadata attributes
    H5::Attribute attr_gx = ds_landmask.openAttribute("GridXTotal");
    H5::Attribute attr_gy = ds_landmask.openAttribute("GridYTotal");
    H5::Attribute attr_ox = ds_landmask.openAttribute("OffsetX");
    H5::Attribute attr_oy = ds_landmask.openAttribute("OffsetY");
    H5::Attribute attr_img_x = ds_landmask.openAttribute("InitImageSizeX");
    H5::Attribute attr_img_y = ds_landmask.openAttribute("InitImageSizeY");
    H5::Attribute attr_cellsize = ds_landmask.openAttribute("CellSize");

    attr_gx.read(H5::PredType::NATIVE_INT, &prms.GridXTotal);
    attr_gy.read(H5::PredType::NATIVE_INT, &prms.GridYTotal);
    attr_ox.read(H5::PredType::NATIVE_INT, &prms.ModeledRegionOffsetX);
    attr_oy.read(H5::PredType::NATIVE_INT, &prms.ModeledRegionOffsetY);
    attr_img_x.read(H5::PredType::NATIVE_INT, &prms.InitializationImageSizeX);
    attr_img_y.read(H5::PredType::NATIVE_INT, &prms.InitializationImageSizeY);
    attr_cellsize.read(H5::PredType::NATIVE_DOUBLE, &prms.cellsize);

    // Load the full projection grid.h5 was built with (not just the
    // derived, tangent-point CellSize) -- this is what's needed later to
    // reconstruct any cell's lat/lon (Projection::ProjectPixel, using
    // OffsetX/OffsetY read above) and therefore its local stereographic
    // scale/area-distortion factor (Projection::ScaleFactor/
    // AreaScaleFactor), rather than relying on simulation.json still having
    // matching PROJ_* values whenever this grid.h5 happens to be loaded.
    H5::Attribute attr_proj_lat0 = ds_landmask.openAttribute("PROJ_LAT_0");
    H5::Attribute attr_proj_lon0 = ds_landmask.openAttribute("PROJ_LON_0");
    H5::Attribute attr_proj_resize = ds_landmask.openAttribute("PROJ_RESIZE_FACTOR");
    H5::Attribute attr_proj_coeffs = ds_landmask.openAttribute("PROJ_TRANSFORM_COEFFS");
    attr_proj_lat0.read(H5::PredType::NATIVE_DOUBLE, &prms.proj.LAT_0);
    attr_proj_lon0.read(H5::PredType::NATIVE_DOUBLE, &prms.proj.LON_0);
    attr_proj_resize.read(H5::PredType::NATIVE_DOUBLE, &prms.proj.RESIZE_FACTOR);
    attr_proj_coeffs.read(H5::PredType::NATIVE_DOUBLE, prms.proj.TRANSFORM_COEFFS);

    // Compute derived parameters
    prms.cellsize_inv = 1.0 / prms.cellsize;

    LOGR("Loaded grid metadata: gx={}, gy={}, offset=({},{}), cellsize={}",
         prms.GridXTotal, prms.GridYTotal, prms.ModeledRegionOffsetX,
         prms.ModeledRegionOffsetY, prms.cellsize);
    LOGR("Loaded projection: LAT_0={}, LON_0={}, RESIZE_FACTOR={}",
         prms.proj.LAT_0, prms.proj.LON_0, prms.proj.RESIZE_FACTOR);

    // Validate dimensions
    if (prms.GridXTotal <= 0 || prms.GridYTotal <= 0) {
      throw std::runtime_error(
          fmt::format("Invalid grid dimensions: GridXTotal={}, GridYTotal={}",
                      prms.GridXTotal, prms.GridYTotal));
    }
    if (prms.InitializationImageSizeX <= 0 ||
        prms.InitializationImageSizeY <= 0) {
      throw std::runtime_error(fmt::format(
          "Invalid image dimensions: ImageX={}, ImageY={}",
          prms.InitializationImageSizeX, prms.InitializationImageSizeY));
    }

    // Check landmask dataset dimensions
    hsize_t landmask_dims[2];
    space_landmask.getSimpleExtentDims(landmask_dims, NULL);
    if ((int)landmask_dims[0] != prms.GridXTotal ||
        (int)landmask_dims[1] != prms.GridYTotal) {
      throw std::runtime_error(fmt::format(
          "Landmask dataset dimensions [{}, {}] don't match metadata [{}, {}]",
          landmask_dims[0], landmask_dims[1], prms.GridXTotal,
          prms.GridYTotal));
    }

    // Check color_grid dataset dimensions
    H5::DataSet ds_color = file.openDataSet("color_grid");
    H5::DataSpace space_color = ds_color.getSpace();
    hsize_t color_dims[3];
    space_color.getSimpleExtentDims(color_dims, NULL);
    if ((int)color_dims[0] != prms.InitializationImageSizeY ||
        (int)color_dims[1] != prms.InitializationImageSizeX ||
        (int)color_dims[2] != 3) {
      throw std::runtime_error(fmt::format(
          "Color grid dataset dimensions [{}, {}, {}] don't match expected "
          "[{}, {}, 3]",
          color_dims[0], color_dims[1], color_dims[2],
          prms.InitializationImageSizeY, prms.InitializationImageSizeX));
    }

    // Always allocate densely -- lazy/sparse on-demand allocation for interactive visualization was removed (it caused stale-data bugs when switching between visualization options).
    AllocateGridArrays(true);

    // Read landmask dataset
    ds_landmask.read(landmask_buffer.data(), H5::PredType::NATIVE_UINT8);

    // Read analysis_mask dataset, if present -- older grid.h5 files (saved
    // before this was added, or projects with no AnalysisRegionMask
    // configured) won't have it, in which case analysis_mask_buffer stays
    // all-zero (nothing included).
    if (H5Lexists(file.getId(), "analysis_mask", H5P_DEFAULT) > 0) {
      H5::DataSet ds_analysis_mask = file.openDataSet("analysis_mask");
      ds_analysis_mask.read(analysis_mask_buffer.data(), H5::PredType::NATIVE_UINT8);
    } else {
      std::fill(analysis_mask_buffer.begin(), analysis_mask_buffer.end(), 0);
    }

    // Read color_grid dataset
    ds_color.read(original_image_colors_rgb.data(), H5::PredType::NATIVE_UINT8);

    file.close();

    LOGR("LoadGridDataFromFile: completed successfully");

  } catch (const H5::Exception &e) {
    throw std::runtime_error(
        fmt::format("Failed to load grid.h5: {}", e.getCDetailMsg()));
  } catch (const std::exception &e) {
    throw std::runtime_error(
        fmt::format("LoadGridDataFromFile failed: {}", e.what()));
  }
}

void HostSideData::ReadPointsFromSnapshot(std::string fileNameSnapshotHDF5) {
  LOGR("ReadPointsFromSnapshot {}", fileNameSnapshotHDF5);

  // Open HDF5 file and dataset
  H5::H5File file(fileNameSnapshotHDF5, H5F_ACC_RDONLY);
  H5::DataSet ds = file.openDataSet("pts_data");

  // Get dimensions from file to determine actual point count
  H5::DataSpace dsp = ds.getSpace();
  hsize_t dims[2];
  dsp.getSimpleExtentDims(dims, nullptr);

  // Read attributes
  ds.openAttribute("nPtsInitial")
      .read(H5::PredType::NATIVE_INT, &prms.nPtsInitial);

  // Ensure we allocate enough space for what's in the file, even if it exceeds
  // nPtsInitial
  size_t pts_in_file = dims[1];
  size_t required_for_file =
      (size_t)(pts_in_file * (1.0 + prms.extra_space_pts));

  // Also respect the standard heuristic based on nPtsInitial
  size_t standard_heuristic =
      (size_t)(double(prms.nPtsInitial) * (1.0 + prms.extra_space_pts));

  const size_t requested_capacity =
      std::max(required_for_file, standard_heuristic);

  LOGR("ReadPointsFromSnapshot: nPtsInitial={}, file_pts={}. Allocating "
       "capacity={}",
       prms.nPtsInitial, pts_in_file, requested_capacity);

  hssoa.Allocate(requested_capacity);
  allocated_bytes[1] = (size_t)hssoa.capacity *
                        (size_t)SimParams::PtArrIdx::nPtsArrays *
                        sizeof(double);

  ds.openAttribute("SimulationStep")
      .read(H5::PredType::NATIVE_INT, &prms.SimulationStep);
  ds.openAttribute("SimulationTime")
      .read(H5::PredType::NATIVE_DOUBLE, &prms.SimulationTime);

  ds.openAttribute("ParticleArea")
      .read(H5::PredType::NATIVE_DOUBLE, &prms.ParticleArea);
  if (H5Aexists(ds.getId(), "GridYTotal") > 0) {
    int readGridY;
    ds.openAttribute("GridYTotal").read(H5::PredType::NATIVE_INT, &readGridY);
    if (prms.GridYTotal != 0) {
      if (prms.GridYTotal != readGridY) {
        throw std::runtime_error(fmt::format(
            "ReadPointsFromSnapshot GridYTotal mismatch: expected {}, got {}",
            prms.GridYTotal, readGridY));
      }
    }
    prms.GridYTotal = readGridY;
  }
  if (H5Aexists(ds.getId(), "GridXTotal") > 0) {
    int readGridX;
    ds.openAttribute("GridXTotal").read(H5::PredType::NATIVE_INT, &readGridX);
    if (prms.GridXTotal != 0) {
      if (prms.GridXTotal != readGridX) {
        throw std::runtime_error(fmt::format(
            "ReadPointsFromSnapshot GridXTotal mismatch: expected {}, got {}",
            prms.GridXTotal, readGridX));
      }
    }
    prms.GridXTotal = readGridX;
  }

  int nPtsArrays;
  ds.openAttribute("nPtsArrays").read(H5::PredType::NATIVE_INT, &nPtsArrays);

  // Dimensions already read above into 'dims'

  if (nPtsArrays != SimParams::PtArrIdx::nPtsArrays) {
    LOGR("dims {} x {}; nPtsInitial {}", dims[0], dims[1], prms.nPtsInitial);
    throw std::runtime_error("ReadSnapshot array size mismatch");
  }

  hssoa.size = dims[1];

  // Define hyperslab
  hsize_t dims_mem[2] = {SimParams::PtArrIdx::nPtsArrays, hssoa.capacity};
  H5::DataSpace memspace(2, dims_mem);
  hsize_t offset[2] = {0, 0};
  hsize_t count[2] = {dims[0], dims[1]};
  memspace.selectHyperslab(H5S_SELECT_SET, count, offset);

  LOGR("ReadPointsFromSnapshot attempting to read {} pts into hssoa with "
       "capacity {}",
       dims[1], hssoa.capacity);
  ds.read(hssoa.host_buffer, H5::PredType::NATIVE_DOUBLE, memspace, dsp);
  LOGR("ReadPointsFromSnapshot: read successfully; GridYTotal {}",
       prms.GridYTotal);

  hssoa.RemoveDisabledAndSort(prms.GridYTotal);
  VerifyPoints();

  LOGR("ReadPointsFromSnapshot; hssoa capacity {}; size {}", hssoa.capacity,
       hssoa.size);

  // Count cracked/disabled points for diagnostics
  int count_cracked = 0;
  int count_disabled = 0;
  const int n = (int)hssoa.size;
  const unsigned cap = hssoa.capacity;
  double *buf = hssoa.host_buffer;
  for (int i = 0; i < n; i++) {
    ProxyPoint pp;
    pp.isReference = true;
    pp.pos = i;
    pp.pitch = cap;
    pp.soa = buf;

    if (pp.getCrackedStatus())
      count_cracked++;
    if (pp.getDisabledStatus())
      count_disabled++;
  }
  LOGR("ReadPointsFromSnapshot: cracked={}, disabled={} (of {} points)",
       count_cracked, count_disabled, n);
}

// =============================  READ AND WRITE SNAPSHOTS

void HostSideData::SaveSnapshot(int SimulationStep, double SimulationTime,
                                bool compress,
                                const std::string &output_directory,
                                const std::string &prefix,
                                int force_frame_index) {
  LOGR("SaveSnapshot: step {}, time {}{}", SimulationStep, SimulationTime,
       compress ? " (compressed)" : "");

  // Determine output directory
  if (output_directory.empty()) {
    throw std::runtime_error("SaveSnapshot called with empty output_directory");
  }

  fs::path targetPath = output_directory;
  fs::create_directories(targetPath);

  // Save current state
  int frame;
  if (force_frame_index >= 0)
    frame = force_frame_index;
  else
    frame = SimulationStep / prms.UpdateEveryNthStep;

  std::string baseName =
      fmt::format(fmt::runtime("{}{:05d}.h5"), prefix, frame);
  fs::path fullPath = targetPath / baseName;
  H5::H5File file(fullPath.string(), H5F_ACC_TRUNC);

  const auto nPts = hssoa.size;
  const auto capacity = hssoa.capacity;

  // Define file dataspace: what we want to save (nPtsArrays x nPts)
  hsize_t file_dims[2] = {SimParams::PtArrIdx::nPtsArrays, nPts};
  H5::DataSpace file_dataspace(2, file_dims);

  // Define memory dataspace: source data layout (nPtsArrays x capacity)
  hsize_t mem_dims[2] = {SimParams::PtArrIdx::nPtsArrays, capacity};
  H5::DataSpace mem_dataspace(2, mem_dims);

  // Select the region in memory to read from: all arrays, but only first nPts
  // columns
  hsize_t mem_offset[2] = {0, 0};
  hsize_t mem_count[2] = {SimParams::PtArrIdx::nPtsArrays, nPts};
  mem_dataspace.selectHyperslab(H5S_SELECT_SET, mem_count, mem_offset);

  // Create dataset with optional compression
  H5::DataSet dataset_pts;
  if (compress) {
    H5::DSetCreatPropList proplist;
    hsize_t chunk_dims[2] = {SimParams::PtArrIdx::nPtsArrays,
                             std::min((size_t)10000000, (size_t)nPts)};
    proplist.setChunk(2, chunk_dims);
    proplist.setDeflate(1);
    dataset_pts = file.createDataSet("pts_data", H5::PredType::NATIVE_DOUBLE,
                                     file_dataspace, proplist);
  } else {
    dataset_pts =
        file.createDataSet("pts_data", H5::PredType::NATIVE_DOUBLE,
                           file_dataspace, H5::DSetCreatPropList::DEFAULT);
  }

  dataset_pts.write(hssoa.host_buffer, H5::PredType::NATIVE_DOUBLE,
                    mem_dataspace, file_dataspace);

  // Write metadata attributes
  H5::DataSpace att_dspace(H5S_SCALAR);
  dataset_pts
      .createAttribute("SimulationStep", H5::PredType::NATIVE_INT, att_dspace)
      .write(H5::PredType::NATIVE_INT, &SimulationStep);
  dataset_pts
      .createAttribute("SimulationTime", H5::PredType::NATIVE_DOUBLE,
                       att_dspace)
      .write(H5::PredType::NATIVE_DOUBLE, &SimulationTime);

  int nPtsArrays = SimParams::PtArrIdx::nPtsArrays;
  dataset_pts
      .createAttribute("nPtsArrays", H5::PredType::NATIVE_INT, att_dspace)
      .write(H5::PredType::NATIVE_INT, &nPtsArrays);
  dataset_pts
      .createAttribute("nPtsInitial", H5::PredType::NATIVE_INT, att_dspace)
      .write(H5::PredType::NATIVE_INT, &prms.nPtsInitial);
  dataset_pts
      .createAttribute("ParticleArea", H5::PredType::NATIVE_DOUBLE, att_dspace)
      .write(H5::PredType::NATIVE_DOUBLE, &prms.ParticleArea);

  dataset_pts
      .createAttribute("GridYTotal", H5::PredType::NATIVE_INT, att_dspace)
      .write(H5::PredType::NATIVE_INT, &prms.GridYTotal);
  dataset_pts
      .createAttribute("GridXTotal", H5::PredType::NATIVE_INT, att_dspace)
      .write(H5::PredType::NATIVE_INT, &prms.GridXTotal);

  LOGR("SaveSnapshot done");
}

// ============================================================================
// Post-Processor Support Methods
// ============================================================================

void HostSideData::LoadFrameData(int frameIndex,
                                 const std::string &framesDirectory) {
  fs::path framesDir(framesDirectory);

  // Match the filename format from SaveFrame -- one combined file per frame.
  std::string filename = fmt::format("f{:05d}.h5", frameIndex);
  fs::path path = framesDir / filename;

  const int gx = prms.GridXTotal;
  const int gy = prms.GridYTotal;
  const size_t gridSize = (size_t)gx * gy;

  H5::H5File file(path.string(), H5F_ACC_RDONLY);

  file.openAttribute("SimulationStep")
      .read(H5::PredType::NATIVE_INT, &prms.SimulationStep);
  file.openAttribute("SimulationTime")
      .read(H5::PredType::NATIVE_DOUBLE, &prms.SimulationTime);

  H5::DataSet ds = file.openDataSet("rgba");
  frame_rgba.resize(gridSize * 4);
  ds.read(frame_rgba.data(), H5::PredType::NATIVE_UINT8);

  // -------------------------------------------------------
  // Helper Lambda for Grid Arrays (Physics, Strains, etc.)
  // -------------------------------------------------------
  auto load_into_grid = [&](const char *ds_name, int grid_idx,
                            bool normalize = false) {
    // Skip if grid index is not allocated (dynamic sparse loading)
    if (!IsGridArrayAllocated(grid_idx))
      return;

    std::vector<float> &dest_buffer = host_grid_buffer[grid_idx];
    const std::string dataset_name = ds_name;

    // Check if dataset exists before trying to open to avoid exception on
    // missing optional datasets
    if (H5Lexists(file.getId(), dataset_name.c_str(), H5P_DEFAULT) <= 0) {
      return;
    }

    const H5::DataSet dataset = file.openDataSet(dataset_name);
    H5::DataSpace filespace = dataset.getSpace();

    // Validate dimensions
    hsize_t dims_grid[2];
    filespace.getSimpleExtentDims(dims_grid, NULL);
    size_t file_size = dims_grid[0] * dims_grid[1];

    // Safety check to ensure we don't write out of bounds
    if (file_size > dest_buffer.size()) {
      LOGR("Error: Dataset {} is too large for buffer.", dataset_name);
      return;
    }

    // HDF5 automatically converts file types (Double/UInt8) to NATIVE_FLOAT.
    H5::DataType mem_dtype = H5::PredType::NATIVE_FLOAT;
    H5::DataSpace memspace(2, dims_grid);

    dataset.read(dest_buffer.data(), mem_dtype, memspace, filespace);

    // Normalize uint8 [0, 255] -> float [0.0, 1.0] if requested
    if (normalize) {
      float *ptr = dest_buffer.data();
      for (size_t i = 0; i < file_size; ++i)
        ptr[i] = ptr[i] / 255.0f;
    }
  };

  // Fracture status
  load_into_grid("crushed", SimParams::HostGridArrayIndex::grid_idx_vis_crushed, true);
  load_into_grid("cracked", SimParams::HostGridArrayIndex::grid_idx_vis_cracked, true);
  load_into_grid("thickness", SimParams::HostGridArrayIndex::grid_idx_vis_thickness, false);

  // Fracture type
  load_into_grid("tension", SimParams::HostGridArrayIndex::grid_idx_fracture_tension, true);
  load_into_grid("shear", SimParams::HostGridArrayIndex::grid_idx_fracture_shear, true);
  load_into_grid("crush", SimParams::HostGridArrayIndex::grid_idx_fracture_crush, true);

  // Physics
  load_into_grid("mass", SimParams::HostGridArrayIndex::host_grid_idx_mass);
  load_into_grid("vx", SimParams::HostGridArrayIndex::grid_idx_px);
  load_into_grid("vy", SimParams::HostGridArrayIndex::grid_idx_py);

  // Strains
  load_into_grid("strain_eqv", SimParams::HostGridArrayIndex::grid_idx_vis_strain_EqvGreenLagrange);
  load_into_grid("strain_vm", SimParams::HostGridArrayIndex::grid_idx_vis_strain_vonMises);

  // Pressure, shear stress, Jp_inv
  load_into_grid("Jpinv", SimParams::HostGridArrayIndex::grid_idx_vis_Jpinv);
  load_into_grid("P", SimParams::HostGridArrayIndex::grid_idx_vis_P);
  load_into_grid("Q", SimParams::HostGridArrayIndex::grid_idx_vis_Q);

  // Ice strength
  load_into_grid("ice_strength", SimParams::HostGridArrayIndex::grid_idx_vis_ice_strength);

  // Accumulated plastic shear strain (gamma_p)
  load_into_grid("gamma_p", SimParams::HostGridArrayIndex::grid_idx_vis_gamma_p);

  file.close();
  LOGR("LoadFrameData: completed frame {}", frameIndex);
}

template <typename T>
void WriteDatasetHelper(H5::Group &ptr, const std::string &name,
                        const std::vector<T> &data, int gx, int gy,
                        const H5::DataType &dtype, bool compress,
                        int compression_level) {
  hsize_t dims[2] = {(hsize_t)gx, (hsize_t)gy};
  H5::DataSpace dataspace(2, dims);

  H5::DSetCreatPropList proplist = H5::DSetCreatPropList::DEFAULT;
  if (compress) {
    proplist = H5::DSetCreatPropList();
    hsize_t chunk_dims[2] = {(hsize_t)gx, (hsize_t)gy};
    proplist.setChunk(2, chunk_dims);
    proplist.setDeflate(compression_level);
  }

  H5::DataSet dataset = ptr.createDataSet(name, dtype, dataspace, proplist);
  dataset.write(data.data(), dtype);
}

void HostSideData::SaveFrame(const int SimulationStep,
                             const double SimulationTime, bool compress,
                             int compression_level) {
  const int frame = SimulationStep / prms.UpdateEveryNthStep;
  if (prms.SaveAnimationAsJPG && prms.JPG_Crop_Width > 0 &&
      prms.JPG_Crop_Height > 0) {
    RenderAsJPG(frame, VisOpt::grid_fracture_type, prms.JPG_OffsetX,
                prms.JPG_OffsetY, prms.JPG_Crop_Width, prms.JPG_Crop_Height,
                "raster");
  }

  if (!prms.SaveSnapshots)
    return;

  const size_t plane_size = (size_t)prms.GridXTotal * prms.GridYTotal;

  const int gx = prms.GridXTotal;
  const int gy = prms.GridYTotal;
  const std::string frameFileName = fmt::format("f{:05d}.h5", frame);

  fs::path framesDir = dirs.FramesDirectory();
  fs::create_directories(framesDir);
  H5::H5File file((framesDir / frameFileName).string(), H5F_ACC_TRUNC);

  // Attributes (file-level, not tied to any one dataset)
  {
    H5::DataSpace att_dspace(H5S_SCALAR);
    file.createAttribute("SimulationStep", H5::PredType::NATIVE_INT, att_dspace)
        .write(H5::PredType::NATIVE_INT, &SimulationStep);
    file.createAttribute("SimulationTime", H5::PredType::NATIVE_DOUBLE,
                         att_dspace)
        .write(H5::PredType::NATIVE_DOUBLE, &SimulationTime);
  }

  // Physics: mass, vx, vy
  WriteDatasetHelper(
      file, "mass",
      host_grid_buffer[SimParams::HostGridArrayIndex::host_grid_idx_mass], gx,
      gy, H5::PredType::NATIVE_FLOAT, compress, compression_level);
  WriteDatasetHelper(
      file, "vx",
      host_grid_buffer[SimParams::HostGridArrayIndex::grid_idx_px], gx, gy,
      H5::PredType::NATIVE_FLOAT, compress, compression_level);
  WriteDatasetHelper(
      file, "vy",
      host_grid_buffer[SimParams::HostGridArrayIndex::grid_idx_py], gx, gy,
      H5::PredType::NATIVE_FLOAT, compress, compression_level);

  // Color: rgb + alpha (uint8)
  {
#pragma omp parallel for
    for (size_t i = 0; i < plane_size; i++) {
      float d = host_grid_buffer
          [SimParams::HostGridArrayIndex::grid_idx_vis_pts_density][i];
      float alpha = std::clamp(d * 0.4f, 0.0f, 1.0f);

      float r =
          host_grid_buffer[SimParams::HostGridArrayIndex::grid_idx_vis_r][i];
      float g =
          host_grid_buffer[SimParams::HostGridArrayIndex::grid_idx_vis_g][i];
      float b =
          host_grid_buffer[SimParams::HostGridArrayIndex::grid_idx_vis_b][i];

      save_buffer_uint8[i * 4 + 0] = (uint8_t)(std::clamp(r, 0.f, 1.f) * 255.f);
      save_buffer_uint8[i * 4 + 1] = (uint8_t)(std::clamp(g, 0.f, 1.f) * 255.f);
      save_buffer_uint8[i * 4 + 2] = (uint8_t)(std::clamp(b, 0.f, 1.f) * 255.f);
      save_buffer_uint8[i * 4 + 3] = (uint8_t)(alpha * 255.f);
    }

    hsize_t dims[3] = {(hsize_t)gx, (hsize_t)gy, 4};
    H5::DataSpace dataspace(3, dims);

    H5::DSetCreatPropList proplist = H5::DSetCreatPropList::DEFAULT;
    if (compress) {
      proplist = H5::DSetCreatPropList();
      hsize_t chunk_dims[3] = {(hsize_t)gx, (hsize_t)gy, 4};
      proplist.setChunk(3, chunk_dims);
      proplist.setDeflate(compression_level);
    }

    H5::DataSet dataset = file.createDataSet("rgba", H5::PredType::NATIVE_UINT8,
                                             dataspace, proplist);
    dataset.write(save_buffer_uint8.data(), H5::PredType::NATIVE_UINT8);
  }

  // Strains: EqvGreenLagrange, vonMises (float)
  WriteDatasetHelper(
      file, "strain_eqv",
      host_grid_buffer[SimParams::HostGridArrayIndex::
                           grid_idx_vis_strain_EqvGreenLagrange],
      gx, gy, H5::PredType::NATIVE_FLOAT, compress, compression_level);
  WriteDatasetHelper(
      file, "strain_vm",
      host_grid_buffer
          [SimParams::HostGridArrayIndex::grid_idx_vis_strain_vonMises],
      gx, gy, H5::PredType::NATIVE_FLOAT, compress, compression_level);

  // Fracture status: crushed, cracked (uint8), thickness (float)
#pragma omp parallel for
  for (size_t i = 0; i < plane_size; i++) {
    float v =
        host_grid_buffer[SimParams::HostGridArrayIndex::grid_idx_vis_crushed]
                        [i];
    save_buffer_uint8[i] = (uint8_t)(std::clamp(v, 0.f, 1.f) * 255.f);
  }
  WriteDatasetHelper(file, "crushed", save_buffer_uint8, gx, gy,
                     H5::PredType::NATIVE_UINT8, compress, compression_level);

#pragma omp parallel for
  for (size_t i = 0; i < plane_size; i++) {
    float v =
        host_grid_buffer[SimParams::HostGridArrayIndex::grid_idx_vis_cracked]
                        [i];
    save_buffer_uint8[i] = (uint8_t)(std::clamp(v, 0.f, 1.f) * 255.f);
  }
  WriteDatasetHelper(file, "cracked", save_buffer_uint8, gx, gy,
                     H5::PredType::NATIVE_UINT8, compress, compression_level);
  WriteDatasetHelper(
      file, "thickness",
      host_grid_buffer[SimParams::HostGridArrayIndex::grid_idx_vis_thickness],
      gx, gy, H5::PredType::NATIVE_FLOAT, compress, compression_level);

  // Fracture type: tension, shear, crush (uint8)
#pragma omp parallel for
  for (size_t i = 0; i < plane_size; i++) {
    float v = host_grid_buffer
        [SimParams::HostGridArrayIndex::grid_idx_fracture_tension][i];
    save_buffer_uint8[i] = (uint8_t)(std::clamp(v, 0.f, 1.f) * 255.f);
  }
  WriteDatasetHelper(file, "tension", save_buffer_uint8, gx, gy,
                     H5::PredType::NATIVE_UINT8, compress, compression_level);

#pragma omp parallel for
  for (size_t i = 0; i < plane_size; i++) {
    float v = host_grid_buffer
        [SimParams::HostGridArrayIndex::grid_idx_fracture_shear][i];
    save_buffer_uint8[i] = (uint8_t)(std::clamp(v, 0.f, 1.f) * 255.f);
  }
  WriteDatasetHelper(file, "shear", save_buffer_uint8, gx, gy,
                     H5::PredType::NATIVE_UINT8, compress, compression_level);

#pragma omp parallel for
  for (size_t i = 0; i < plane_size; i++) {
    float v = host_grid_buffer
        [SimParams::HostGridArrayIndex::grid_idx_fracture_crush][i];
    save_buffer_uint8[i] = (uint8_t)(std::clamp(v, 0.f, 1.f) * 255.f);
  }
  WriteDatasetHelper(file, "crush", save_buffer_uint8, gx, gy,
                     H5::PredType::NATIVE_UINT8, compress, compression_level);

  // Pressure, shear stress, Jp_inv
  WriteDatasetHelper(
      file, "P",
      host_grid_buffer[SimParams::HostGridArrayIndex::grid_idx_vis_P], gx, gy,
      H5::PredType::NATIVE_FLOAT, compress, compression_level);
  WriteDatasetHelper(
      file, "Q",
      host_grid_buffer[SimParams::HostGridArrayIndex::grid_idx_vis_Q], gx, gy,
      H5::PredType::NATIVE_FLOAT, compress, compression_level);
  WriteDatasetHelper(
      file, "Jpinv",
      host_grid_buffer[SimParams::HostGridArrayIndex::grid_idx_vis_Jpinv], gx,
      gy, H5::PredType::NATIVE_FLOAT, compress, compression_level);

  // Ice strength
  WriteDatasetHelper(
      file, "ice_strength",
      host_grid_buffer[SimParams::HostGridArrayIndex::grid_idx_vis_ice_strength],
      gx, gy, H5::PredType::NATIVE_FLOAT, compress, compression_level);

  // Accumulated plastic shear strain (gamma_p)
  WriteDatasetHelper(
      file, "gamma_p",
      host_grid_buffer[SimParams::HostGridArrayIndex::grid_idx_vis_gamma_p],
      gx, gy, H5::PredType::NATIVE_FLOAT, compress, compression_level);

  file.close();
}

void HostSideData::VerifyPoints() {
  LOGR("Verifying points integrity...");
  int count_invalid_pos = 0;
  int count_invalid_idx = 0;

  // Bounds definitions
  const long long min_x = 0;
  const long long max_x = min_x + prms.GridXTotal;
  const long long min_y = 0;
  const long long max_y = min_y + prms.GridYTotal;

  LOGR("Domain bounds: X [{}, {}), Y [{}, {})", min_x, max_x, min_y, max_y);

  const int n = (int)hssoa.size;
  const unsigned cap = hssoa.capacity;
  double *buf = hssoa.host_buffer;

  for (int i = 0; i < n; i++) {
    ProxyPoint pp;
    pp.isReference = true;
    pp.pos = i;
    pp.pitch = cap;
    pp.soa = buf;

    if (pp.getDisabledStatus())
      continue;

    // Check local coordinates
    // getPos() normally returns local coordinates for PIC/FLIP
    Eigen::Vector2d pos = pp.getPos();

    // Strict check [-0.5, 0.5]
    const double tolerance = 1e-6;
    if (pos.x() < -0.5 - tolerance || pos.x() > 0.5 + tolerance ||
        pos.y() < -0.5 - tolerance || pos.y() > 0.5 + tolerance) {
      count_invalid_pos++;
      // Avoid logging inside OMP parallel region to prevent race/garbled output
      // or excessive locking. Just count them.
    }

    // Check cell indices
    // Manually unpack integer_cell_idx as per standard encoding
    uint64_t cell = pp.getValueUInt64(SimParams::PtArrIdx::integer_cell_idx);
    long long x_idx = (long long)(cell & 0xffffffff);
    long long y_idx = (long long)(cell >> 32);

    if (x_idx < min_x || x_idx >= max_x || y_idx < min_y || y_idx >= max_y) {
      count_invalid_idx++;
      LOGR("bounds [{},{}]x[{},{}]; cell [{},{}]", min_x, min_y, max_x, max_y,
           x_idx, y_idx);
      throw std::runtime_error("point boudns check failed");
      spdlog::default_logger()->flush();
    }
  }

  LOGR("VerifyPoints: all {} active points verified successfully.", hssoa.size);
  spdlog::default_logger()->flush();
}

void HostSideData::RenderGridRaster(const VisOpt::Type visOpt,
                                    std::vector<uint8_t> &renderedImage,
                                    int offsetX, int offsetY, int sizeX,
                                    int sizeY) {
    const std::vector<uint8_t> &grid_status = landmask_buffer;
    const std::vector<uint8_t> &original_colors = original_image_colors_rgb;

    const int width = prms.InitializationImageSizeX;
    const int height = prms.InitializationImageSizeY;

    if (sizeX == -1)
        sizeX = width;
    if (sizeY == -1)
        sizeY = height;

    const int ox = prms.ModeledRegionOffsetX;
    const int oy = prms.ModeledRegionOffsetY;
    const int gx = prms.GridXTotal;
    const int gy = prms.GridYTotal;
    const size_t gridSize = (size_t)gx * gy;
    const double range_from = VisOpt::range_from[visOpt];
    const double range_to = VisOpt::range_to[visOpt];
    // Scale used for magnitude-based alpha blending (calc_mix below): the
    // largest extent of [range_from, range_to] from zero, so a value at the
    // edge of the visible range gives full blend weight.
    const double mix_scale = std::max(std::abs(range_from), std::abs(range_to));
    const double transparency = VisOpt::transparency_coeffs[visOpt];

    // Resize renderedImage for the custom sub-region
    size_t out_size = (size_t)sizeX * sizeY * 3;
    renderedImage.resize(out_size);

// Fill the background for the sub-region
#pragma omp parallel for
    for (int j = 0; j < sizeY; j++) {
        for (int i = 0; i < sizeX; i++) {
            int im_x = i + offsetX;
            int im_y = j + offsetY;
            int out_idx = (i + j * sizeX) * 3;

            if (im_x >= 0 && im_x < width && im_y >= 0 && im_y < height) {
                int orig_idx = (im_x + im_y * width) * 3;
                renderedImage[out_idx + 0] = original_colors[orig_idx + 0];
                renderedImage[out_idx + 1] = original_colors[orig_idx + 1];
                renderedImage[out_idx + 2] = original_colors[orig_idx + 2];
            } else {
                renderedImage[out_idx + 0] = 0;
                renderedImage[out_idx + 1] = 0;
                renderedImage[out_idx + 2] = 0;
            }
        }
    }

    if (visOpt == VisOpt::regions)
        std::fill(renderedImage.begin(), renderedImage.end(), 200);

    // grid_schematic must show pure schematic colors everywhere, with no
    // natural-image pixels left showing through. The per-cell loop below
    // only covers [grid_i_min,grid_i_max]x[grid_j_min,grid_j_max] -- the
    // overlap between the requested sub-region and the MPM grid rectangle
    // (gx x gy), which is smaller than the full image (width x height) the
    // background fill above just painted with natural colors. Pixels
    // outside the grid rectangle have no landmask_buffer entry at all
    // (trimmed from the modeled domain), so pre-fill the whole canvas with
    // the open-boundary color here; the loop then overwrites every pixel
    // that *is* covered by the grid with its real classification
    // (ice/water/land/open-boundary).
    if (visOpt == VisOpt::grid_schematic) {
        for (size_t px = 0; px + 2 < renderedImage.size(); px += 3) {
            renderedImage[px + 0] = ColorMap::rgb_land[0];
            renderedImage[px + 1] = ColorMap::rgb_land[1];
            renderedImage[px + 2] = ColorMap::rgb_land[2];
        }
    }

    const float *ptr_density = GetGridBufferPointer(
        SimParams::HostGridArrayIndex::grid_idx_vis_pts_density);
    const float *ptr_crushed =
        GetGridBufferPointer(SimParams::HostGridArrayIndex::grid_idx_vis_crushed);
    const float *ptr_r =
        GetGridBufferPointer(SimParams::HostGridArrayIndex::grid_idx_vis_r);
    const float *ptr_g =
        GetGridBufferPointer(SimParams::HostGridArrayIndex::grid_idx_vis_g);
    const float *ptr_b =
        GetGridBufferPointer(SimParams::HostGridArrayIndex::grid_idx_vis_b);

    const float *ptr_mass =
        GetGridBufferPointer(SimParams::HostGridArrayIndex::host_grid_idx_mass);
    const float *ptr_Jpinv =
        GetGridBufferPointer(SimParams::HostGridArrayIndex::grid_idx_vis_Jpinv);
    const float *ptr_P =
        GetGridBufferPointer(SimParams::HostGridArrayIndex::grid_idx_vis_P);
    const float *ptr_Q =
        GetGridBufferPointer(SimParams::HostGridArrayIndex::grid_idx_vis_Q);
    const float *ptr_px =
        GetGridBufferPointer(SimParams::HostGridArrayIndex::grid_idx_px);
    const float *ptr_py =
        GetGridBufferPointer(SimParams::HostGridArrayIndex::grid_idx_py);
    const float *ptr_cracked =
        GetGridBufferPointer(SimParams::HostGridArrayIndex::grid_idx_vis_cracked);

    const float *ptr_frac_tension = GetGridBufferPointer(
        SimParams::HostGridArrayIndex::grid_idx_fracture_tension);
    const float *ptr_frac_shear = GetGridBufferPointer(
        SimParams::HostGridArrayIndex::grid_idx_fracture_shear);
    const float *ptr_frac_crush = GetGridBufferPointer(
        SimParams::HostGridArrayIndex::grid_idx_fracture_crush);

    const float *ptr_strain_eqv = GetGridBufferPointer(
        SimParams::HostGridArrayIndex::grid_idx_vis_strain_EqvGreenLagrange);
    const float *ptr_strain_vm = GetGridBufferPointer(
        SimParams::HostGridArrayIndex::grid_idx_vis_strain_vonMises);
    const float *ptr_thickness = GetGridBufferPointer(
        SimParams::HostGridArrayIndex::grid_idx_vis_thickness);
    const float *ptr_ice_strength = GetGridBufferPointer(
        SimParams::HostGridArrayIndex::grid_idx_vis_ice_strength);
    const float *ptr_gamma_p = GetGridBufferPointer(
        SimParams::HostGridArrayIndex::grid_idx_vis_gamma_p);

    // Calculate grid iteration bounds that overlap with the sub-region
    int grid_i_min = std::max(0, offsetX - ox);
    int grid_i_max = std::min(gx - 1, offsetX + sizeX - 1 - ox);
    int grid_j_min = std::max(0, offsetY - oy);
    int grid_j_max = std::min(gy - 1, offsetY + sizeY - 1 - oy);

#pragma omp parallel for
    for (int i = grid_i_min; i <= grid_i_max; i++) {
        for (int j = grid_j_min; j <= grid_j_max; j++) {
            const size_t grid_idx = (size_t)j + (size_t)i * gy;

            // Map to sub-region indices
            int sub_x = (i + ox) - offsetX;
            int sub_y = (j + oy) - offsetY;
            if (sub_x < 0 || sub_x >= sizeX || sub_y < 0 || sub_y >= sizeY)
                continue;

            const size_t render_idx = ((size_t)sub_x + (size_t)sub_y * sizeX) * 3;

            if (grid_status[grid_idx] == SimParams::MAX_REGIONS) {
                // Modeled area
                double val_pt_density = 0, val_crushed = 0;
                std::array<uint8_t, 3> _rgb = {0, 0, 0};
                float alpha = 0.0f;

                if (ptr_density)
                    val_pt_density = ptr_density[grid_idx];
                if (ptr_crushed)
                    val_crushed = ptr_crushed[grid_idx];

                // Determine base color and alpha
                if (frame_rgba.size() == gridSize * 4) {
                    _rgb[0] = frame_rgba[grid_idx * 4 + 0];
                    _rgb[1] = frame_rgba[grid_idx * 4 + 1];
                    _rgb[2] = frame_rgba[grid_idx * 4 + 2];
                    alpha = frame_rgba[grid_idx * 4 + 3] / 255.0f;
                } else if (ptr_r && ptr_g && ptr_b) {
                    float vr = ptr_r[grid_idx];
                    float vg = ptr_g[grid_idx];
                    float vb = ptr_b[grid_idx];
                    _rgb[0] = (uint8_t)(std::clamp(vr, 0.f, 1.f) * 255);
                    _rgb[1] = (uint8_t)(std::clamp(vg, 0.f, 1.f) * 255);
                    _rgb[2] = (uint8_t)(std::clamp(vb, 0.f, 1.f) * 255);

                    alpha = std::min(val_pt_density * (2.0 / 5.0), 1.0);
                }

                std::array<uint8_t, 3> c =
                    ColorMap::mergeColors(ColorMap::rgb_water, _rgb, alpha);

                auto set_pixel = [&](const std::array<uint8_t, 3> &col) {
                    renderedImage[render_idx + 0] = col[0];
                    renderedImage[render_idx + 1] = col[1];
                    renderedImage[render_idx + 2] = col[2];
                };

                auto blend_and_set = [&](const std::array<uint8_t, 3> &col, float mix,
                                         bool with_water = false) {
                    auto c2 = ColorMap::mergeColors(c, col, mix);
                    if (with_water)
                        c2 = ColorMap::mergeColors(ColorMap::rgb_water, c2, alpha);
                    set_pixel(c2);
                };

                // Uniform color-fraction mapping: frac=0 at range_from, frac=1
                // at range_to (ColorMap::getColor clamps outside [0,1]).
                auto frac = [&](float val) {
                    return (val - range_from) / (range_to - range_from);
                };
                auto calc_mix = [&](float val) {
                    return alpha * (std::abs(val) / mix_scale + (1. - transparency));
                };

                if (visOpt == VisOpt::regions) {
                    set_pixel({100, 150, 200});
                } else if (visOpt == VisOpt::grid_colors) {
                    set_pixel(c);
                } else if (visOpt == VisOpt::grid_mass) {
                    if (!ptr_mass)
                        continue;
                    blend_and_set(colormap.getColor(ColorMap::Palette::ANSYS,
                                                    frac(ptr_mass[grid_idx])),
                                  alpha * (1. - transparency), true);
                } else if (visOpt == VisOpt::grid_pt_count) {
                    if (!ptr_density)
                        continue;
                    set_pixel(colormap.getColor(ColorMap::Palette::ANSYS,
                                                frac(ptr_density[grid_idx])));
                } else if (visOpt == VisOpt::grid_Jpinv) {
                    if (!ptr_Jpinv)
                        continue;
                    float val = ptr_Jpinv[grid_idx] - 1.0f;
                    blend_and_set(colormap.getColor(ColorMap::Palette::Pressure,
                                                    frac(val)),
                                  calc_mix(val), true);
                } else if (visOpt == VisOpt::grid_P) {
                    if (!ptr_P)
                        continue;
                    float val = ptr_P[grid_idx];
                    blend_and_set(colormap.getColor(ColorMap::Palette::Pressure,
                                                    frac(val)),
                                  calc_mix(val), true);
                } else if (visOpt == VisOpt::grid_Q) {
                    if (!ptr_Q)
                        continue;
                    float val = ptr_Q[grid_idx];
                    blend_and_set(
                        colormap.getColor(ColorMap::Palette::ANSYS, frac(val)),
                        calc_mix(val), true);
                } else if (visOpt == VisOpt::grid_vnorm) {
                    if (!ptr_px || !ptr_py)
                        continue;
                    float vx = ptr_px[grid_idx], vy = ptr_py[grid_idx];
                    blend_and_set(colormap.getColor(ColorMap::Palette::ANSYS,
                                                    frac(std::sqrt(vx * vx + vy * vy))),
                                  alpha * (1. - transparency), true);
                } else if (visOpt == VisOpt::grid_cracked) {
                    if (!ptr_cracked)
                        continue;
                    float val_cracked = ptr_cracked[grid_idx];
                    float val_crushed = ptr_crushed ? ptr_crushed[grid_idx] : 0.0f;

                    auto combined_color = c;
                    if (val_cracked > 0.0)
                        combined_color =
                            ColorMap::mergeColors(combined_color, ColorMap::rgb_green,
                                                  std::min(1.0f, val_cracked));
                    if (val_crushed > 0.0)
                        combined_color = ColorMap::mergeColors(
                            combined_color, ColorMap::rgb_red, std::min(1.0f, val_crushed));

                    set_pixel(ColorMap::mergeColors(ColorMap::rgb_water, combined_color,
                                                    alpha));
                } else if (visOpt == VisOpt::grid_fracture_type) {

                    if (!ptr_frac_tension || !ptr_frac_shear || !ptr_frac_crush) continue;
                    float val_tension = std::clamp(ptr_frac_tension[grid_idx], 0.0f, 1.0f);
                    float val_shear = std::clamp(ptr_frac_shear[grid_idx], 0.0f, 1.0f);
                    float val_crush = std::clamp(ptr_frac_crush[grid_idx], 0.0f, 1.0f);

                    float fracture_intensity = std::max({val_tension, val_shear, val_crush});

                    std::array<uint8_t, 3> darkened_ice;
                    for (int k = 0; k < 3; ++k) {
                        darkened_ice[k] = static_cast<uint8_t>(c[k] * std::max(0.0, 1.0 - transparency * fracture_intensity));
                    }

                    if (val_crush > 0.0f) {
                        darkened_ice = ColorMap::mergeColors(darkened_ice, ColorMap::rgb_red, val_crush);
                    }

                    // Jp_inv is ~1.0 for intact material. As it drops below 1.0 (dilation/melt),
                    // fade the material toward open water: fully opaque at Jp_inv >= 1.0,
                    float jpinv_alpha = 1.0f;
                    if (ptr_Jpinv) {
                        float val_jpinv = ptr_Jpinv[grid_idx];
                        jpinv_alpha = std::clamp((val_jpinv - 0.75f) / 0.25f, 0.0f, 1.0f);
                    }

                    set_pixel(ColorMap::mergeColors(ColorMap::rgb_water, darkened_ice, jpinv_alpha));

                } else if (visOpt == VisOpt::grid_schematic) {
                    // Same fracture-darkening logic as grid_fracture_type, but with
                    // pure schematic colors instead of the natural satellite-image
                    // color: white ice (darkened to grey/red by fracture state),
                    // black open water. Land/open-boundary colors are set in the
                    // "else" branch below (grid_status != MAX_REGIONS).
                    if (!ptr_frac_tension || !ptr_frac_shear || !ptr_frac_crush) continue;
                    float val_tension = std::clamp(ptr_frac_tension[grid_idx], 0.0f, 1.0f);
                    float val_shear = std::clamp(ptr_frac_shear[grid_idx], 0.0f, 1.0f);
                    float val_crush = std::clamp(ptr_frac_crush[grid_idx], 0.0f, 1.0f);

                    float fracture_intensity = std::max({val_tension, val_shear, val_crush});

                    constexpr std::array<uint8_t, 3> rgb_ice_pure = {255, 255, 255};
                    std::array<uint8_t, 3> darkened_ice;
                    for (int k = 0; k < 3; ++k) {
                        darkened_ice[k] = static_cast<uint8_t>(rgb_ice_pure[k] * std::max(0.0, 1.0 - transparency * fracture_intensity));
                    }

                    if (val_crush > 0.0f) {
                        darkened_ice = ColorMap::mergeColors(darkened_ice, ColorMap::rgb_red, val_crush);
                    }

                    float jpinv_alpha = 1.0f;
                    if (ptr_Jpinv) {
                        float val_jpinv = ptr_Jpinv[grid_idx];
                        jpinv_alpha = std::clamp((val_jpinv - 0.75f) / 0.25f, 0.0f, 1.0f);
                    }

                    constexpr std::array<uint8_t, 3> rgb_water_pure = {0, 0, 0};
                    set_pixel(ColorMap::mergeColors(rgb_water_pure, darkened_ice, jpinv_alpha));

                } else if (visOpt == VisOpt::str_EqvGreenLagrange) {
                    if (!ptr_strain_eqv)
                        continue;
                    float val = ptr_strain_eqv[grid_idx];
                    blend_and_set(
                        colormap.getColor(ColorMap::Palette::ANSYS, frac(val)),
                        calc_mix(val));
                } else if (visOpt == VisOpt::str_vonMises) {
                    if (!ptr_strain_vm)
                        continue;
                    float val = ptr_strain_vm[grid_idx];
                    blend_and_set(
                        colormap.getColor(ColorMap::Palette::ANSYS, frac(val)),
                        calc_mix(val));
                } else if (visOpt == VisOpt::grid_thickness) {
                    if (!ptr_thickness)
                        continue;
                    float val = ptr_thickness[grid_idx];
                    blend_and_set(
                        colormap.getColor(ColorMap::Palette::Ice, frac(val)),
                        calc_mix(val));
                } else if (visOpt == VisOpt::grid_ice_strength) {
                    if (!ptr_ice_strength)
                        continue;
                    float val = ptr_ice_strength[grid_idx];
                    blend_and_set(
                        colormap.getColor(ColorMap::Palette::ANSYS, frac(val)),
                        calc_mix(val));
                } else if (visOpt == VisOpt::grid_gamma_p) {
                    if (!ptr_gamma_p)
                        continue;
                    float val = ptr_gamma_p[grid_idx];
                    blend_and_set(
                        colormap.getColor(ColorMap::Palette::ANSYS, frac(val)),
                        calc_mix(val));
                } else if (visOpt == VisOpt::glo12_ocean) {
                    auto [uv, vv] = currentInterp.GetOceanValue(i, j);
                    float val = std::sqrt(uv * uv + vv * vv);
                    set_pixel(colormap.getColor(ColorMap::Palette::ANSYS, frac(val)));
                } else if (visOpt == VisOpt::carra1_wind) {
                    auto [uv, vv] = windInterp.GetWindValue(i, j);
                    float val = std::sqrt(uv * uv + vv * vv);
                    set_pixel(colormap.getColor(ColorMap::Palette::ANSYS, frac(val)));
                } else if (visOpt == VisOpt::carra1_temperature) {
                    // CARRA1 gives Kelvin; convert to Celsius purely for
                    // visualization (VisOpt::range_from/to for this option
                    // are expressed in Celsius).
                    float val = windInterp.GetTemperatureValue(i, j) - 273.15f;
                    set_pixel(colormap.getColor(ColorMap::Palette::ANSYS, frac(val)));
                } else if (visOpt == VisOpt::v_glo12_thickness) {
                    float val = currentInterp.GetThicknessValue(i, j);
                    set_pixel(colormap.getColor(ColorMap::Palette::ANSYS, frac(val)));
                }
            } else {
                if (visOpt == VisOpt::regions) {
                    uint8_t region_id = grid_status[grid_idx];
                    float val = (region_id % 13) / 12.0f;
                    std::array<uint8_t, 3> c =
                        colormap.getColor(ColorMap::Palette::Pastel, val);
                    for (int k = 0; k < 3; k++)
                        renderedImage[render_idx + k] = c[k];
                } else if (visOpt == VisOpt::grid_schematic) {
                    uint8_t region_id = grid_status[grid_idx];
                    std::array<uint8_t, 3> c = ColorMap::rgb_land;
                    for (int k = 0; k < 3; k++)
                        renderedImage[render_idx + k] = c[k];
                }
            }
        }
    }
}

void HostSideData::RenderAsJPG(int frameIndex, const VisOpt::Type visOpt,
                               int offsetX, int offsetY, int sizeX, int sizeY,
                               std::string outPrefix, int finalW, int finalH) {
  std::vector<uint8_t> renderedImage;
  RenderGridRaster(visOpt, renderedImage, offsetX, offsetY, sizeX, sizeY);

  int frame = frameIndex;
  std::string visName = VisOpt::descriptions.at(visOpt).first;

  // determine output directory
  std::string out_dir = output_directory.empty() ? "output" : output_directory;
  fs::path targetPath(out_dir);
  fs::path framesDir = targetPath / outPrefix / visName;
  fs::create_directories(framesDir);

  std::string filename =
      (framesDir / fmt::format("{:05d}.jpg", frame)).string();

  int imgW = (sizeX == -1) ? prms.InitializationImageSizeX : sizeX;
  int imgH = (sizeY == -1) ? prms.InitializationImageSizeY : sizeY;

  if (finalW == -1)
    finalW = imgW;
  if (finalH == -1)
    finalH = imgH;

  // Flip vertically before saving because RenderGridRaster produces row 0 at
  // the bottom (scientific coords) but JPG expects row 0 at the top.
  stbi_flip_vertically_on_write(1);

  std::vector<uint8_t> resizedImage;
  if (finalW == imgW && finalH == imgH) {
    resizedImage = std::move(renderedImage);
  } else {
    resizedImage.resize(finalW * finalH * 3);
    stbir_resize_uint8_linear(renderedImage.data(), imgW, imgH, imgW * 3,
                              resizedImage.data(), finalW, finalH, finalW * 3,
                              STBIR_RGB);
  }

  // Save as JPG using stb_image_write
  stbi_write_jpg(filename.c_str(), finalW, finalH, 3, resizedImage.data(), 90);
  LOGR("Saved sub-region JPG to {} ({}x{})", filename, finalW, finalH);
}

void HostSideData::generate_ffmpeg_script(
    int frameFrom, int frameTo, const std::string &dirName,
    const std::string &currentFrameDirectory,
    const std::vector<VisOpt::Type> &visOptsToRender) {
  LOGR("Generating ffmpeg script for frames {} to {} in {}", frameFrom, frameTo,
       dirName);

  const int totalFrames = (frameTo - frameFrom) + 1;
  if (totalFrames <= 0)
    return;

  namespace fs = std::filesystem;
  fs::path frameDir(currentFrameDirectory);
  fs::path rasterPath = frameDir.parent_path() / dirName;

  if (!fs::exists(rasterPath)) {
    fs::create_directories(rasterPath);
  }

  fs::path scriptFilename = rasterPath / "genvideo.sh";

  std::ofstream scriptFile(scriptFilename.string());
  if (!scriptFile.is_open()) {
    LOGR("Failed to open script file for writing: {}", scriptFilename.string());
    return;
  }

  scriptFile << "#!/bin/bash\n";
  scriptFile << "# This script will generate mp4 videos from the rendered JPG "
                "frames.\n";
  scriptFile << "# It is designed to be run from within the '" << dirName
             << "' directory.\n";
  scriptFile << "cd \"$(dirname \"$0\")\"\n\n";

  scriptFile << "FPS=30\n";
  scriptFile << "START_FRAME=" << frameFrom << "\n";
  scriptFile << "LAST_FRAME=" << frameTo << "\n";
  scriptFile << "NUM_FRAMES=$((LAST_FRAME - START_FRAME + 1))\n\n";

  const std::string fmtStr =
      R"(ffmpeg -y -r $FPS -f image2 -start_number $START_FRAME -i "{0:}" -vframes $NUM_FRAMES -vcodec libx264 -vf "scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:-1:-1:white" -crf 21 -pix_fmt yuv420p "{1:}")";

  for (const auto &visOpt : visOptsToRender) {
    const std::string visName = VisOpt::descriptions.at(visOpt).first;
    const std::string inputFilePattern = visName + "/%05d.jpg";
    const std::string outputVideoFile = visName + ".mp4";

    std::string command =
        fmt::format(fmt::runtime(fmtStr), inputFilePattern, outputVideoFile);

    scriptFile << "# Generate video for " << visName << "\n";
    scriptFile << command << "\n\n";
  }

  scriptFile.close();
  int result = std::system(("chmod +x " + scriptFilename.string()).c_str());
  (void)result;
}

// currently unused code

/*
void HostSideData::SaveForces(const int frame)
{
    fs::path targetPath;
    if (!output_directory.empty()) {
        targetPath = output_directory;
    } else {
        targetPath = "output";
    }
    fs::path framesDir = targetPath / "frames";
    fs::create_directories(framesDir);

    // save forces
    fs::path fullPathForces = framesDir / "forces.h5";
    bool file_exists = std::filesystem::exists(fullPathForces);
    H5::H5File file_forces(fullPathForces.string(), file_exists ? H5F_ACC_RDWR :
H5F_ACC_TRUNC); H5::DataSet ds_forces;

    if(!file_exists)
    {
        hsize_t initial_dims[3] = {0, (hsize_t)SimParams::MAX_REGIONS, 2};
        hsize_t max_dims[3] = {H5S_UNLIMITED, (hsize_t)SimParams::MAX_REGIONS,
2}; H5::DataSpace file_dataspace_for_creation(3, initial_dims, max_dims);

        H5::DSetCreatPropList dcpl;
        hsize_t chunk_dims[3] = {1, (hsize_t)SimParams::MAX_REGIONS, 2};
        dcpl.setChunk(3, chunk_dims);
        ds_forces = file_forces.createDataSet("ds_forces",
H5::PredType::NATIVE_DOUBLE, file_dataspace_for_creation, dcpl);

        H5::DataSpace scalar_space(H5S_SCALAR);
        ds_forces.createAttribute("cellsize", H5::PredType::NATIVE_DOUBLE,
scalar_space) .write(H5::PredType::NATIVE_DOUBLE, &prms.cellsize);
        ds_forces.createAttribute("InitialTimeStep",
H5::PredType::NATIVE_DOUBLE, scalar_space) .write(H5::PredType::NATIVE_DOUBLE,
&prms.InitialTimeStep); ds_forces.createAttribute("AnimationFramePeriod",
H5::PredType::NATIVE_DOUBLE, scalar_space) .write(H5::PredType::NATIVE_DOUBLE,
&prms.AnimationFramePeriod);
    }
    else
    {
        ds_forces = file_forces.openDataSet("ds_forces");
    }

    // Get current dataspace and dimensions
    H5::DataSpace file_space = ds_forces.getSpace();
    hsize_t current_dims_on_file[3];
    file_space.getSimpleExtentDims(current_dims_on_file);

    // Extend if needed
    hsize_t required_frame_capacity = static_cast<hsize_t>(frame) + 1;
    if (required_frame_capacity > current_dims_on_file[0]) {
        hsize_t new_dims[3] = {required_frame_capacity,
static_cast<hsize_t>(SimParams::MAX_REGIONS), 2}; ds_forces.extend(new_dims);
        file_space = ds_forces.getSpace();
    }

    // Define hyperslab
    hsize_t offset[3] = {static_cast<hsize_t>(frame), 0, 0};
    hsize_t slab_dims[3] = {1, static_cast<hsize_t>(SimParams::MAX_REGIONS), 2};
    file_space.selectHyperslab(H5S_SELECT_SET, slab_dims, offset);

    H5::DataSpace memory_space(3, slab_dims);

    // Write the data
    ds_forces.write(grid_forces_summary_per_region.data(),
H5::PredType::NATIVE_DOUBLE, memory_space, file_space);
}
*/

void HostSideData::AppendAverageIceTemperature(long long epochTimestamp, double avgTempC,
                                                unsigned sampleCount, const std::string& phase)
{
    fs::path csvPath = fs::path(data_directory) / "average_ice_temperature.csv";
    bool writeHeader = !fs::exists(csvPath);

    std::ofstream f(csvPath, std::ios::app);
    if (writeHeader) f << "timestamp,avg_temperature_C,sample_count,phase\n";
    f << epochTimestamp << "," << std::setprecision(6) << avgTempC << ","
      << sampleCount << "," << phase << "\n";
}
