// visual_representation.cpp

#include "visual_representation.h"
#include "host_side_data.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <ctime>
#include <fstream>
#include <map>
#include <spdlog/spdlog.h>

#include <vtkCellArray.h>
#include <vtkContourFilter.h>
#include <vtkCoordinate.h>
#include <vtkPointData.h>
#include <vtkPolyDataMapper2D.h>
#include <vtkProperty.h>
#include <vtkProperty2D.h>
#include <vtkTextProperty.h>

VisualRepresentation::VisualRepresentation(HostSideData &data) : hsd(data) {
  LOGR("VisualRepresentation constructor");

  populateLut(ColorMap::Palette::Pressure, lut_Pressure);
  populateLut(ColorMap::Palette::P2, lut_P2);
  populateLut(ColorMap::Palette::ANSYS, lut_ANSYS);
  populateLut(ColorMap::Palette::Ice, lut_Ice);

  pts_colors->SetNumberOfComponents(3);
  pts_colors->SetName("pts_colors");
  points_polydata->SetPoints(points);
  points_polydata->GetPointData()->AddArray(pts_colors);
  points_filter->SetInputData(points_polydata);
  points_filter->Update();
  points_mapper->SetInputData(points_filter->GetOutput());
  actor_points->SetMapper(points_mapper);
  actor_points->GetProperty()->SetPointSize(2);
  actor_points->GetProperty()->SetVertexColor(1, 0, 0);
  actor_points->GetProperty()->SetColor(151. / 255, 188. / 255, 215. / 255);
  actor_points->GetProperty()->LightingOff();
  actor_points->GetProperty()->ShadingOff();
  actor_points->GetProperty()->SetInterpolationToFlat();
  actor_points->PickableOff();

  scalarBar->SetMaximumWidthInPixels(150);
  scalarBar->SetBarRatio(0.1);
  scalarBar->SetMaximumHeightInPixels(200);
  scalarBar->GetPositionCoordinate()->SetCoordinateSystemToNormalizedDisplay();
  scalarBar->GetPositionCoordinate()->SetValue(0.01, 0.015, 0.0);
  scalarBar->SetLabelFormat("%.1e");
  scalarBar->GetLabelTextProperty()->BoldOff();
  scalarBar->GetLabelTextProperty()->ItalicOff();
  scalarBar->GetLabelTextProperty()->ShadowOff();
  scalarBar->GetLabelTextProperty()->SetColor(0.1, 0.1, 0.1);

  vtkTextProperty *scalarBarTitleProp = scalarBar->GetTitleTextProperty();
  scalarBarTitleProp->ShadowOff();
  scalarBarTitleProp->SetColor(0.1, 0.1, 0.1);
  scalarBarTitleProp->BoldOff();
  scalarBarTitleProp->ItalicOff();

  vtkTextProperty *txtprop = actorText->GetTextProperty();
  txtprop->SetFontFamilyToArial();
  txtprop->BoldOn();
  txtprop->SetFontSize(30);
  txtprop->ShadowOff();
  txtprop->SetColor(0.1, 0.1, 0.1);
  actorText->SetDisplayPosition(1600, 10);

  vtkTextProperty *titleTextProp = actorTextTitle->GetTextProperty();
  titleTextProp->SetFontFamilyToArial();
  titleTextProp->BoldOn();
  titleTextProp->SetFontSize(30);
  titleTextProp->ShadowOff();
  titleTextProp->SetColor(0.1, 0.1, 0.1);
  titleTextProp->SetJustificationToCentered();
  actorTextTitle->SetDisplayPosition(800, 10);

  // Text background
  vtkNew<vtkPoints> textBgPoints;
  textBgPoints->InsertNextPoint(580, 5, 0);
  textBgPoints->InsertNextPoint(1910, 5, 0);
  textBgPoints->InsertNextPoint(1910, 60, 0);
  textBgPoints->InsertNextPoint(580, 60, 0);

  vtkNew<vtkCellArray> textBgPoly;
  vtkIdType textIds[4] = {0, 1, 2, 3};
  textBgPoly->InsertNextCell(4, textIds);

  vtkNew<vtkPolyData> textBgPolyData;
  textBgPolyData->SetPoints(textBgPoints);
  textBgPolyData->SetPolys(textBgPoly);

  vtkNew<vtkPolyDataMapper2D> textBgMapper;
  textBgMapper->SetInputData(textBgPolyData);

  textBgActor->SetMapper(textBgMapper);
  textBgActor->GetProperty()->SetColor(1.0, 1.0, 1.0);
  textBgActor->GetProperty()->SetOpacity(0.8);
  textBgActor->SetLayerNumber(2);

  // Absolute date text + background, upper-left corner. Uses normalized
  // viewport coordinates throughout (not fixed pixel offsets like actorText/
  // textBgActor above) so it stays anchored to the corner -- and therefore
  // stays visible/in-frame -- regardless of window size or crop.
  vtkTextProperty *dateTextProp = actorDateText->GetTextProperty();
  dateTextProp->SetFontFamilyToArial();
  dateTextProp->BoldOn();
  dateTextProp->SetFontSize(28);
  dateTextProp->ShadowOff();
  dateTextProp->SetColor(0.1, 0.1, 0.1);
  dateTextProp->SetJustificationToLeft();
  dateTextProp->SetVerticalJustificationToCentered();
  actorDateText->GetPositionCoordinate()->SetCoordinateSystemToNormalizedViewport();
  actorDateText->SetPosition(0.015, 0.965);

  vtkNew<vtkCoordinate> dateBgCoord;
  dateBgCoord->SetCoordinateSystemToNormalizedViewport();

  vtkNew<vtkPoints> dateBgPoints;
  dateBgPoints->InsertNextPoint(0.005, 0.94, 0);
  dateBgPoints->InsertNextPoint(0.22, 0.94, 0);
  dateBgPoints->InsertNextPoint(0.22, 0.99, 0);
  dateBgPoints->InsertNextPoint(0.005, 0.99, 0);

  vtkNew<vtkCellArray> dateBgPoly;
  vtkIdType dateBgIds[4] = {0, 1, 2, 3};
  dateBgPoly->InsertNextCell(4, dateBgIds);

  vtkNew<vtkPolyData> dateBgPolyData;
  dateBgPolyData->SetPoints(dateBgPoints);
  dateBgPolyData->SetPolys(dateBgPoly);

  vtkNew<vtkPolyDataMapper2D> dateBgMapper;
  dateBgMapper->SetInputData(dateBgPolyData);
  dateBgMapper->SetTransformCoordinate(dateBgCoord);

  dateBgActor->SetMapper(dateBgMapper);
  dateBgActor->GetProperty()->SetColor(1.0, 1.0, 1.0);
  dateBgActor->GetProperty()->SetOpacity(0.8);
  dateBgActor->SetLayerNumber(2);

  // Scalar bar background
  vtkNew<vtkPoints> sbBgPoints;
  sbBgPoints->InsertNextPoint(10, 10, 0);
  sbBgPoints->InsertNextPoint(200, 10, 0);
  sbBgPoints->InsertNextPoint(200, 280, 0);
  sbBgPoints->InsertNextPoint(10, 280, 0);

  vtkNew<vtkCellArray> sbBgPoly;
  vtkIdType sbIds[4] = {0, 1, 2, 3};
  sbBgPoly->InsertNextCell(4, sbIds);

  vtkNew<vtkPolyData> sbBgPolyData;
  sbBgPolyData->SetPoints(sbBgPoints);
  sbBgPolyData->SetPolys(sbBgPoly);

  vtkNew<vtkPolyDataMapper2D> sbBgMapper;
  sbBgMapper->SetInputData(sbBgPolyData);

  scalarBarBgActor->SetMapper(sbBgMapper);
  scalarBarBgActor->GetProperty()->SetColor(1.0, 1.0, 1.0);
  scalarBarBgActor->GetProperty()->SetOpacity(0.8);
  scalarBarBgActor->SetLayerNumber(2);

  // Flow Visualization Infrastructure
  flow_vectors->SetNumberOfComponents(3);
  flow_vectors->SetName("flow_vectors");
  flow_grid->GetPointData()->SetVectors(flow_vectors);

  // Hedgehog Overlay
  hedgehog_flow_vectors->SetNumberOfComponents(3);
  hedgehog_flow_vectors->SetName("hedgehog_flow_vectors");
  hedgehog_flow_grid->GetPointData()->SetVectors(hedgehog_flow_vectors);

  hedgehog_filter->SetInputData(hedgehog_flow_grid);
  hedgehog_filter->SetVectorModeToUseVector();
  hedgehog_filter->SetScaleFactor(1.0);
  hedgehog_mapper->SetInputConnection(hedgehog_filter->GetOutputPort());
  hedgehog_actor_vectors->SetMapper(hedgehog_mapper);
  hedgehog_actor_vectors->GetProperty()->SetColor(0.0, 0.0,
                                                  0.0); // Black vectors
  hedgehog_actor_vectors->GetProperty()->SetLineWidth(2.0);
  hedgehog_actor_vectors->VisibilityOff();

  actorText->SetLayerNumber(1);
  actorTextTitle->SetLayerNumber(1);
  actorDateText->SetLayerNumber(1);
  scalarBar->SetLayerNumber(1);

  VisOpt::LoadVisualizationState();
  LOGR("VisualRepresentation constructor done");
}

void VisualRepresentation::ChangeVisualizationOption(int option) {
  LOGR("VisualRepresentation::ChangeVisualizationOption {}", option);
  VisualizingVariable = (VisOpt::Type)option;
  SynchronizeTopology();
}

void VisualRepresentation::populateLut(ColorMap::Palette palette,
                                       vtkNew<vtkLookupTable> &table) {
  const std::vector<Eigen::Vector3f> &colorTable =
      ColorMap::getColorTable(palette);
  int size = static_cast<int>(colorTable.size());
  if (size < 2) {
    return;
  }

  const int m = 256;
  table->SetNumberOfTableValues(m);
  table->Build();

  for (int i = 0; i < m; ++i) {
    float t = static_cast<float>(i) / (m - 1);
    float scaledT = t * (size - 1);
    int lowerIdx = static_cast<int>(std::floor(scaledT));
    int upperIdx = static_cast<int>(std::ceil(scaledT));
    float localT = scaledT - lowerIdx;
    const Eigen::Vector3f &lowerColor = colorTable[lowerIdx];
    const Eigen::Vector3f &upperColor = colorTable[upperIdx];
    Eigen::Vector3f interpolatedColor =
        (1.0f - localT) * lowerColor + localT * upperColor;
    table->SetTableValue(i, interpolatedColor[0], interpolatedColor[1],
                         interpolatedColor[2], 1.0);
  }
}

void VisualRepresentation::SynchronizeTopology() {
  LOGR("VisualRepresentation::SynchronizeTopology(): {}",
       (int)VisualizingVariable);

  const SimParams &prms = hsd.prms;
  const std::vector<uint8_t> &grid_status = hsd.landmask_buffer;
  const std::vector<uint8_t> &original_colors = hsd.original_image_colors_rgb;

  const int width = prms.InitializationImageSizeX;
  const int height = prms.InitializationImageSizeY;
  const double h = prms.cellsize;

  // Update raster image
  hsd.RenderGridRaster(VisualizingVariable, renderedImage, 0, 0, width, height);

  // Update VTK raster image
  raster_scalars->SetNumberOfComponents(3);
  raster_scalars->SetArray(renderedImage.data(), renderedImage.size(), 1);
  raster_scalars->Modified();
  raster_imageData->SetDimensions(width, height, 1);
  raster_imageData->GetPointData()->SetScalars(raster_scalars);

  raster_plane->SetOrigin(-h / 2, -h / 2, -1.0);
  raster_plane->SetPoint1((width - 0.5) * h, -h / 2, -1.0);
  raster_plane->SetPoint2(-h / 2, (height - 0.5) * h, -1.0);

  raster_mapper->SetInputConnection(raster_plane->GetOutputPort());
  raster_texture->SetInputData(raster_imageData);
  raster_actor->SetMapper(raster_mapper);
  raster_actor->SetTexture(raster_texture);
  raster_mapper->Update();
  raster_texture->Update();

  UpdateFlowData();
  SynchronizeValues();
  ConfigureScalarBar();
  UpdateTimeText();
}

void VisualRepresentation::UpdateFlowData() {
  const SimParams &prms = hsd.prms;
  const int width = prms.InitializationImageSizeX;
  const int height = prms.InitializationImageSizeY;
  const int ox = prms.ModeledRegionOffsetX;
  const int oy = prms.ModeledRegionOffsetY;
  const int gx = prms.GridXTotal;
  const int gy = prms.GridYTotal;
  const double h = prms.cellsize;

  // Flow Vis Logic
  bool show_hedgehog = (HedgehogVariable != VisOpt::hedgehog_none);

  auto update_grid_and_vectors = [&](vtkStructuredGrid *grid,
                                     vtkFloatArray *vectors, bool sample_ocean,
                                     bool sample_wind, double z_pos) {
    int vis_gx = std::min(gx, 2000);
    int vis_gy = (gx > 0) ? (int)((long long)gy * vis_gx / gx) : 0;
    vis_gx = std::max(vis_gx, 2);
    vis_gy = std::max(vis_gy, 2);

    int dims[3];
    grid->GetDimensions(dims);
    if (dims[0] != vis_gx || dims[1] != vis_gy) {
      grid->SetDimensions(vis_gx, vis_gy, 1);
      vectors->SetNumberOfTuples(vis_gx * vis_gy);

      vtkNew<vtkPoints> gp;
      gp->SetNumberOfPoints(vis_gx * vis_gy);
      for (int j = 0; j < vis_gy; j++) {
        for (int i = 0; i < vis_gx; i++) {
          double r_i = (double)i / (vis_gx - 1);
          double r_j = (double)j / (vis_gy - 1);
          double phys_x = (ox + r_i * (gx - 1)) * h;
          double phys_y = (oy + r_j * (gy - 1)) * h;
          vtkIdType vtk_idx = i + j * vis_gx;
          gp->SetPoint(vtk_idx, phys_x, phys_y, z_pos);
        }
      }
      grid->SetPoints(gp);
    }

#pragma omp parallel for
    for (int i = 0; i < vis_gx; i++) {
      for (int j = 0; j < vis_gy; j++) {
        int orig_i = (int)((double)i / (vis_gx - 1) * (gx - 1));
        int orig_j = (int)((double)j / (vis_gy - 1) * (gy - 1));

        if (orig_i >= gx)
          orig_i = gx - 1;
        if (orig_j >= gy)
          orig_j = gy - 1;

        float vx = 0, vy = 0;
        if (sample_ocean) {
          auto [uv, vv] = hsd.currentInterp.GetOceanValue(orig_i, orig_j);
          vx = uv;
          vy = vv;
        } else if (sample_wind) {
          auto [uv, vv] = hsd.windInterp.GetWindValue(orig_i, orig_j);
          vx = uv;
          vy = vv;
        }

        vtkIdType vtk_idx = i + j * vis_gx;
        vectors->SetTuple3(vtk_idx, vx, vy, 0.0);
      }
    }
  };

  if (show_hedgehog) {
    bool sample_ocean = (HedgehogVariable == VisOpt::hedgehog_ocean);
    bool sample_wind = (HedgehogVariable == VisOpt::hedgehog_wind);

    // For applied shear, we handle it custom below
    if (HedgehogVariable == VisOpt::hedgehog_applied_shear) {
      // Apply a custom lambda that writes custom stuff. For now, a placeholder
      // overlaying wind and ocean
      auto placeholder_update = [&](vtkStructuredGrid *grid,
                                    vtkFloatArray *vectors, double z_pos) {
        int vis_gx = std::min(gx, 2000);
        int vis_gy = (gx > 0) ? (int)((long long)gy * vis_gx / gx) : 0;
        vis_gx = std::max(vis_gx, 2);
        vis_gy = std::max(vis_gy, 2);

        int dims[3];
        grid->GetDimensions(dims);
        if (dims[0] != vis_gx || dims[1] != vis_gy) {
          grid->SetDimensions(vis_gx, vis_gy, 1);
          vectors->SetNumberOfTuples(vis_gx * vis_gy);

          vtkNew<vtkPoints> gp;
          gp->SetNumberOfPoints(vis_gx * vis_gy);
          for (int j = 0; j < vis_gy; j++) {
            for (int i = 0; i < vis_gx; i++) {
              double r_i = (double)i / (vis_gx - 1);
              double r_j = (double)j / (vis_gy - 1);
              double phys_x = (ox + r_i * (gx - 1)) * h;
              double phys_y = (oy + r_j * (gy - 1)) * h;
              vtkIdType vtk_idx = i + j * vis_gx;
              gp->SetPoint(vtk_idx, phys_x, phys_y, z_pos);
            }
          }
          grid->SetPoints(gp);
        }

#pragma omp parallel for
        for (int i = 0; i < vis_gx; i++) {
          for (int j = 0; j < vis_gy; j++) {
            int orig_i = (int)((double)i / (vis_gx - 1) * (gx - 1));
            int orig_j = (int)((double)j / (vis_gy - 1) * (gy - 1));

            if (orig_i >= gx)
              orig_i = gx - 1;
            if (orig_j >= gy)
              orig_j = gy - 1;

            auto [uv, vv] = hsd.currentInterp.GetOceanValue(orig_i, orig_j);
            auto [uv_w, vv_w] = hsd.windInterp.GetWindValue(orig_i, orig_j);

            // Placeholder: Combine them conceptually, to be filled in with
            // actual physics later
            float vx = uv + uv_w;
            float vy = vv + vv_w;

            vtkIdType vtk_idx = i + j * vis_gx;
            vectors->SetTuple3(vtk_idx, vx, vy, 0.0);
          }
        }
      };
      placeholder_update(hedgehog_flow_grid, hedgehog_flow_vectors, 0.5);
    } else {
      // use a small z_pos (e.g. 0.5) to render on top of raster but below
      // points (1.0)
      update_grid_and_vectors(hedgehog_flow_grid, hedgehog_flow_vectors,
                              sample_ocean, sample_wind, 0.5);
    }
  }

  hedgehog_actor_vectors->VisibilityOff();
  if (show_hedgehog) {
    int dims[3];
    hedgehog_flow_grid->GetDimensions(dims);
    int f_gx = dims[0];
    int f_gy = dims[1];

    int stride = std::max(1, f_gx / vectorDensity);

    for (int j = 0; j < f_gy; j++) {
      for (int i = 0; i < f_gx; i++) {
        if (i % stride != 0 || j % stride != 0) {
          hedgehog_flow_vectors->SetTuple3(i + j * f_gx, 0.0, 0.0, 0.0);
        }
      }
    }
    hedgehog_flow_vectors->Modified();

    // Use hedgehog_ranges factor to scale arrowheads safely
    double scale_control = VisOpt::hedgehog_ranges[(int)HedgehogVariable];
    hedgehog_filter->SetScaleFactor(h * 5.0 * scale_control);

    hedgehog_filter->Update();
    hedgehog_actor_vectors->VisibilityOn();
  }
}

void VisualRepresentation::SynchronizeValues() {
  HostSideSOA &hssoa = hsd.hssoa;
  const SimParams &prms = hsd.prms;
  const int nPts = hssoa.size;

  if (nPts == 0) {
    actor_points->VisibilityOff();
    return;
  }

  points->SetNumberOfPoints(nPts);
  pts_colors->SetNumberOfValues(nPts * 3);

  const int ox = prms.ModeledRegionOffsetX;
  const int oy = prms.ModeledRegionOffsetY;
  const double h = prms.cellsize;

  for (int i = 0; i < nPts; i++) {
    SOAIterator s = hssoa.begin() + i;
    Eigen::Vector2d pos = s->getPos(h);
    points->SetPoint((vtkIdType)i, pos[0] + ox * h, pos[1] + oy * h, 1.0);
  }

  actor_points->VisibilityOn();
  points_mapper->ScalarVisibilityOn();
  points_mapper->SetColorModeToDirectScalars();

  const double range_from = VisOpt::range_from[VisualizingVariable];
  const double range_to = VisOpt::range_to[VisualizingVariable];
  // Scale used for magnitude-based alpha blending (e.g. Jp_inv):
  // the largest extent of [range_from, range_to] from zero, so a value at
  // the edge of the visible range gives full blend weight.
  const double mix_scale = std::max(std::abs(range_from), std::abs(range_to));
  const double transparency =
      VisOpt::transparency_coeffs[(int)VisualizingVariable];

  auto get_util_color = [](uint64_t util) -> std::array<uint8_t, 3> {
    return {static_cast<uint8_t>((util >> 24) & 0xFF),
            static_cast<uint8_t>((util >> 32) & 0xFF),
            static_cast<uint8_t>((util >> 40) & 0xFF)};
  };

  if (VisualizingVariable == VisOpt::pt_color) {
    for (int i = 0; i < nPts; i++) {
      SOAIterator s = hssoa.begin() + i;
      uint64_t util = s->getValueUInt64(SimParams::PtArrIdx::idx_utility_data);
      auto c = get_util_color(util);
      pts_colors->SetTuple3((vtkIdType)i, c[0], c[1], c[2]);
    }
  } else if (VisualizingVariable == VisOpt::pt_status) {
    for (int i = 0; i < nPts; i++) {
      SOAIterator s = hssoa.begin() + i;
      uint64_t util = s->getValueUInt64(SimParams::PtArrIdx::idx_utility_data);
      std::array<uint8_t, 3> c = ColorMap::rgb_white; // Default Intact (White)
      if (util & SimParams::fracture_crush)
        c = ColorMap::rgb_red;
      else if (util & SimParams::status_cracked)
        c = ColorMap::rgb_green;
      pts_colors->SetTuple3((vtkIdType)i, c[0], c[1], c[2]);
    }
  } else if (VisualizingVariable == VisOpt::pt_fracture_type) {
    for (int i = 0; i < nPts; i++) {
      SOAIterator s = hssoa.begin() + i;
      uint64_t util = s->getValueUInt64(SimParams::PtArrIdx::idx_utility_data);
      auto c = get_util_color(util);

      if (util &
          (SimParams::fracture_tension | SimParams::fracture_compression_shear |
           SimParams::fracture_crush)) {
        c = {0, 0, 0};
        if (util & SimParams::fracture_tension)
          c[2] = 255; // Blue
        if (util & SimParams::fracture_compression_shear)
          c[1] = 255; // Green
        if (util & SimParams::fracture_crush)
          c = {255, 0, 0}; // Red
      }
      pts_colors->SetTuple3((vtkIdType)i, c[0], c[1], c[2]);
    }
  } else if (VisualizingVariable == VisOpt::none) {
    for (int i = 0; i < nPts; i++) {
      pts_colors->SetTuple3((vtkIdType)i, 240, 122, 122);
    }
  } else if (VisualizingVariable == VisOpt::pt_Jp_inv) {
    for (int i = 0; i < nPts; i++) {
      SOAIterator s = hssoa.begin() + i;
      double val = s->getValue(SimParams::PtArrIdx::idx_Jp_inv) - 1.0;
      double alpha = (1.0 - transparency) +
                     transparency * std::min(1.0, std::abs(val) / mix_scale);

      uint64_t util = s->getValueUInt64(SimParams::PtArrIdx::idx_utility_data);
      auto orig = get_util_color(util);
      auto c = colormap.getColor(ColorMap::Palette::Pressure,
                                  (val - range_from) / (range_to - range_from));
      auto c2 = colormap.mergeColors(orig, c, alpha);
      pts_colors->SetTuple3((vtkIdType)i, c2[0], c2[1], c2[2]);
    }
  } else if (VisualizingVariable == VisOpt::pt_thickness) {
    const double thickness_min = VisOpt::range_from[VisOpt::pt_thickness];
    const double thickness_max = VisOpt::range_to[VisOpt::pt_thickness];
    const double thickness_span = thickness_max - thickness_min;
    for (int i = 0; i < nPts; i++) {
      SOAIterator s = hssoa.begin() + i;
      double val = s->getValue(SimParams::PtArrIdx::idx_thickness);
      auto c = colormap.getColor(ColorMap::Palette::ANSYS,
                                  (val - thickness_min) / thickness_span);
      pts_colors->SetTuple3((vtkIdType)i, c[0], c[1], c[2]);
    }
    lut_ANSYS->SetTableRange(thickness_min, thickness_max);
    scalarBar->SetLookupTable(lut_ANSYS);
    scalarBar->SetLabelFormat("%.2f");
    scalarBar->VisibilityOn();
  } else if (VisualizingVariable == VisOpt::pt_ice_strength) {
    const double xi_min = VisOpt::range_from[VisOpt::pt_ice_strength];
    const double xi_max = VisOpt::range_to[VisOpt::pt_ice_strength];
    const double xi_span = xi_max - xi_min;
    // idx_xi is already the raw, unscaled Timco & O'Brien flexural strength
    // (see partition_kernel_compute_ice_strength) -- no factor to divide out.
    for (int i = 0; i < nPts; i++) {
      SOAIterator s = hssoa.begin() + i;
      double val = s->getValue(SimParams::PtArrIdx::idx_xi);
      auto c = colormap.getColor(ColorMap::Palette::ANSYS,
                                  (val - xi_min) / xi_span);
      pts_colors->SetTuple3((vtkIdType)i, c[0], c[1], c[2]);
    }
    lut_ANSYS->SetTableRange(xi_min, xi_max);
    scalarBar->SetLookupTable(lut_ANSYS);
    scalarBar->SetLabelFormat("%.2f");
    scalarBar->VisibilityOn();
  } else if (VisualizingVariable == VisOpt::pt_gamma_p) {
    const double gp_min = VisOpt::range_from[VisOpt::pt_gamma_p];
    const double gp_max = VisOpt::range_to[VisOpt::pt_gamma_p];
    const double gp_span = gp_max - gp_min;
    for (int i = 0; i < nPts; i++) {
      SOAIterator s = hssoa.begin() + i;
      double val = s->getValue(SimParams::PtArrIdx::idx_gamma_p);
      auto c = colormap.getColor(ColorMap::Palette::ANSYS,
                                  (val - gp_min) / gp_span);
      pts_colors->SetTuple3((vtkIdType)i, c[0], c[1], c[2]);
    }
    lut_ANSYS->SetTableRange(gp_min, gp_max);
    scalarBar->SetLookupTable(lut_ANSYS);
    scalarBar->SetLabelFormat("%.2f");
    scalarBar->VisibilityOn();
  } else if (VisualizingVariable == VisOpt::pt_Jp_inv_max) {
    const double jm_min = VisOpt::range_from[VisOpt::pt_Jp_inv_max];
    const double jm_max = VisOpt::range_to[VisOpt::pt_Jp_inv_max];
    const double jm_span = jm_max - jm_min;
    for (int i = 0; i < nPts; i++) {
      SOAIterator s = hssoa.begin() + i;
      double val = s->getValue(SimParams::PtArrIdx::idx_Jp_inv_max);
      auto c = colormap.getColor(ColorMap::Palette::ANSYS,
                                  (val - jm_min) / jm_span);
      pts_colors->SetTuple3((vtkIdType)i, c[0], c[1], c[2]);
    }
    lut_ANSYS->SetTableRange(jm_min, jm_max);
    scalarBar->SetLookupTable(lut_ANSYS);
    scalarBar->SetLabelFormat("%.2f");
    scalarBar->VisibilityOn();
  } else if (VisualizingVariable == VisOpt::pt_partitions) {
    for (int i = 0; i < nPts; i++) {
      SOAIterator s = hssoa.begin() + i;
      uint64_t util = s->getValueUInt64(SimParams::PtArrIdx::idx_utility_data);
      auto c = colormap.getColor(ColorMap::Palette::NCD, (util & 0xFFFF) / 8.0);
      pts_colors->SetTuple3((vtkIdType)i, c[0], c[1], c[2]);
    }
  } else {
    // Default case: disable point rendering (for v_u, v_v, v_norm, and other
    // unhandled modes)
    actor_points->VisibilityOff();
  }

  // Disabled points are never removed from hssoa on every step (only
  // occasionally, via RemoveDisabledAndSort's threshold-triggered squeeze --
  // see Model::Step), and their position stops updating the instant they're
  // disabled (partition_kernel_g2p's early return), so they'd otherwise sit
  // frozen and visibly "accumulate" wherever they were last active (e.g. the
  // open boundary). Painting them black hides them against the background,
  // which is solid black there (see HostSideData::FillModelledAreaWithBlueColor).
  for (int i = 0; i < nPts; i++) {
    SOAIterator s = hssoa.begin() + i;
    uint64_t util = s->getValueUInt64(SimParams::PtArrIdx::idx_utility_data);
    if (util & SimParams::status_disabled)
      pts_colors->SetTuple3((vtkIdType)i, 0, 0, 0);
  }

  // Notify VTK that data has changed
  pts_colors->Modified();
  points_polydata->GetPointData()->SetActiveScalars("pts_colors");
  points_polydata->Modified();
  points->Modified();
  points_filter->Update();
  actor_points->GetProperty()->SetPointSize(ParticleViewSize);
}

void VisualRepresentation::ConfigureScalarBar() {
  const double range_from = VisOpt::range_from[VisualizingVariable];
  const double range_to = VisOpt::range_to[VisualizingVariable];

  // Default to visible
  scalarBar->VisibilityOn();
  scalarBarBgActor->VisibilityOn();
  actorTextTitle->SetInput(
      VisOpt::descriptions.at(VisualizingVariable).second.data());

  switch (VisualizingVariable) {

  case VisOpt::pt_Jp_inv:
  case VisOpt::grid_Jpinv:
  case VisOpt::grid_P:
    lut_Pressure->SetTableRange(range_from, range_to);
    scalarBar->SetLookupTable(lut_Pressure);
    scalarBar->SetLabelFormat("%.1e");
    break;

  case VisOpt::pt_thickness:
  case VisOpt::v_glo12_thickness:
    lut_ANSYS->SetTableRange(range_from, range_to);
    scalarBar->SetLookupTable(lut_ANSYS);
    scalarBar->SetLabelFormat("%.2f");
    scalarBar->VisibilityOn();
    break;

  case VisOpt::grid_thickness:
    lut_Ice->SetTableRange(range_from, range_to);
    scalarBar->SetLookupTable(lut_Ice);
    scalarBar->SetLabelFormat("%.2f");
    scalarBar->VisibilityOn();
    break;

  case VisOpt::pt_ice_strength:
  case VisOpt::grid_ice_strength:
    // Pa-scale (up to ~7e6) -- scientific notation, same as the Pressure group.
    lut_ANSYS->SetTableRange(range_from, range_to);
    scalarBar->SetLookupTable(lut_ANSYS);
    scalarBar->SetLabelFormat("%.2e");
    scalarBar->VisibilityOn();
    break;

  case VisOpt::pt_gamma_p:
  case VisOpt::grid_gamma_p:
    lut_ANSYS->SetTableRange(range_from, range_to);
    scalarBar->SetLookupTable(lut_ANSYS);
    scalarBar->SetLabelFormat("%.2f");
    scalarBar->VisibilityOn();
    break;

  case VisOpt::carra1_temperature:
    lut_ANSYS->SetTableRange(range_from, range_to);
    scalarBar->SetLookupTable(lut_ANSYS);
    scalarBar->SetLabelFormat("%.1f");
    scalarBar->VisibilityOn();
    break;

  case VisOpt::grid_Q:
  case VisOpt::grid_vnorm:
  case VisOpt::str_EqvGreenLagrange:
  case VisOpt::str_vonMises:
  case VisOpt::grid_pt_count: // Enable scalar bar
  case VisOpt::grid_mass:
  case VisOpt::glo12_ocean:
  case VisOpt::carra1_wind:
    lut_ANSYS->SetTableRange(range_from, range_to);
    scalarBar->SetLookupTable(lut_ANSYS);
    scalarBar->SetLabelFormat("%.1e");
    scalarBar->VisibilityOn();
    break;

  default:
    scalarBar->VisibilityOff();
    scalarBarBgActor->VisibilityOff();
    break;
  }

  // User-driven override (View > Show Color Bar) -- applied last so it wins
  // regardless of which case above ran; re-asserted on every call since this
  // function re-derives visibility from scratch on each topology sync.
  if (!userWantsScalarBarVisible) {
    scalarBar->VisibilityOff();
    scalarBarBgActor->VisibilityOff();
  }
}

void VisualRepresentation::ExportScalarBarSVG(const std::string &filePath) {
  vtkLookupTable *lut = vtkLookupTable::SafeDownCast(scalarBar->GetLookupTable());
  if (!lut) {
    LOGR("ExportScalarBarSVG: no lookup table assigned (scalar bar hidden for this variable)");
    return;
  }

  const double *range = lut->GetTableRange();
  const int nColors = lut->GetNumberOfTableValues();
  const int nLabels = scalarBar->GetNumberOfLabels();
  const char *labelFormat = scalarBar->GetLabelFormat();

  // Physical units per variable -- not carried by VisOpt::descriptions, so
  // listed here next to the only place that renders them.
  static const std::map<VisOpt::Type, std::string> units = {
      {VisOpt::grid_P, "Pa"},           {VisOpt::pt_ice_strength, "Pa"},
      {VisOpt::grid_ice_strength, "Pa"},{VisOpt::grid_Q, "Pa"},
      {VisOpt::pt_thickness, "m"},      {VisOpt::grid_thickness, "m"},
      {VisOpt::v_glo12_thickness, "m"}, {VisOpt::grid_vnorm, "m/s"},
      {VisOpt::glo12_ocean, "m/s"},     {VisOpt::carra1_wind, "m/s"},
      {VisOpt::grid_mass, "kg/m^2"},
  };
  auto uIt = units.find(VisualizingVariable);
  const std::string unit = (uIt != units.end()) ? uIt->second : "";
  const std::string title = VisOpt::descriptions.at(VisualizingVariable).second;

  const double barW = 40, barH = 300;
  const double marginL = 20, marginT = 50, marginR = 140, marginB = 30;
  const double canvasW = marginL + barW + marginR;
  const double canvasH = marginT + barH + marginB;

  std::ofstream f(filePath);
  if (!f) {
    LOGR("ExportScalarBarSVG: failed to open {}", filePath);
    return;
  }

  f << "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n";
  f << "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"" << canvasW << "\" height=\"" << canvasH
    << "\" viewBox=\"0 0 " << canvasW << " " << canvasH << "\">\n";
  f << "<rect x=\"0\" y=\"0\" width=\"" << canvasW << "\" height=\"" << canvasH << "\" fill=\"white\"/>\n";

  // Gradient built from the exact same 256-entry table populateLut() derived
  // from the palette, so it reproduces the on-screen bar rather than an
  // approximation of it. userSpaceOnUse (absolute coords) rather than the
  // objectBoundingBox default -- Inkscape has a known bug where moving or
  // grouping an objectBoundingBox-gradient object collapses it to a solid
  // color, since it recomputes the gradient against the shape's bbox in the
  // new context; absolute coordinates sidestep that entirely.
  f << "<defs><linearGradient id=\"grad\" gradientUnits=\"userSpaceOnUse\" x1=\""
    << (marginL + barW / 2) << "\" y1=\"" << (marginT + barH) << "\" x2=\""
    << (marginL + barW / 2) << "\" y2=\"" << marginT << "\">\n";
  for (int i = 0; i < nColors; ++i) {
    double *rgba = lut->GetTableValue(i);
    const double offset = (nColors > 1) ? (double)i / (nColors - 1) : 0.0;
    const int r = (int)std::round(rgba[0] * 255);
    const int g = (int)std::round(rgba[1] * 255);
    const int b = (int)std::round(rgba[2] * 255);
    f << "  <stop offset=\"" << offset << "\" stop-color=\"rgb(" << r << "," << g << "," << b << ")\"/>\n";
  }
  f << "</linearGradient></defs>\n";

  f << "<rect x=\"" << marginL << "\" y=\"" << marginT << "\" width=\"" << barW << "\" height=\"" << barH
    << "\" fill=\"url(#grad)\" stroke=\"black\" stroke-width=\"1\"/>\n";

  f << "<text x=\"" << marginL << "\" y=\"" << (marginT - 15)
    << "\" font-family=\"sans-serif\" font-size=\"16\" font-weight=\"bold\" fill=\"black\">" << title
    << (unit.empty() ? "" : (" (" + unit + ")")) << "</text>\n";

  // Ticks at NumberOfLabels equally spaced values, low value at the bottom,
  // matching vtkScalarBarActor's default vertical orientation.
  char buf[64];
  for (int i = 0; i < nLabels; ++i) {
    const double t = (nLabels > 1) ? (double)i / (nLabels - 1) : 0.0;
    const double value = range[0] + t * (range[1] - range[0]);
    const double y = marginT + barH * (1.0 - t);

    snprintf(buf, sizeof(buf), labelFormat, value);

    f << "<line x1=\"" << (marginL + barW) << "\" y1=\"" << y << "\" x2=\"" << (marginL + barW + 6)
      << "\" y2=\"" << y << "\" stroke=\"black\" stroke-width=\"1\"/>\n";
    f << "<text x=\"" << (marginL + barW + 10) << "\" y=\"" << (y + 8)
      << "\" font-family=\"sans-serif\" font-size=\"22\" fill=\"black\">" << buf << "</text>\n";
  }

  f << "</svg>\n";
  f.close();

  LOGR("ExportScalarBarSVG: wrote {}", filePath);
}

void VisualRepresentation::UpdateTimeText() {
  LOGR("VisualRepresentation::UpdateTimeText()");
  // Convert total seconds to days:hours, relative to simulationTime's zero
  // reference (SimulationStartDate) -- negative values (before the
  // simulation start, e.g. while scrubbing pre-simulation CARRA1/GLO12 data)
  // are valid and use floor division so hours always stays in [0,23]: e.g.
  // -3600s is "Day -1 Hour 23" (one hour before the start), not "Day 0 Hour -1".
  long long total_seconds = static_cast<long long>(simulationTime);

  const long long seconds_per_day = 24 * 3600;
  const long long seconds_per_hour = 3600;

  long long days = total_seconds / seconds_per_day;
  long long remaining_seconds = total_seconds % seconds_per_day;
  if (remaining_seconds < 0) {
    remaining_seconds += seconds_per_day;
    days -= 1;
  }
  int hours = static_cast<int>(remaining_seconds / seconds_per_hour);

  char buffer[100];
  snprintf(buffer, sizeof(buffer), "Day %02lld  Hour %02d", days, hours);
  actorText->SetInput(buffer);

  // Absolute calendar date, e.g. "May 20, 2024" -- hsd.prms.SimulationStartTime
  // is the Unix epoch at simulationTime==0 (same zero reference as above).
  std::time_t abs_epoch = (std::time_t)(hsd.prms.SimulationStartTime + total_seconds);
  std::tm tm_utc;
  gmtime_r(&abs_epoch, &tm_utc);
  char dateBuffer[64];
  std::strftime(dateBuffer, sizeof(dateBuffer), "%B %d, %Y, %Hh UTC", &tm_utc);
  actorDateText->SetInput(dateBuffer);
}

void VisualRepresentation::SetupRegionBoundary(int gx, int gy, int ox, int oy,
                                               double h) {
  // Rectangle corners at cell boundaries, with cell centers at grid nodes
  // (ox, oy) is the grid offset in image coordinates
  // Grid extends from (ox, oy) to (ox + gx - 1, oy + gy - 1) in cell centers
  // Rectangle edges are at half-cell distance from corners

  double x_min = (ox - 0.5) * h;
  double x_max = (ox + gx - 0.5) * h;
  double y_min = (oy - 0.5) * h;
  double y_max = (oy + gy - 0.5) * h;

  vtkNew<vtkPoints> boundary_points;
  boundary_points->InsertNextPoint(x_min, y_min, 0.0); // 0: bottom-left
  boundary_points->InsertNextPoint(x_max, y_min, 0.0); // 1: bottom-right
  boundary_points->InsertNextPoint(x_max, y_max, 0.0); // 2: top-right
  boundary_points->InsertNextPoint(x_min, y_max, 0.0); // 3: top-left

  vtkNew<vtkCellArray> boundary_lines;
  vtkIdType line_ids[2];
  // Bottom edge
  line_ids[0] = 0;
  line_ids[1] = 1;
  boundary_lines->InsertNextCell(2, line_ids);
  // Right edge
  line_ids[0] = 1;
  line_ids[1] = 2;
  boundary_lines->InsertNextCell(2, line_ids);
  // Top edge
  line_ids[0] = 2;
  line_ids[1] = 3;
  boundary_lines->InsertNextCell(2, line_ids);
  // Left edge
  line_ids[0] = 3;
  line_ids[1] = 0;
  boundary_lines->InsertNextCell(2, line_ids);

  vtkNew<vtkPolyData> boundary_polydata;
  boundary_polydata->SetPoints(boundary_points);
  boundary_polydata->SetLines(boundary_lines);

  vtkNew<vtkPolyDataMapper> boundary_mapper;
  boundary_mapper->SetInputData(boundary_polydata);
  actor_region_boundary->SetMapper(boundary_mapper);
  actor_region_boundary->GetProperty()->SetColor(0.0, 0.0, 0.0); // Black lines
  actor_region_boundary->GetProperty()->SetLineWidth(2.0);
  actor_region_boundary->VisibilityOn();
}

VisualRepresentation::~VisualRepresentation() {
  VisOpt::SaveVisualizationState();
}


void VisualRepresentation::SetArrowheadSize(double size) {
  hedgehog_filter->SetArrowheadSize(size);
  hedgehog_filter->Modified();
}

void VisualRepresentation::SetParticleViewSize(int size) {
  ParticleViewSize = size;
  actor_points->GetProperty()->SetPointSize(ParticleViewSize);
}
