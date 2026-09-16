// visual_representation.h
#ifndef VISUAL_REPRESENTATION_H
#define VISUAL_REPRESENTATION_H

#include <QObject>
#include <vector>

#include <vtkNew.h>
#include <vtkActor.h>
#include <vtkPolyDataMapper.h>
#include <vtkLookupTable.h>
#include <vtkPolyData.h>
#include <vtkPoints.h>
#include <vtkVertexGlyphFilter.h>
#include <vtkScalarBarActor.h>
#include <vtkTextActor.h>
#include <vtkPlaneSource.h>
#include <vtkTexture.h>
#include <vtkActor2D.h>
#include <vtkUnsignedCharArray.h>
#include <vtkImageData.h>
#include <vtkStructuredGrid.h>
#include "vtkHedgeHogCustom.h"
#include <vtkStreamTracer.h>
#include <vtkTubeFilter.h>
#include <vtkFloatArray.h>

#include "colormap.h"
#include "visualization_options.h"

class HostSideData;

class VisualRepresentation : public QObject
{
    Q_OBJECT

public:
    explicit VisualRepresentation(HostSideData& data);
    ~VisualRepresentation();

    vtkNew<vtkActor> actor_points;
    vtkNew<vtkActor> raster_actor;
    vtkNew<vtkActor> actor_region_boundary;  // boundary lines for regions visualization

    // New actors for flow visualization
    vtkNew<vtkActor> hedgehog_actor_vectors; // For Hedgehog Overlay

    vtkNew<vtkTextActor> actorText;
    vtkNew<vtkTextActor> actorTextTitle;
    vtkNew<vtkScalarBarActor> scalarBar;
    vtkNew<vtkActor2D> textBgActor;
    vtkNew<vtkActor2D> scalarBarBgActor;

    // Absolute calendar date (e.g. "May 20, 2024"), anchored to the upper-left
    // corner via normalized viewport coordinates so it stays visible/in-frame
    // regardless of window size or cropping -- unlike actorText/textBgActor,
    // which use fixed pixel positions tuned for a specific window size.
    vtkNew<vtkTextActor> actorDateText;
    vtkNew<vtkActor2D> dateBgActor;

    HostSideData& hsd;

    double simulationTime = 0;

    VisOpt::Type VisualizingVariable = VisOpt::none;
    // User override from View > Show Color Bar -- ConfigureScalarBar()
    // enforces this regardless of what VisualizingVariable would otherwise show.
    bool userWantsScalarBarVisible = true;
    VisOpt::HedgehogType HedgehogVariable = VisOpt::hedgehog_none;

    void SynchronizeTopology();
    void ChangeVisualizationOption(int option);
    void ConfigureScalarBar();
    void UpdateTimeText();
    void ExportScalarBarSVG(const std::string& filePath);  // vector reproduction of scalarBar, saved separately

    double ParticleViewSize = 1.0;
    void SetParticleViewSize(int size);

    void SetArrowheadSize(double size);

private:
    constexpr static int vectorDensity = 25;

    ColorMap colormap;
    void SynchronizeValues();
    void UpdateFlowData();

    void SetupRegionBoundary(int gx, int gy, int ox, int oy, double h);  // draw rectangle for modeled area

    vtkNew<vtkLookupTable> lut_Pressure, lut_P2, lut_ANSYS, lut_Ice;
    void populateLut(ColorMap::Palette palette, vtkNew<vtkLookupTable>& table);

    // points
    vtkNew<vtkPoints> points;
    vtkNew<vtkPolyData> points_polydata;
    vtkNew<vtkPolyDataMapper> points_mapper;
    vtkNew<vtkVertexGlyphFilter> points_filter;
    vtkNew<vtkUnsignedCharArray> pts_colors;

    // background image
    std::vector<uint8_t> renderedImage;
    vtkNew<vtkImageData> raster_imageData;
    vtkNew<vtkUnsignedCharArray> raster_scalars;
    vtkNew<vtkPlaneSource> raster_plane;
    vtkNew<vtkTexture> raster_texture;
    vtkNew<vtkPolyDataMapper> raster_mapper;
    
    // Flow Visualization Infrastructure
    vtkNew<vtkStructuredGrid> flow_grid;
    vtkNew<vtkFloatArray> flow_vectors;

    // HedgeHog
    vtkNew<vtkStructuredGrid> hedgehog_flow_grid;
    vtkNew<vtkFloatArray> hedgehog_flow_vectors;
    vtkNew<vtkHedgeHogCustom> hedgehog_filter;
    vtkNew<vtkPolyDataMapper> hedgehog_mapper;
};

#endif
