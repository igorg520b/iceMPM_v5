// wind_rose_widget.h
#ifndef WIND_ROSE_WIDGET_H
#define WIND_ROSE_WIDGET_H

#include <array>
#include <functional>
#include <string>
#include <utility>
#include <vector>

#include <vtkActor2D.h>
#include <vtkCoordinate.h>
#include <vtkNew.h>
#include <vtkPolyData.h>
#include <vtkPolyDataMapper2D.h>
#include <vtkTextActor.h>

class vtkRenderer;
class HostSideData;

// Screen-space "wind rose" HUD overlay for the visualizer: a conventional
// wind rose (radial axis = frequency, i.e. the fraction of samples blowing
// from that direction; stacked color bands within each of the 16 compass
// sectors = the speed distribution within that direction, "from"
// convention) sampled over the modeled area -- land and open-boundary grid
// cells excluded -- at the currently displayed frame's time.
//
// Data-source-agnostic: despite the name (kept for continuity -- this
// started as a wind-only widget), one instance can drive either a wind rose
// or an ocean-current rose, distinguished only by the constructor's title/
// speedBinEdges/dataSource/anchorNY -- see PPMainWindow's two instances.
// Sharing one class was more elegant than forking two near-identical ~350
// line VTK-geometry files that would need every future tweak applied twice.
//
// Deliberately self-contained (not part of VisualRepresentation): it owns
// its own vtkPolyData/mapper/actor and is added directly to the
// visualizer's vtkRenderer, so it can be dropped or reworked later without
// touching VisualRepresentation at all. Data source is HostSideData's
// existing WindInterpolator/CurrentInterpolator + landmask_buffer -- no new
// simulation-side state (WindInterpolator/CurrentInterpolator each gained
// one small public method, Get{Wind,Ocean}ValueGeographic, needed to get a
// true compass bearing -- see WindInterpolator's header comment for the
// rationale, identical for both).
//
// Fixed pixel size on screen regardless of window size or 3D camera zoom
// (standard VTK HUD technique: PositionCoordinate anchored in
// NormalizedViewport space, actual geometry built as plain pixel offsets
// from that anchor).
class WindRoseWidget
{
public:
    static constexpr int N_SPEED_BINS = 5;

    // Returns the true geographic (eastward, northward) vector at grid cell
    // (i,j) -- e.g. a lambda wrapping WindInterpolator::GetWindValueGeographic
    // or CurrentInterpolator::GetOceanValueGeographic.
    using DataSourceFn = std::function<std::pair<double,double>(HostSideData&, int, int)>;

    // title: short label drawn in the bottom-right corner, e.g. "Wind" or
    //   "Current".
    // speedBinEdges: N_SPEED_BINS bin start values (m/s); the last bin is
    //   open-ended (>= speedBinEdges[N_SPEED_BINS-1]). Wind and ocean
    //   current need very different ranges (currents here top out around
    //   0.7 m/s vs. ~25 m/s for wind), hence this being a constructor
    //   parameter rather than a shared constant.
    // dataSource: see DataSourceFn above.
    // anchorNY: vertical NormalizedViewport anchor (0=bottom, 1=top) --
    //   lets multiple instances stack without overlapping; ANCHOR_NX (the
    //   horizontal anchor) is shared/fixed since all instances line up in
    //   the same corner.
    WindRoseWidget(std::string title, std::array<double, N_SPEED_BINS> speedBinEdges,
                   DataSourceFn dataSource, double anchorNY);

    // Adds this widget's actors to the renderer. Call once, at setup.
    void AddToRenderer(vtkRenderer* ren);

    void SetVisible(bool visible);
    bool GetVisible() const { return visible; }

    // Rebuilds the (subsampled) list of modeled-area grid cells to sample.
    // Call once whenever a new project/grid is loaded (landmask_buffer /
    // grid dimensions change) -- NOT on every frame; sampling the same
    // ~163M-cell grid every frame would be far too slow. A fixed-size
    // subsample is enough since the underlying CARRA1/GLO12 fields are far
    // coarser than the ~50 m MPM grid.
    void RebuildSampleCells(HostSideData& hsd);

    // Re-bins samples (via dataSource, at whatever time the interpolator is
    // currently set to) and rebuilds the rose geometry. Call whenever the
    // displayed frame changes. No-op if not visible, or if
    // RebuildSampleCells hasn't found any modeled-area cells yet.
    void Update(HostSideData& hsd);

private:
    static constexpr int N_DIR_BINS = 16;
    static constexpr int ARC_SUBDIV = 3;      // extra points along each wedge's arcs, for a rounder look
    static constexpr double SECTOR_GAP_FRACTION = 0.18;  // gap between adjacent wedges, as a fraction of sector width
    static constexpr int N_GRIDLINES = 3;     // concentric frequency-percent reference circles
    // Radial axis ceiling, as a fraction of all samples (i.e. the outermost
    // gridline is FREQ_MAX_FRACTION*100%). Fixed, not renormalized to
    // whatever the busiest direction happens to be this frame, so the scale
    // means the same thing on every frame -- unlike the very first version
    // of this widget. Outer ring = 33% (not 100%): in practice a dominant
    // sector often lands well under 50% (see the logged histograms), so a
    // higher ceiling left most of the radius unused; BuildRoseGeometry
    // doesn't clamp petal length to OUTER_RADIUS_PX, so a sector that does
    // exceed 33% just draws past the outer ring instead of being capped.
    static constexpr double FREQ_MAX_FRACTION = 1.0 / 3.0;
    static constexpr double OUTER_RADIUS_PX = 120.0;  // 80 * 1.5
    static constexpr double BG_MARGIN_PX = 46.0;      // background rect margin beyond the outer radius/labels
    static constexpr double LABEL_FONT_SIZE = 14.0;
    static constexpr double LEGEND_FONT_SIZE = 11.0;
    static constexpr double LEGEND_SWATCH_PX = 12.0;
    static constexpr double LEGEND_ROW_HEIGHT_PX = 16.0;
    static constexpr double LEGEND_WIDTH_PX = 92.0;   // extra background width reserved for the legend, left of the rose
    static constexpr double TITLE_INSET_FRACTION = 0.20;  // title pulled 20% closer to center than the bg corner
    static constexpr double ANCHOR_NX = 0.9;   // NormalizedViewport anchor -- bottom-right corner, shared by all instances
    // Target number of subsampled cells (see RebuildSampleCells) -- large
    // enough for a smooth-looking histogram, small enough to stay fast.
    static constexpr int TARGET_SAMPLE_COUNT = 6000;

    const std::string title;
    const std::array<double, N_SPEED_BINS> speedBinEdges;
    const DataSourceFn dataSource;
    const double anchorNY;

    std::vector<std::pair<int,int>> sampleCells;

    vtkNew<vtkPolyData> polyData;
    vtkNew<vtkPolyDataMapper2D> mapper;
    vtkNew<vtkActor2D> actor;

    // White background rectangle, drawn behind everything else.
    vtkNew<vtkPolyData> bgPolyData;
    vtkNew<vtkPolyDataMapper2D> bgMapper;
    vtkNew<vtkActor2D> bgActor;

    // Concentric reference (gridline) circles, built once (fixed geometry).
    vtkNew<vtkPolyData> ringPolyData;
    vtkNew<vtkPolyDataMapper2D> ringMapper;
    vtkNew<vtkActor2D> ringActor;

    // Speed-bin legend swatches (fixed colors -- built once).
    vtkNew<vtkPolyData> legendPolyData;
    vtkNew<vtkPolyDataMapper2D> legendMapper;
    vtkNew<vtkActor2D> legendActor;

    // N/E/S/W compass labels, the title, gridline (%) labels, and the
    // per-bin legend text, all positioned via vtkCoordinate reference-
    // coordinate chaining off the same NormalizedViewport anchor as the
    // rose geometry (so everything moves together with it).
    std::array<vtkNew<vtkTextActor>, 4> compassLabels;
    vtkNew<vtkTextActor> titleLabel;
    std::array<vtkNew<vtkTextActor>, N_GRIDLINES> gridlineLabels;
    std::array<vtkNew<vtkTextActor>, N_SPEED_BINS> legendLabels;

    bool visible = false;

    void BuildStaticGeometry();
    void BuildRoseGeometry(const std::array<std::array<int, N_SPEED_BINS>, N_DIR_BINS>& counts,
                            int grandTotal);
};

#endif
