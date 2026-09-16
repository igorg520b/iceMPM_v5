// wind_rose_widget.cpp
#include "wind_rose_widget.h"

#include <algorithm>
#include <cmath>
#include <cstdio>

#include <vtkCellArray.h>
#include <vtkCellData.h>
#include <vtkPoints.h>
#include <vtkPolyLine.h>
#include <vtkProperty2D.h>
#include <vtkRenderer.h>
#include <vtkTextProperty.h>
#include <vtkUnsignedCharArray.h>

#include "gui/colormap.h"
#include "host_side_data.h"
#include "parameters_sim.h"

WindRoseWidget::WindRoseWidget(std::string title_, std::array<double, N_SPEED_BINS> speedBinEdges_,
                               DataSourceFn dataSource_, double anchorNY_)
    : title(std::move(title_)), speedBinEdges(speedBinEdges_),
      dataSource(std::move(dataSource_)), anchorNY(anchorNY_)
{
    // Background rectangle: covers the ring/rose plus compass labels/title
    // on the right, and the speed-bin legend column on the left. Drawn on
    // layer 2 (behind the rest, layer 1) -- same layer convention already
    // used for the text-background rectangles in visual_representation.cpp.
    {
        const double right = OUTER_RADIUS_PX + BG_MARGIN_PX;
        const double left = -(OUTER_RADIUS_PX + BG_MARGIN_PX + LEGEND_WIDTH_PX);
        const double top = OUTER_RADIUS_PX + BG_MARGIN_PX;
        const double bottom = -(OUTER_RADIUS_PX + BG_MARGIN_PX);
        vtkNew<vtkPoints> bgPoints;
        bgPoints->InsertNextPoint(left, bottom, 0.0);
        bgPoints->InsertNextPoint(right, bottom, 0.0);
        bgPoints->InsertNextPoint(right, top, 0.0);
        bgPoints->InsertNextPoint(left, top, 0.0);
        vtkNew<vtkCellArray> bgPoly;
        vtkIdType bgIds[4] = {0, 1, 2, 3};
        bgPoly->InsertNextCell(4, bgIds);
        bgPolyData->SetPoints(bgPoints);
        bgPolyData->SetPolys(bgPoly);
    }
    bgMapper->SetInputData(bgPolyData);
    bgActor->SetMapper(bgMapper);
    bgActor->GetPositionCoordinate()->SetCoordinateSystemToNormalizedViewport();
    bgActor->SetPosition(ANCHOR_NX, anchorNY);
    bgActor->GetProperty()->SetColor(1.0, 1.0, 1.0);
    bgActor->GetProperty()->SetOpacity(1.0);
    bgActor->SetLayerNumber(2);
    bgActor->VisibilityOff();

    mapper->SetInputData(polyData);
    actor->SetMapper(mapper);
    actor->GetPositionCoordinate()->SetCoordinateSystemToNormalizedViewport();
    actor->SetPosition(ANCHOR_NX, anchorNY);
    actor->SetLayerNumber(1);
    actor->VisibilityOff();

    ringMapper->SetInputData(ringPolyData);
    ringActor->SetMapper(ringMapper);
    ringActor->GetPositionCoordinate()->SetCoordinateSystemToNormalizedViewport();
    ringActor->SetPosition(ANCHOR_NX, anchorNY);
    ringActor->GetProperty()->SetColor(0.55, 0.55, 0.55);
    ringActor->GetProperty()->SetLineWidth(1.0);
    ringActor->SetLayerNumber(1);
    ringActor->VisibilityOff();

    legendMapper->SetInputData(legendPolyData);
    legendActor->SetMapper(legendMapper);
    legendActor->GetPositionCoordinate()->SetCoordinateSystemToNormalizedViewport();
    legendActor->SetPosition(ANCHOR_NX, anchorNY);
    legendActor->SetLayerNumber(1);
    legendActor->VisibilityOff();

    // N/E/S/W labels, chained off the same NormalizedViewport anchor point
    // as the rose/ring/background geometry via ReferenceCoordinate -- their
    // own PositionCoordinate is then a plain pixel offset from that anchor,
    // so everything moves together and stays a fixed size on screen.
    const char* labelText[4] = {"N", "E", "S", "W"};
    const double labelOffset[4][2] = {
        {0.0, OUTER_RADIUS_PX + 12.0},
        {OUTER_RADIUS_PX + 8.0, -7.0},
        {0.0, -OUTER_RADIUS_PX - 22.0},
        {-OUTER_RADIUS_PX - 18.0, -7.0},
    };
    for (int k = 0; k < 4; ++k) {
        vtkTextActor* label = compassLabels[k];
        label->SetInput(labelText[k]);
        label->GetTextProperty()->SetFontSize((int)LABEL_FONT_SIZE);
        label->GetTextProperty()->SetColor(0.0, 0.0, 0.0);
        label->GetTextProperty()->SetJustificationToCentered();
        label->GetTextProperty()->SetVerticalJustificationToCentered();
        label->GetTextProperty()->BoldOn();
        label->GetPositionCoordinate()->SetCoordinateSystemToDisplay();
        label->GetPositionCoordinate()->SetReferenceCoordinate(actor->GetPositionCoordinate());
        label->GetPositionCoordinate()->SetValue(labelOffset[k][0], labelOffset[k][1]);
        label->SetLayerNumber(1);
        label->VisibilityOff();
    }

    // Title, bottom-right corner of the background rectangle (empty space
    // there since the rose itself is round), pulled TITLE_INSET_FRACTION
    // closer to the center than the bare corner.
    titleLabel->SetInput(title.c_str());
    titleLabel->GetTextProperty()->SetFontSize((int)LABEL_FONT_SIZE);
    titleLabel->GetTextProperty()->SetColor(0.0, 0.0, 0.0);
    titleLabel->GetTextProperty()->SetJustificationToRight();
    titleLabel->GetTextProperty()->SetVerticalJustificationToBottom();
    titleLabel->GetTextProperty()->BoldOn();
    titleLabel->GetPositionCoordinate()->SetCoordinateSystemToDisplay();
    titleLabel->GetPositionCoordinate()->SetReferenceCoordinate(actor->GetPositionCoordinate());
    titleLabel->GetPositionCoordinate()->SetValue(
        (OUTER_RADIUS_PX + BG_MARGIN_PX) * (1.0 - TITLE_INSET_FRACTION),
        -(OUTER_RADIUS_PX + BG_MARGIN_PX) * (1.0 - TITLE_INSET_FRACTION));
    titleLabel->SetLayerNumber(1);
    titleLabel->VisibilityOff();

    // Gridline (%) labels, along the NE (45 deg) spoke -- conventional wind
    // rose placement for the frequency-axis scale.
    for (int g = 1; g <= N_GRIDLINES; ++g) {
        const double r = OUTER_RADIUS_PX * g / N_GRIDLINES;
        const double rad = 45.0 * M_PI / 180.0;
        char buf[16];
        std::snprintf(buf, sizeof(buf), "%d%%", (int)std::lround(100.0 * FREQ_MAX_FRACTION * g / N_GRIDLINES));

        vtkTextActor* label = gridlineLabels[g - 1];
        label->SetInput(buf);
        label->GetTextProperty()->SetFontSize((int)LEGEND_FONT_SIZE);
        label->GetTextProperty()->SetColor(0.3, 0.3, 0.3);
        label->GetTextProperty()->SetJustificationToCentered();
        label->GetTextProperty()->SetVerticalJustificationToCentered();
        label->GetPositionCoordinate()->SetCoordinateSystemToDisplay();
        label->GetPositionCoordinate()->SetReferenceCoordinate(actor->GetPositionCoordinate());
        label->GetPositionCoordinate()->SetValue(r * std::sin(rad), r * std::cos(rad));
        label->SetLayerNumber(1);
        label->VisibilityOff();
    }

    // Speed-bin legend text, one row per bin, to the right of each swatch
    // (swatches themselves are built in BuildStaticGeometry). Values come
    // from this instance's own speedBinEdges (wind and ocean current use
    // very different ranges).
    {
        const double swatchX = -(OUTER_RADIUS_PX + BG_MARGIN_PX + LEGEND_WIDTH_PX) + 10.0 + LEGEND_SWATCH_PX / 2.0;
        const double textX = swatchX + LEGEND_SWATCH_PX / 2.0 + 5.0;
        const double topY = (N_SPEED_BINS - 1) / 2.0 * LEGEND_ROW_HEIGHT_PX;
        for (int s = 0; s < N_SPEED_BINS; ++s) {
            char buf[24];
            if (s < N_SPEED_BINS - 1)
                std::snprintf(buf, sizeof(buf), "%.2g-%.2g", speedBinEdges[s], speedBinEdges[s + 1]);
            else
                std::snprintf(buf, sizeof(buf), "%.2g+", speedBinEdges[s]);

            vtkTextActor* label = legendLabels[s];
            label->SetInput(buf);
            label->GetTextProperty()->SetFontSize((int)LEGEND_FONT_SIZE);
            label->GetTextProperty()->SetColor(0.0, 0.0, 0.0);
            label->GetTextProperty()->SetJustificationToLeft();
            label->GetTextProperty()->SetVerticalJustificationToCentered();
            label->GetPositionCoordinate()->SetCoordinateSystemToDisplay();
            label->GetPositionCoordinate()->SetReferenceCoordinate(actor->GetPositionCoordinate());
            label->GetPositionCoordinate()->SetValue(textX, topY - s * LEGEND_ROW_HEIGHT_PX);
            label->SetLayerNumber(1);
            label->VisibilityOff();
        }
    }

    BuildStaticGeometry();
}

void WindRoseWidget::AddToRenderer(vtkRenderer* ren)
{
    ren->AddViewProp(bgActor);
    ren->AddViewProp(ringActor);
    ren->AddViewProp(legendActor);
    ren->AddViewProp(actor);
    ren->AddViewProp(titleLabel);
    for (auto& label : compassLabels)
        ren->AddViewProp(label);
    for (auto& label : gridlineLabels)
        ren->AddViewProp(label);
    for (auto& label : legendLabels)
        ren->AddViewProp(label);
}

void WindRoseWidget::SetVisible(bool v)
{
    visible = v;
    bgActor->SetVisibility(v);
    actor->SetVisibility(v);
    ringActor->SetVisibility(v);
    legendActor->SetVisibility(v);
    titleLabel->SetVisibility(v);
    for (auto& label : compassLabels)
        label->SetVisibility(v);
    for (auto& label : gridlineLabels)
        label->SetVisibility(v);
    for (auto& label : legendLabels)
        label->SetVisibility(v);
}

void WindRoseWidget::BuildStaticGeometry()
{
    // Concentric frequency-percent gridline circles.
    {
        vtkNew<vtkPoints> points;
        vtkNew<vtkCellArray> lines;
        constexpr int N_RING_PTS = 64;

        for (int g = 1; g <= N_GRIDLINES; ++g) {
            const double r = OUTER_RADIUS_PX * g / N_GRIDLINES;
            vtkIdType ids[N_RING_PTS];
            for (int k = 0; k < N_RING_PTS; ++k) {
                const double t = 2.0 * M_PI * k / N_RING_PTS;
                ids[k] = points->InsertNextPoint(r * std::sin(t), r * std::cos(t), 0.0);
            }
            vtkNew<vtkPolyLine> polyLine;
            polyLine->GetPointIds()->SetNumberOfIds(N_RING_PTS + 1);
            for (int k = 0; k < N_RING_PTS; ++k)
                polyLine->GetPointIds()->SetId(k, ids[k]);
            polyLine->GetPointIds()->SetId(N_RING_PTS, ids[0]);
            lines->InsertNextCell(polyLine);
        }

        ringPolyData->SetPoints(points);
        ringPolyData->SetLines(lines);
    }

    // Speed-bin legend swatches -- fixed colors, same ANSYS-palette mapping
    // used for the rose's own stacked segments.
    {
        vtkNew<vtkPoints> points;
        vtkNew<vtkCellArray> polys;
        vtkNew<vtkUnsignedCharArray> colors;
        colors->SetNumberOfComponents(3);
        colors->SetName("Colors");

        const double swatchX = -(OUTER_RADIUS_PX + BG_MARGIN_PX + LEGEND_WIDTH_PX) + 10.0 + LEGEND_SWATCH_PX / 2.0;
        const double topY = (N_SPEED_BINS - 1) / 2.0 * LEGEND_ROW_HEIGHT_PX;
        const double half = LEGEND_SWATCH_PX / 2.0;

        for (int s = 0; s < N_SPEED_BINS; ++s) {
            const double cy = topY - s * LEGEND_ROW_HEIGHT_PX;
            vtkIdType ids[4];
            ids[0] = points->InsertNextPoint(swatchX - half, cy - half, 0.0);
            ids[1] = points->InsertNextPoint(swatchX + half, cy - half, 0.0);
            ids[2] = points->InsertNextPoint(swatchX + half, cy + half, 0.0);
            ids[3] = points->InsertNextPoint(swatchX - half, cy + half, 0.0);
            polys->InsertNextCell(4, ids);

            const auto rgb = ColorMap::getColor(ColorMap::Palette::ANSYS, (float)s / (N_SPEED_BINS - 1));
            colors->InsertNextTuple3(rgb[0], rgb[1], rgb[2]);
        }

        legendPolyData->SetPoints(points);
        legendPolyData->SetPolys(polys);
        legendPolyData->GetCellData()->SetScalars(colors);
    }
}

void WindRoseWidget::RebuildSampleCells(HostSideData& hsd)
{
    sampleCells.clear();

    const int gx = hsd.prms.GridXTotal;
    const int gy = hsd.prms.GridYTotal;
    if (gx <= 0 || gy <= 0 || hsd.landmask_buffer.size() < (size_t)gx * (size_t)gy)
        return;

    // Subsampling stride: the CARRA1/GLO12 fields are far coarser than the
    // ~50 m MPM grid, so a fixed-size subsample of modeled-area cells is
    // statistically equivalent to querying every one, at a fraction of the
    // per-frame cost.
    const double totalCells = (double)gx * (double)gy;
    const int stride = std::max(1, (int)std::round(std::sqrt(totalCells / TARGET_SAMPLE_COUNT)));

    for (int i = 0; i < gx; i += stride) {
        for (int j = 0; j < gy; j += stride) {
            const size_t idx = (size_t)j + (size_t)i * gy;
            if (hsd.landmask_buffer[idx] == SimParams::ModelledAreaIndicator)
                sampleCells.push_back({i, j});
        }
    }
}

void WindRoseWidget::Update(HostSideData& hsd)
{
    if (!visible || sampleCells.empty())
        return;

    std::array<std::array<int, N_SPEED_BINS>, N_DIR_BINS> counts{};
    for (auto& row : counts)
        row.fill(0);

    constexpr double sectorWidth = 360.0 / N_DIR_BINS;
    int grandTotal = 0;
    // Diagnostics -- see the LOGR summary below. noDataCount and calmCount
    // are deliberately distinguished (NaN vs. a real {0,0}): a large
    // noDataCount would mean a real gap in data coverage over part of the
    // modeled area (e.g. CARRA1/GLO12 not reaching that far), not a bug in
    // the land/open-boundary exclusion, which happens earlier in
    // RebuildSampleCells and is not revisited here.
    int noDataCount = 0, calmCount = 0;

    for (const auto& [i, j] : sampleCells) {
        // True geographic (east, north) components -- NOT the raw grid-
        // local frame, which only aligns with true compass directions near
        // the projection's tangent point (see
        // WindInterpolator::GetWindValueGeographic's header comment).
        const auto [vx, vy] = dataSource(hsd, i, j);
        if (std::isnan(vx) || std::isnan(vy)) {
            noDataCount++;  // Projection::ProjectPixel invalid for this cell
            continue;
        }
        const double speed = std::sqrt(vx * vx + vy * vy);
        if (speed < 1e-6) {
            calmCount++;  // genuinely (near-)zero -- direction undefined, skip
            continue;
        }

        // Meteorological "from" convention: the direction the flow is
        // coming FROM, degrees clockwise from North.
        double bearing_deg = std::atan2(-vx, -vy) * 180.0 / M_PI;
        if (bearing_deg < 0.0)
            bearing_deg += 360.0;

        const int dirBin = (int)std::floor(bearing_deg / sectorWidth + 0.5) % N_DIR_BINS;

        int speedBin = N_SPEED_BINS - 1;
        for (int b = 0; b < N_SPEED_BINS - 1; ++b) {
            if (speed < speedBinEdges[b + 1]) { speedBin = b; break; }
        }

        counts[dirBin][speedBin]++;
        grandTotal++;
    }

    // Self-check: every counted sample increments exactly one counts[d][s]
    // and grandTotal together, so this sum is guaranteed equal by
    // construction -- logged anyway so a mismatch here (which would mean an
    // actual coding bug, not a data-coverage issue) can never go unnoticed.
    int countsSum = 0;
    for (const auto& row : counts)
        for (int c : row)
            countsSum += c;

    LOGR("{}Rose: {} candidate cells (land/open-boundary already excluded); "
         "{} used for the rose ({:.1f}%), {} no-data (invalid projection), {} calm. "
         "counts-sum check: {} (should equal usedCount)",
         title, sampleCells.size(), grandTotal,
         sampleCells.empty() ? 0.0 : 100.0 * grandTotal / sampleCells.size(),
         noDataCount, calmCount, countsSum);

    BuildRoseGeometry(counts, grandTotal);
}

void WindRoseWidget::BuildRoseGeometry(const std::array<std::array<int, N_SPEED_BINS>, N_DIR_BINS>& counts,
                                       int grandTotal)
{
    vtkNew<vtkPoints> points;
    vtkNew<vtkCellArray> polys;
    vtkNew<vtkUnsignedCharArray> cellColors;
    cellColors->SetNumberOfComponents(3);
    cellColors->SetName("Colors");

    if (grandTotal > 0) {
        constexpr double sectorWidth = 360.0 / N_DIR_BINS;
        const double wedgeHalfWidth = sectorWidth * (1.0 - SECTOR_GAP_FRACTION) / 2.0;
        // Conventional wind-rose radial scale: a full ring at OUTER_RADIUS_PX
        // represents FREQ_MAX_FRACTION (e.g. 33%) of all samples -- fixed,
        // not renormalized to whatever the busiest direction happens to be
        // this frame, so the scale reads the same on every frame.
        const double pxPerSample = OUTER_RADIUS_PX / (FREQ_MAX_FRACTION * grandTotal);

        for (int d = 0; d < N_DIR_BINS; ++d) {
            int dirTotal = 0;
            for (int s = 0; s < N_SPEED_BINS; ++s)
                dirTotal += counts[d][s];
            if (dirTotal == 0)
                continue;

            const double bearingCenter = d * sectorWidth;
            const double bearing0 = bearingCenter - wedgeHalfWidth;
            const double bearing1 = bearingCenter + wedgeHalfWidth;

            double innerR = 0.0;
            for (int s = 0; s < N_SPEED_BINS; ++s) {
                if (counts[d][s] == 0)
                    continue;
                const double outerR = innerR + counts[d][s] * pxPerSample;

                constexpr int nArcPts = ARC_SUBDIV + 2;
                vtkIdType polyPts[2 * nArcPts];
                int nPts = 0;
                // Outer arc, bearing0 -> bearing1
                for (int k = 0; k < nArcPts; ++k) {
                    const double t = (double)k / (nArcPts - 1);
                    const double bearing = bearing0 + t * (bearing1 - bearing0);
                    const double rad = bearing * M_PI / 180.0;
                    polyPts[nPts++] = points->InsertNextPoint(outerR * std::sin(rad), outerR * std::cos(rad), 0.0);
                }
                // Inner arc, bearing1 -> bearing0 (reversed, so the polygon closes correctly)
                for (int k = nArcPts - 1; k >= 0; --k) {
                    const double t = (double)k / (nArcPts - 1);
                    const double bearing = bearing0 + t * (bearing1 - bearing0);
                    const double rad = bearing * M_PI / 180.0;
                    polyPts[nPts++] = points->InsertNextPoint(innerR * std::sin(rad), innerR * std::cos(rad), 0.0);
                }

                polys->InsertNextCell(nPts, polyPts);

                const auto rgb = ColorMap::getColor(ColorMap::Palette::ANSYS,
                                                    (float)s / (N_SPEED_BINS - 1));
                cellColors->InsertNextTuple3(rgb[0], rgb[1], rgb[2]);

                innerR = outerR;
            }
        }
    }

    polyData->SetPoints(points);
    polyData->SetPolys(polys);
    polyData->GetCellData()->SetScalars(cellColors);
    polyData->Modified();
}
