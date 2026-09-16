// visualizer_mainwindow.h

#ifndef PPMAINWINDOW_H
#define PPMAINWINDOW_H

#include <array>
#include <fstream>
#include <utility>
#include <vector>
#include <QMainWindow>

#include <QSizePolicy>
#include <QPushButton>
#include <QSplitter>
#include <QLabel>
#include <QVBoxLayout>
#include <QTreeWidget>
#include <QProgressBar>
#include <QMenu>
#include <QList>
#include <QDebug>
#include <QComboBox>
#include <QMetaEnum>
#include <QDir>
#include <QString>
#include <QCheckBox>
#include <QFile>
#include <QTextStream>
#include <QIODevice>
#include <QSettings>
#include <QDoubleSpinBox>
#include <QSpinBox>
#include <QFileInfo>
#include <QScrollArea>
#include <QStatusBar>
#include <QSlider>
#include <QToolBar>
#include <QKeyEvent>

#include <QVTKOpenGLNativeWidget.h>

#include <vtkGenericOpenGLRenderWindow.h>
#include <vtkRenderWindow.h>
#include <vtkRenderer.h>
#include <vtkCamera.h>
#include <vtkProperty.h>
#include <vtkNew.h>
#include <vtkInteractorStyleImage.h>
#include <vtkRendererCollection.h>

#include <vtkWindowToImageFilter.h>
#include <vtkPNGWriter.h>
#include <vtkJPEGWriter.h>

#include "visual_representation.h"
#include "host_side_data.h"
#include "render_selector_dialog.h"
#include "wind_rose_widget.h"


class PPMainWindow : public QMainWindow
{
    Q_OBJECT

public:
    PPMainWindow(QWidget *parent = nullptr);
    ~PPMainWindow();

    void closeEvent( QCloseEvent* event ) override;
    void keyPressEvent( QKeyEvent* event ) override;
    void LoadParametersFile(QString fileName);
    void LoadFramesDirectory(QString framesDirectory);
    void TryLoadDefaultFrames();  // Auto-load frames from default location if they exist

private Q_SLOTS:
    void cameraReset_triggered();
    void comboboxIndexChanged_visualizations(int index);
    void limits_changed(double val);
    void comboboxIndexChanged_hedgehog(int index);
    void hedgehog_scale_changed(double val);
    void arrowhead_size_changed(int val);
    void sliderValueChanged(int val);
    void render_frame_triggered();
    void render_all_triggered();
    void render_all_alt_triggered();
    void render_full_grid_snapshot_triggered();
    void export_scalar_bar_svg_triggered();
    void exportRegionDiagnostics_triggered();
    void submitFrame_triggered();
    void forward24h_triggered();
    void jumpToTime_triggered();
    void determine_crop_extents_triggered();
    void openProject_triggered();
    void openFrames_triggered();
    void toggleRenderSelector();
    void toggleContours(bool checked);
    void contourIntervalChanged(double val);
    void windRoseToggled(bool checked);

private:
    HostSideData hsd;
    VisualRepresentation representation;
    std::string currentFrameDirectory;   // Directory containing frame files
    // DirectoryManager-resolved runtime project directory (RuntimeDirectory/input/<ProjectName>
    // when configured, else the JSON config's own directory) -- where output/frames,
    // last_frame.txt, crop_extents.txt, region_diagnostics.csv live. NOT the JSON file's
    // own directory when RuntimeDirectory is configured -- see LoadParametersFile.
    std::string currentProjectDirectory;

    QString settingsFileName;       // includes current dir
    QComboBox *comboBox_visualizations;
    QComboBox *comboBox_hedgehog;
    QDoubleSpinBox *qdsbValRangeFrom;   // low limit for value scale
    QDoubleSpinBox *qdsbValRangeTo;     // high limit for value scale
    QDoubleSpinBox *qdsbTransparency;
    QDoubleSpinBox *qdsbHedgehogScale;
    QSpinBox *qsbArrowheadSize;
    QDoubleSpinBox *qdsbContourInterval;

    QSpinBox *qsbFrameFrom, *qsbFrameTo, *qsbCurrentFrame;
    QPushButton *btnSubmitFrame;
    // Off by default -- when checked, render_all_triggered also writes
    // region_diagnostics.csv alongside the rendered frames (one pass over
    // the frame data instead of two separate 6h+ passes over the same
    // HDD-resident frames). Uses the same AnalysisRegionMask validation
    // and CSV format as the standalone "Export Region Diagnostics" menu
    // action (exportRegionDiagnostics_triggered) -- see WriteRegionDiagnosticsRow.
    QAction *actionExportRegionDiagnosticsOnRenderAll = nullptr;
    QPushButton *btnForward24h;
    QDoubleSpinBox *qdsbJumpToTime;
    QPushButton *btnJumpToTime;
    QSlider *slider2;
    QScrollArea *scrollArea;
    QToolBar *toolBar;
    QStatusBar *statusBar;
    QLabel *dateLabel;   // permanent status-bar widget: full simulated date/time (UTC)

    // VTK
    vtkNew<vtkGenericOpenGLRenderWindow> renderWindow;
    QVTKOpenGLNativeWidget *qt_vtk_widget;
    vtkNew<vtkRenderer> renderer;

    vtkNew<vtkWindowToImageFilter> windowToImageFilter;
    vtkNew<vtkJPEGWriter> writer;
    // other
    vtkNew<vtkInteractorStyleImage> interactor;

    // Render selector dialog
    RenderSelectorDialog* m_renderSelectorDialog;

    // Screen-space wind/current rose overlays (see wind_rose_widget.h); not
    // part of VisualRepresentation on purpose. Same toggle controls both
    // (see windRoseToggled) -- currentRose sits directly above windRose
    // (see their anchorNY values, set in the constructor initializer list).
    WindRoseWidget windRose;
    WindRoseWidget currentRose;

    void UpdateFlowInterpolatorsTime();
    void generate_ffmpeg_script(int frameFrom, int frameTo, std::string dirName = "raster");

    // Shared with exportRegionDiagnostics_triggered's standalone menu
    // action and render_all_triggered's optional in-line export (see
    // actionExportRegionDiagnosticsOnRenderAll) -- same mask validation, cell
    // list, and per-frame row format either way.
    bool HasAnalysisRegionMaskCells() const;
    std::vector<std::pair<int,int>> CollectMaskedCells() const;
    // Assumes hsd already has frameNum's data loaded and the flow
    // interpolators already set to that frame's time (UpdateFlowInterpolatorsTime()) --
    // caller's responsibility, since both call sites already do this for
    // their own purposes (rendering / windRose+currentRose) before this runs.
    void WriteRegionDiagnosticsRow(std::ofstream& csv, int frameNum,
                                    const std::vector<std::pair<int,int>>& maskedCells);

    // Formats an absolute Unix epoch (seconds, UTC) as "YYYY-MM-DD HH:MM UTC" --
    // matches gplate's/preparer's MainWindow::FormatUtcDateTime exactly.
    static QString FormatUtcDateTime(long long epoch);

    // Grid size threshold for slider tracking behavior
    static constexpr int GRID_SIZE_TRACKING_THRESHOLD = 4000;

    // Camera Slots
    std::array<double, 7> m_cameraSlots[5];
    bool m_useCameraSlot[5] = {false, false, false, false, false};
    QAction* showTitleAction = nullptr;
    QAction* showDateTimeAction = nullptr;
    QAction* showColorBarAction = nullptr;
    void captureCameraToSlot(int index);
};
#endif
