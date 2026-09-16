#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QLabel>
#include <QDoubleSpinBox>
#include <QSpinBox>
#include <QComboBox>
#include <QSlider>
#include <QMenuBar>
#include <QAction>
#include <QStatusBar>
#include <QLocale>
#include <QSettings>

#include <QVTKOpenGLNativeWidget.h>
#include <vtkGenericOpenGLRenderWindow.h>
#include <vtkRenderWindow.h>
#include <vtkRenderer.h>
#include <vtkCamera.h>
#include <vtkNew.h>
#include <vtkInteractorStyleImage.h>
#include <vtkWindowToImageFilter.h>
#include <vtkPNGWriter.h>

#include "visual_representation.h"

#include "parameterparser.h"
#include "host_side_data.h"

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

    void LoadParameterFile(QString fileName);

private Q_SLOTS:
    void openJsonFile_triggered();
    void resetCamera_triggered();
    void determine_crop_extents_triggered();
    void precompute_temp_wind_cache_triggered();
    void precompute_current_cache_triggered();
    void generate_all_cache_triggered();
    void export_temperature_time_series_triggered();
    void thermal_spinup_triggered();
    void comboboxIndexChanged_visualizations(int index);
    void comboboxIndexChanged_hedgehog(int index);
    void limits_changed(double val);
    void hedgehog_scale_changed(double val);
    void arrowhead_size_changed(int size);
    void flowTimeSliderChanged(int value);
    void closeEvent(QCloseEvent *event) override;

private:
    void setupUI();
    void createMenuBar();
    void loadSettings();
    void saveSettings();
    void loadCameraState();
    void saveCameraState();

    // Formats an absolute Unix epoch as "YYYY-MM-DD HH:MM UTC" -- the date/
    // time the currently-displayed CARRA1/GLO12 data was rendered for.
    static QString FormatUtcDateTime(long long epoch);

    ParameterParser params;
    HostSideData hsd;
    VisualRepresentation representation;

    // Visualization controls
    QComboBox *comboBox_visualizations;
    QComboBox *comboBox_hedgehog;
    QDoubleSpinBox *qdsbValRangeFrom;
    QDoubleSpinBox *qdsbValRangeTo;
    QDoubleSpinBox *qdsbTransparency;
    QDoubleSpinBox *qdsbHedgehogScale;
    QSpinBox *qsbArrowheadSize;
    QSlider *flowTimeSlider;              // slider for flow field time control
    // Absolute Unix epoch bounds mapped onto flowTimeSlider's 0..1000 range;
    // set from hsd.prms.PreSimulationStartTime/SimulationEndDateTime (with
    // fallbacks) each time a JSON config is loaded -- see LoadParameterFile.
    long long sliderStartTime = 0;
    long long sliderEndTime = 0;
    QLabel *statusLabel;                  // status bar label for point count
    QLabel *timeLabel;                    // status bar label for flow time display

    QAction *showTitleAction = nullptr;   // "View > Show Title" toggle
    QAction *renderSpinupFramesAction = nullptr;   // "Tools > Render Spin-Up Frames" toggle

    // Renders visOpt into the live VTK window and saves it as a PNG under
    // outDir/frame_NNNNN.png. Used only by thermal_spinup_triggered() when
    // renderSpinupFramesAction is checked -- a rarely-used diagnostic path.
    void CaptureSpinupFrame(VisOpt::Type visOpt, const QString &outDir, long long frameNumber);

    // VTK
    vtkNew<vtkGenericOpenGLRenderWindow> renderWindow;
    QVTKOpenGLNativeWidget *qt_vtk_widget;
    vtkNew<vtkRenderer> renderer;
    vtkNew<vtkInteractorStyleImage> interactorStyle;

    QString settingsFileName;
};
#endif // MAINWINDOW_H
