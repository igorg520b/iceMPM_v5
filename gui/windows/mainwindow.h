#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>

#include <QSizePolicy>
#include <QPushButton>
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

#include <QVTKOpenGLNativeWidget.h>

#include <vtkGenericOpenGLRenderWindow.h>
#include <vtkRenderWindow.h>
#include <vtkRenderer.h>
#include <vtkCamera.h>
#include <vtkProperty.h>
#include <vtkNew.h>

#include <vtkWindowToImageFilter.h>
#include <vtkPNGWriter.h>
#include <vtkInteractorStyleImage.h>

#include "visual_representation.h"
#include "model.h"
#include "backgroundworker.h"
#include "host_side_data.h"

#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <filesystem>

#include <spdlog/spdlog.h>

QT_BEGIN_NAMESPACE
namespace Ui { class MainWindow; }
QT_END_NAMESPACE

class MainWindow : public QMainWindow
{
    Q_OBJECT
private:
    Ui::MainWindow *ui;

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();
    void closeEvent( QCloseEvent* event ) override;
    Model model;
    void LoadParameterFile(QString qFileName);    // return file name of the point cloud

private Q_SLOTS:
    void quit_triggered();

    void background_worker_paused();
    void simulation_data_ready();

    void simulation_start_pause(bool checked);
    void cameraReset_triggered();
    void load_parameter_triggered();

    void comboboxIndexChanged_visualizations(int index);
    void limits_changed(double val);
    void spinbox_slowdown_value_changed(int val);

    void toggle_scalarbar(bool checked);
    void arrowhead_size_changed(int size);
    void comboboxIndexChanged_hedgehog(int index);
    void hedgehog_scale_changed(double val);
    void particle_size_changed(int size);

private:
    void updateGUI();   // when simulation is started/stopped or when a step is advanced
    void updateActorText();
    void save_binary_data();

    void OpenSnapshot(QString fileName);

    BackgroundWorker *worker;
    VisualRepresentation representation;

    bool cameraNeedsReset = false;
    QString settingsFileName;       // includes current dir
    QLabel *statusLabel;                    // statusbar
    QLabel *labelElapsedTime;
    QLabel *labelStepCount;
    QLabel *labelDateTime;                  // current hindcast date/time (UTC)
    static QString FormatUtcDateTime(long long epoch);
    QComboBox *comboBox_visualizations;
    QComboBox *comboBox_hedgehog;
    QDoubleSpinBox *qdsbValRangeFrom;   // low limit for value scale
    QDoubleSpinBox *qdsbValRangeTo;     // high limit for value scale
    QDoubleSpinBox *qdsbTransparency;
    QDoubleSpinBox *qdsbHedgehogScale;
    QSpinBox *qsbArrowheadSize;
    QSpinBox *qsbParticleSize;
    QSpinBox *qsbIntentionalSlowdown;

    // VTK
    vtkNew<vtkGenericOpenGLRenderWindow> renderWindow;
    QVTKOpenGLNativeWidget *qt_vtk_widget;
    vtkNew<vtkRenderer> renderer;
    vtkNew<vtkInteractorStyleImage> interactorStyle;
    // other
    const std::string outputDirectory = "default_output";


    // screenshots
    const std::string screenshot_directory = "screenshots";

    friend class SpecialSelector2D;

};
#endif // MAINWINDOW_H
