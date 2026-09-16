#include "preparer_mainwindow.h"
#include "visual_representation.h"
#include "simulation/parameters_sim.h"
#include "data_preparer.h"
#include <QFileDialog>
#include <QMetaEnum>
#include <QVBoxLayout>
#include <QToolBar>
#include <QLabel>
#include <QMenu>
#include <QAction>
#include <QSettings>
#include <QDir>
#include <QFileInfo>
#include <QFile>
#include <QTextStream>
#include <QDebug>
#include <spdlog/spdlog.h>
#include <ctime>
#include <algorithm>
#include <cmath>
#include <QCoreApplication>
#include <QEventLoop>
#include <QThread>
#include <QMessageBox>
#include <QProgressDialog>
#include <QDialog>
#include <QFormLayout>
#include <QDialogButtonBox>
#include <filesystem>
#include <H5Cpp.h>

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent), representation(hsd)
{
    setWindowTitle("Preparer");
    setGeometry(100, 100, 1200, 800);

    // Setup settings file
    settingsFileName = QDir::currentPath() + "/preparer.ini";

    // Setup UI programmatically
    setupUI();
    createMenuBar();

    // Loading a CARRA1/GLO12 frame is dominated by file I/O (~seconds), long
    // enough that the OS flags the window as "not responding" if we just
    // block. Pump the event loop between frame loads instead -- excluding
    // user input specifically, so a stray click/keypress during the pump
    // can't reenter flowTimeSliderChanged (or similar) mid-load and corrupt
    // interpolator state. LoadWindFrame/LoadGLO12Frame can also run on a
    // background preload thread, where calling processEvents() would be
    // unsafe -- guard for that explicitly.
    auto pumpEventsIfOnGuiThread = []() {
        if (QThread::currentThread() == QCoreApplication::instance()->thread()) {
            QCoreApplication::processEvents(QEventLoop::ExcludeUserInputEvents);
        }
    };
    hsd.windInterp.on_loading_progress = pumpEventsIfOnGuiThread;
    hsd.currentInterp.on_loading_progress = pumpEventsIfOnGuiThread;

    // The trackbar jumps around non-sequentially (and in both directions),
    // so background-preloading "the next frame" is usually wasted work here
    // -- unlike gplate/cplate, which always advance sequentially.
    hsd.windInterp.SetPreloadEnabled(false);
    hsd.currentInterp.SetPreloadEnabled(false);

    // Load saved settings (visualization mode and camera state)
    loadSettings();
    loadCameraState();
}

MainWindow::~MainWindow()
{
}

void MainWindow::setupUI()
{
    // Create central VTK widget
    qt_vtk_widget = new QVTKOpenGLNativeWidget();
    qt_vtk_widget->setRenderWindow(renderWindow);
    setCentralWidget(qt_vtk_widget);

    // Configure renderer
    renderer->SetBackground(0.9, 0.9, 0.85);
    renderWindow->AddRenderer(renderer);
    renderWindow->GetInteractor()->SetInteractorStyle(interactorStyle);

    // Create toolbar with visualization controls
    QToolBar *toolBar = addToolBar("Visualization");

    // Combobox for visualization options
    comboBox_visualizations = new QComboBox();
    toolBar->addWidget(comboBox_visualizations);

    toolBar->addSeparator();

    comboBox_hedgehog = new QComboBox();
    toolBar->addWidget(comboBox_hedgehog);

    toolBar->addSeparator();

    // Range label and [from, to] spinboxes -- see VisOpt::range_from/range_to
    QLabel *lbl1 = new QLabel("range:");
    toolBar->addWidget(lbl1);

    qdsbValRangeFrom = new QDoubleSpinBox();
    qdsbValRangeFrom->setRange(-1e7, 1e7);
    qdsbValRangeFrom->setValue(0);
    qdsbValRangeFrom->setDecimals(4);
    qdsbValRangeFrom->setSingleStep(0.1);
    toolBar->addWidget(qdsbValRangeFrom);

    QLabel *lblRangeTo = new QLabel("to:");
    toolBar->addWidget(lblRangeTo);

    qdsbValRangeTo = new QDoubleSpinBox();
    qdsbValRangeTo->setRange(-1e7, 1e7);
    qdsbValRangeTo->setValue(1);
    qdsbValRangeTo->setDecimals(4);
    qdsbValRangeTo->setSingleStep(0.1);
    toolBar->addWidget(qdsbValRangeTo);

    // Transparency label and spinbox
    QLabel *lbl2 = new QLabel("tr:");
    toolBar->addWidget(lbl2);

    qdsbTransparency = new QDoubleSpinBox();
    qdsbTransparency->setRange(0, 1000);
    qdsbTransparency->setValue(0);
    qdsbTransparency->setDecimals(1);
    qdsbTransparency->setSingleStep(0.1);
    toolBar->addWidget(qdsbTransparency);

    QLabel *lblHScale = new QLabel(" h_sc:");
    toolBar->addWidget(lblHScale);

    qdsbHedgehogScale = new QDoubleSpinBox();
    qdsbHedgehogScale->setRange(0, 1000);
    qdsbHedgehogScale->setValue(1.0);
    qdsbHedgehogScale->setDecimals(2);
    qdsbHedgehogScale->setSingleStep(0.1);
    toolBar->addWidget(qdsbHedgehogScale);

    // Arrowhead size
    QLabel *lblArrow = new QLabel(" arrow:");
    toolBar->addWidget(lblArrow);
    
    qsbArrowheadSize = new QSpinBox();
    qsbArrowheadSize->setRange(0, 10000);
    qsbArrowheadSize->setValue(100);
    toolBar->addWidget(qsbArrowheadSize);

    // Flow time slider: on its own toolbar row (forced by addToolBarBreak),
    // unlabeled, so it can expand to the full window width instead of being
    // crowded by the fixed-size controls above.
    addToolBarBreak();
    QToolBar *flowTimeToolBar = addToolBar("Flow Time");
    flowTimeSlider = new QSlider(Qt::Horizontal);
    flowTimeSlider->setRange(0, 1000);  // 0 to 1000 subdivisions representing 0 to 1000 seconds
    flowTimeSlider->setValue(0);
    flowTimeSlider->setTracking(false);
    flowTimeSlider->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Preferred);
    flowTimeToolBar->addWidget(flowTimeSlider);

    // Add representation actors to renderer
    renderer->AddActor(representation.textBgActor);
    renderer->AddActor(representation.scalarBarBgActor);
    renderer->AddActor(representation.dateBgActor);
    renderer->AddActor(representation.actor_points);
    renderer->AddActor(representation.raster_actor);
    renderer->AddActor(representation.actor_region_boundary);
    renderer->AddActor(representation.actorText);
    renderer->AddActor(representation.actorTextTitle);
    renderer->AddActor(representation.actorDateText);
    renderer->AddActor(representation.scalarBar);
    
    // Flow Vis Actors
    renderer->AddActor(representation.hedgehog_actor_vectors);

    // Create status bar with point count (left side) and time display (right side)
    statusLabel = new QLabel("Points: 0");
    statusBar()->addWidget(statusLabel);

    // Add permanent widget on the right side of status bar for time display
    timeLabel = new QLabel("Time: 0.0 s");
    timeLabel->setFixedWidth(180);
    timeLabel->setAlignment(Qt::AlignRight);
    statusBar()->addPermanentWidget(timeLabel);

    // Populate visualization options combobox
    for (const auto& [key, text] : VisOpt::descriptions) {
        comboBox_visualizations->addItem(QString::fromStdString(text.first),
                                         QVariant::fromValue(static_cast<int>(key)));
    }

    for (const auto& [key, text] : VisOpt::hedgehog_descriptions) {
        comboBox_hedgehog->addItem(QString::fromStdString(text.first),
                                   QVariant::fromValue(static_cast<int>(key)));
    }

    // Connect signals
    connect(comboBox_visualizations, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &MainWindow::comboboxIndexChanged_visualizations);

    connect(comboBox_hedgehog, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &MainWindow::comboboxIndexChanged_hedgehog);

    connect(qdsbValRangeFrom, QOverload<double>::of(&QDoubleSpinBox::valueChanged),
            this, &MainWindow::limits_changed);
    connect(qdsbValRangeTo, QOverload<double>::of(&QDoubleSpinBox::valueChanged),
            this, &MainWindow::limits_changed);

    connect(qdsbTransparency, QOverload<double>::of(&QDoubleSpinBox::valueChanged),
            this, &MainWindow::limits_changed);

    connect(qdsbHedgehogScale, QOverload<double>::of(&QDoubleSpinBox::valueChanged),
            this, &MainWindow::hedgehog_scale_changed);

    connect(qsbArrowheadSize, QOverload<int>::of(&QSpinBox::valueChanged),
            this, &MainWindow::arrowhead_size_changed);

    connect(flowTimeSlider, QOverload<int>::of(&QSlider::valueChanged),
            this, &MainWindow::flowTimeSliderChanged);
}

void MainWindow::createMenuBar()
{
    // File menu
    QMenu *fileMenu = menuBar()->addMenu("&File");

    QAction *openJsonAction = fileMenu->addAction("&Open JSON");
    openJsonAction->setShortcut(QKeySequence::Open);
    connect(openJsonAction, &QAction::triggered, this, &MainWindow::openJsonFile_triggered);

    fileMenu->addSeparator();

    QAction *exitAction = fileMenu->addAction("E&xit");
    exitAction->setShortcut(QKeySequence::Quit);
    connect(exitAction, &QAction::triggered, this, &QWidget::close);

    // View menu
    QMenu *viewMenu = menuBar()->addMenu("&View");

    QAction *resetCameraAction = viewMenu->addAction("&Reset Camera");
    resetCameraAction->setShortcut(Qt::Key_R);
    connect(resetCameraAction, &QAction::triggered, this, &MainWindow::resetCamera_triggered);

    viewMenu->addSeparator();

    showTitleAction = viewMenu->addAction("Show Title");
    showTitleAction->setCheckable(true);
    showTitleAction->setChecked(true); // Default is shown
    connect(showTitleAction, &QAction::toggled, this, [this](bool checked){
        representation.textBgActor->SetVisibility(checked);
        representation.actorText->SetVisibility(checked);
        representation.actorTextTitle->SetVisibility(checked);
        renderWindow->Render();
    });

    // Tools menu
    QMenu *toolsMenu = menuBar()->addMenu("&Tools");
    QAction *determineCropExtentsAction = toolsMenu->addAction("Save Crop &Extents");
    connect(determineCropExtentsAction, &QAction::triggered, this, &MainWindow::determine_crop_extents_triggered);

    QAction *precomputeTempWindCacheAction = toolsMenu->addAction("Precompute &Temperature+Wind Cache...");
    connect(precomputeTempWindCacheAction, &QAction::triggered, this, &MainWindow::precompute_temp_wind_cache_triggered);

    QAction *precomputeCurrentCacheAction = toolsMenu->addAction("Precompute &Current Cache...");
    connect(precomputeCurrentCacheAction, &QAction::triggered, this, &MainWindow::precompute_current_cache_triggered);

    QAction *generateAllCacheAction = toolsMenu->addAction("Generate &All Cache");
    connect(generateAllCacheAction, &QAction::triggered, this, &MainWindow::generate_all_cache_triggered);

    toolsMenu->addSeparator();
    QAction *exportTempSeriesAction = toolsMenu->addAction("&Export Temperature Time Series...");
    connect(exportTempSeriesAction, &QAction::triggered, this, &MainWindow::export_temperature_time_series_triggered);

    toolsMenu->addSeparator();
    QAction *thermalSpinupAction = toolsMenu->addAction("Run &Thermal Spin-Up...");
    connect(thermalSpinupAction, &QAction::triggered, this, &MainWindow::thermal_spinup_triggered);

    renderSpinupFramesAction = toolsMenu->addAction("Render Spin-Up &Frames");
    renderSpinupFramesAction->setCheckable(true);
}

void MainWindow::openJsonFile_triggered()
{
    QString fileName = QFileDialog::getOpenFileName(this, "Open JSON Configuration", "",
                                                    "JSON Files (*.json);;All Files (*)");
    if (!fileName.isEmpty()) {
        LoadParameterFile(fileName);
    }
}

void MainWindow::resetCamera_triggered()
{
    qDebug() << "MainWindow::on_action_camera_reset_triggered()";
    vtkCamera* camera = renderer->GetActiveCamera();
    renderer->ResetCamera();
    camera->ParallelProjectionOn();
    camera->SetClippingRange(1e-1,1e3);

    const double dx = hsd.prms.cellsize * hsd.prms.InitializationImageSizeX/2;
    const double dy = hsd.prms.cellsize * hsd.prms.InitializationImageSizeY/2;

    camera->SetPosition(dx, dy, 50.);
    camera->SetFocalPoint(dx, dy, 0.);
    camera->SetViewUp(0.0, 1.0, 0.0);
    camera->SetParallelScale(std::min(dx,dy)*1.1);

    camera->Modified();
    renderWindow->Render();
}

void MainWindow::determine_crop_extents_triggered()
{
    qDebug() << "determine_crop_extents_triggered()";
    vtkCamera* cam = renderer->GetActiveCamera();
    
    double focalPoint[3];
    cam->GetFocalPoint(focalPoint);
    
    double parallelScale = cam->GetParallelScale();
    double cellsize = hsd.prms.cellsize;
    if (cellsize <= 0.0) {
        qWarning() << "Invalid cellsize.";
        return;
    }
    
    // Calculate visible area in terms of grid indices
    double visibleHeightGrid = (2.0 * parallelScale) / cellsize;
    
    int* winSize = renderWindow->GetSize();
    double winAspect = 1.0;
    if (winSize[1] > 0) winAspect = (double)winSize[0] / (double)winSize[1];
    double visibleWidthGrid = visibleHeightGrid * winAspect;
    
    // We want a perfect 16:9 crop. So Dimensions must be 16*k x 9*k.
    // Pick a k that best encompasses what we actually see on screen.
    int k_h = std::round(visibleHeightGrid / 9.0);
    int k_w = std::round(visibleWidthGrid / 16.0);
    int k = std::max(k_h, k_w); // Take maximum so we don't truncate the view
    if (k < 1) k = 1;
    
    int crop_width = 16 * k;
    int crop_height = 9 * k;
    
    // Find image focal point indices (world origin is at 0,0)
    int cx = std::round(focalPoint[0] / cellsize);
    int cy = std::round(focalPoint[1] / cellsize);
    
    // Calculate the sub-region bounds natively in full-image coordinates
    int offsetX = cx - crop_width / 2;
    int offsetY = cy - crop_height / 2;
    
    QString extentsMsg = QString("%1, %2, %3, %4")
                                 .arg(offsetX).arg(offsetY)
                                 .arg(crop_width).arg(crop_height);
    
    qDebug() << extentsMsg;
    
    QString filename = QString::fromStdString(hsd.dirs.ProjectDirectory()) + "/crop_extents.txt";
    
    QFile file(filename);
    if (file.open(QIODevice::WriteOnly | QIODevice::Text)) {
        QTextStream out(&file);
        out << extentsMsg << "\n";
        file.close();
        if (statusBar()) {
            statusBar()->showMessage(QString("Saved extents to %1").arg(filename), 5000);
        }
    } else {
        qWarning() << "Failed to save extents to" << filename;
        if (statusBar()) {
            statusBar()->showMessage("Failed to save extents", 5000);
        }
    }
}

void MainWindow::precompute_temp_wind_cache_triggered()
{
    if (hsd.windInterp.GetCARRA1NumFrames() <= 0) {
        QMessageBox::warning(this, "No CARRA1 Data",
            "Load a project with CARRA1 temperature/wind data before precomputing a cache.");
        return;
    }

    const int numFrames = hsd.windInterp.GetCARRA1NumFrames();

    const std::string tempCacheFile = hsd.windInterp.TemperatureCacheFilename();
    const std::string windCacheFile = hsd.windInterp.WindCacheFilename();
    const bool tempExists = std::filesystem::exists(tempCacheFile);
    const bool windExists = std::filesystem::exists(windCacheFile);
    if (tempExists || windExists) {
        QString msg = "The following cache file(s) already exist:\n\n";
        if (tempExists) msg += QString::fromStdString(tempCacheFile) + "\n";
        if (windExists) msg += QString::fromStdString(windCacheFile) + "\n";
        msg += "\nRegenerate?";
        auto reply = QMessageBox::question(this, "Cache Already Exists", msg,
            QMessageBox::Yes | QMessageBox::No, QMessageBox::No);
        if (reply != QMessageBox::Yes) return;
    }
    std::filesystem::create_directories(std::filesystem::path(tempCacheFile).parent_path());
    std::filesystem::create_directories(std::filesystem::path(windCacheFile).parent_path());

    // Disable the main window BEFORE constructing the progress dialog --
    // setEnabled(false) cascades into existing child widgets, and the dialog
    // (parented to `this`) would otherwise get swept up too, disabling its
    // own Abort button along with everything else (see thermal_spinup_triggered,
    // which avoids this the same way).
    setEnabled(false);
    QProgressDialog progress("Precomputing temperature+wind cache...", "Abort", 0, numFrames, this);
    progress.setWindowModality(Qt::WindowModal);
    QCoreApplication::processEvents();

    // The Qt-free rendering/HDF5-writing core lives in DataPreparer so
    // preparer_cli can drive it headlessly too -- this function just wraps
    // it with the progress dialog / abort button and reports the result.
    DataPreparer preparer(hsd);
    bool aborted = false;
    try {
        preparer.GenerateTempWindCache(tempCacheFile, windCacheFile,
            [&](int frameIdx, int total) {
                progress.setValue(frameIdx);
                progress.setLabelText(QString("Rendering frame %1/%2...").arg(frameIdx + 1).arg(total));
                // Deliberately NOT ExcludeUserInputEvents here (unlike the pump
                // used elsewhere in this file for slider drags) -- this dialog's
                // whole point is a clickable Abort button, and that flag would
                // silently discard the mouse click that's supposed to trigger it.
                QCoreApplication::processEvents();
                aborted = progress.wasCanceled();
                return !aborted;
            });
    } catch (const std::exception& e) {
        // Crash loudly and immediately rather than logging and limping on --
        // a console log line is too easy to miss; qFatal prints to stderr
        // and aborts the process (SIGABRT), which isn't.
        qFatal("%s", e.what());
    }

    progress.setValue(numFrames);
    setEnabled(true);

    const QString summary = aborted
        ? QString("Aborted: cache generation stopped early -- see log for frame count written to %1 and %2")
              .arg(QString::fromStdString(tempCacheFile)).arg(QString::fromStdString(windCacheFile))
        : QString("Wrote %1 frames to %2 and %3")
              .arg(numFrames)
              .arg(QString::fromStdString(tempCacheFile)).arg(QString::fromStdString(windCacheFile));
    if (statusBar()) statusBar()->showMessage(summary, 5000);
}

void MainWindow::precompute_current_cache_triggered()
{
    if (hsd.currentInterp.GetGLO12NumFrames() <= 0) {
        QMessageBox::warning(this, "No GLO12 Data",
            "Load a project with GLO12 ocean current data before precomputing a cache.");
        return;
    }

    const int gx = hsd.prms.GridXTotal;
    const int gy = hsd.prms.GridYTotal;
    const int numFrames = hsd.currentInterp.GetGLO12NumFrames();

    const std::string cacheFile = hsd.currentInterp.CurrentCacheFilename();
    if (std::filesystem::exists(cacheFile)) {
        auto reply = QMessageBox::question(this, "Cache Already Exists",
            QString("A current cache already exists at %1.\n\nRegenerate it?")
                .arg(QString::fromStdString(cacheFile)),
            QMessageBox::Yes | QMessageBox::No, QMessageBox::No);
        if (reply != QMessageBox::Yes) return;
    }
    std::filesystem::create_directories(std::filesystem::path(cacheFile).parent_path());

    // Disable the main window BEFORE constructing the progress dialog -- see
    // the identical comment in precompute_temp_wind_cache_triggered.
    setEnabled(false);
    QProgressDialog progress("Precomputing current cache...", "Abort", 0, numFrames, this);
    progress.setWindowModality(Qt::WindowModal);
    QCoreApplication::processEvents();

    // Create the extendible cache datasets up front (0 frames), then extend
    // and write one frame at a time below -- an aborted or failed run still
    // leaves everything written so far as a valid, usable partial cache.
    H5::H5File cacheHfile(cacheFile, H5F_ACC_TRUNC);

    hsize_t initDims[3] = {0, (hsize_t)gx, (hsize_t)gy};
    hsize_t maxDims[3] = {H5S_UNLIMITED, (hsize_t)gx, (hsize_t)gy};
    H5::DSetCreatPropList props;
    hsize_t chunk[3] = {1, (hsize_t)gx, (hsize_t)gy};
    props.setChunk(3, chunk);
    props.setDeflate(1);

    H5::DataSpace vxSpace(3, initDims, maxDims);
    H5::DataSet vxDs = cacheHfile.createDataSet("vx", H5::PredType::NATIVE_FLOAT, vxSpace, props);
    H5::DataSpace vySpace(3, initDims, maxDims);
    H5::DataSet vyDs = cacheHfile.createDataSet("vy", H5::PredType::NATIVE_FLOAT, vySpace, props);
    H5::DataSpace thickSpace(3, initDims, maxDims);
    H5::DataSet thickDs = cacheHfile.createDataSet("thickness", H5::PredType::NATIVE_FLOAT, thickSpace, props);

    H5::DataSpace attSpace(H5S_SCALAR);
    int gxAttr = gx, gyAttr = gy;
    vxDs.createAttribute("GridXTotal", H5::PredType::NATIVE_INT, attSpace).write(H5::PredType::NATIVE_INT, &gxAttr);
    vxDs.createAttribute("GridYTotal", H5::PredType::NATIVE_INT, attSpace).write(H5::PredType::NATIVE_INT, &gyAttr);

    hsize_t timeInitDims[1] = {0};
    hsize_t timeMaxDims[1] = {H5S_UNLIMITED};
    H5::DataSpace timeSpace(1, timeInitDims, timeMaxDims);
    H5::DSetCreatPropList timeProps;
    hsize_t timeChunk[1] = {256};
    timeProps.setChunk(1, timeChunk);
    H5::DataSet timeDs = cacheHfile.createDataSet("timestamps", H5::PredType::NATIVE_INT64, timeSpace, timeProps);

    std::vector<float> scratch_vx, scratch_vy, scratch_thickness;
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

    for (int frameIdx = 0; frameIdx < numFrames; ++frameIdx) {
        if (progress.wasCanceled()) { aborted = true; break; }

        progress.setValue(frameIdx);
        progress.setLabelText(QString("Rendering frame %1/%2...").arg(frameIdx + 1).arg(numFrames));
        QCoreApplication::processEvents();

        long long ts;
        try {
            hsd.currentInterp.RenderCurrentFrameUncached(frameIdx, scratch_vx, scratch_vy, scratch_thickness);
            ts = hsd.currentInterp.GetGLO12Timestamp(frameIdx);
        } catch (const std::exception& e) {
            qFatal("Current cache generation failed at frame %d/%d (%d frame(s) already written to %s): %s",
                   frameIdx, numFrames, framesWritten, cacheFile.c_str(), e.what());
        }

        const hsize_t newFrameCount = (hsize_t)framesWritten + 1;
        writeComponent(vxDs, scratch_vx, newFrameCount);
        writeComponent(vyDs, scratch_vy, newFrameCount);
        writeComponent(thickDs, scratch_thickness, newFrameCount);

        hsize_t timeNewDims[1] = {newFrameCount};
        timeDs.extend(timeNewDims);
        H5::DataSpace timeFilespace = timeDs.getSpace();
        hsize_t timeOffset[1] = {(hsize_t)framesWritten};
        hsize_t timeCount[1] = {1};
        timeFilespace.selectHyperslab(H5S_SELECT_SET, timeCount, timeOffset);
        H5::DataSpace timeMemspace(1, timeCount);
        int64_t ts64 = ts;
        timeDs.write(&ts64, H5::PredType::NATIVE_INT64, timeMemspace, timeFilespace);

        framesWritten++;
    }

    cacheHfile.close();
    progress.setValue(numFrames);
    setEnabled(true);

    hsd.currentInterp.InvalidateCurrentCache();

    const QString summary = aborted
        ? QString("Aborted: wrote %1/%2 frames to %3")
              .arg(framesWritten).arg(numFrames).arg(QString::fromStdString(cacheFile))
        : QString("Wrote %1 frames to %2").arg(framesWritten).arg(QString::fromStdString(cacheFile));
    if (statusBar()) statusBar()->showMessage(summary, 5000);
}

void MainWindow::generate_all_cache_triggered()
{
    // Just runs the individual precompute_*_cache_triggered slots back to
    // back, so the whole set can be kicked off once and left to run
    // unattended instead of manually invoking each menu item and waiting for
    // its progress dialog to finish before starting the next. Each one keeps
    // its own "no data loaded" warning and "cache already exists, regenerate?"
    // confirmation exactly as when invoked individually.
    precompute_temp_wind_cache_triggered();
    precompute_current_cache_triggered();

    if (statusBar()) statusBar()->showMessage("Generate All Cache: finished", 5000);
}

void MainWindow::export_temperature_time_series_triggered()
{
    if (hsd.windInterp.GetCARRA1NumFrames() <= 0) {
        QMessageBox::warning(this, "No CARRA1 Data",
            "Load a project with CARRA1 temperature data before exporting a time series.");
        return;
    }

    QDialog dialog(this);
    dialog.setWindowTitle("Export Temperature Time Series");
    QFormLayout *form = new QFormLayout(&dialog);

    QDoubleSpinBox *latSpin = new QDoubleSpinBox(&dialog);
    latSpin->setRange(-90.0, 90.0);
    latSpin->setDecimals(4);
    latSpin->setValue(80.0);
    form->addRow("Latitude:", latSpin);

    QDoubleSpinBox *lonSpin = new QDoubleSpinBox(&dialog);
    lonSpin->setRange(-180.0, 180.0);
    lonSpin->setDecimals(4);
    lonSpin->setValue(-69.0);
    form->addRow("Longitude:", lonSpin);

    QDialogButtonBox *buttons = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel, &dialog);
    connect(buttons, &QDialogButtonBox::accepted, &dialog, &QDialog::accept);
    connect(buttons, &QDialogButtonBox::rejected, &dialog, &QDialog::reject);
    form->addRow(buttons);

    if (dialog.exec() != QDialog::Accepted) return;

    double lat = latSpin->value();
    double lon = lonSpin->value();

    QString defaultName = QString("carra1_temp_%1_%2.csv")
                               .arg(lat, 0, 'f', 4).arg(lon, 0, 'f', 4);
    QString csvPath = QFileDialog::getSaveFileName(this, "Export Temperature Time Series", defaultName, "CSV Files (*.csv)");
    if (csvPath.isEmpty()) return;

    hsd.windInterp.ExportTemperatureTimeSeries(lat, lon, csvPath.toStdString());

    if (statusBar()) {
        statusBar()->showMessage(QString("Exported temperature time series to %1").arg(csvPath), 5000);
    }
}

void MainWindow::CaptureSpinupFrame(VisOpt::Type visOpt, const QString &outDir, long long frameNumber)
{
    representation.ChangeVisualizationOption(visOpt);   // sets VisualizingVariable + SynchronizeTopology()
    renderWindow->Render();

    vtkNew<vtkWindowToImageFilter> windowToImageFilter;
    windowToImageFilter->SetInput(renderWindow);
    windowToImageFilter->SetInputBufferTypeToRGB();
    windowToImageFilter->Modified();

    vtkNew<vtkPNGWriter> writer;
    writer->SetInputConnection(windowToImageFilter->GetOutputPort());
    QString path = QString("%1/frame_%2.png").arg(outDir).arg(frameNumber, 5, 10, QChar('0'));
    writer->SetFileName(path.toStdString().c_str());
    writer->Write();
}

void MainWindow::thermal_spinup_triggered()
{
    // See _python_code/heat_equation_spec.md, sections 2 and 6. The actual
    // computation lives in DataPreparer's three ThermalSpinUp* primitives --
    // this function just owns the loop: friendly pre-checks/confirmation,
    // then a plain for loop driving the progress dialog, exactly like
    // MainWindow::precompute_temp_wind_cache_triggered does for cache
    // generation.
    if (hsd.windInterp.GetCARRA1NumFrames() <= 0) {
        QMessageBox::warning(this, "No CARRA1 Data",
            "Load a project with CARRA1 temperature data before running thermal spin-up.");
        return;
    }

    const long long window_start = hsd.prms.PreSimulationStartTime;
    const long long window_end = hsd.prms.SimulationStartTime;
    if (window_start <= 0 || window_end <= window_start) {
        QMessageBox::warning(this, "Spin-Up Window Not Configured",
            "Thermal spin-up needs both PreSimulationStartDate (the lead-in window start) and "
            "SimulationStartDate set in the project JSON, with the former strictly before the latter.");
        return;
    }

    const uint64_t numPoints = hsd.hssoa.size;
    if (numPoints == 0) {
        QMessageBox::warning(this, "No Points", "Load a project with simulation points before running thermal spin-up.");
        return;
    }

    setEnabled(false);
    DataPreparer preparer(hsd);
    auto [firstFrameIdx, lastFrameIdx] = hsd.windInterp.GetFrameRangeForTimeSpan(
        hsd.prms.PreSimulationStartTime, hsd.prms.SimulationStartTime);
    long long N = lastFrameIdx - firstFrameIdx;

    const bool renderFrames = renderSpinupFramesAction && renderSpinupFramesAction->isChecked();
    VisOpt::Type savedVisOpt = representation.VisualizingVariable;
    QString outXiDir, outTempDir;
    if (renderFrames) {
        std::filesystem::path outXi = hsd.dirs.SpinupIceStrengthDirectory();
        std::filesystem::path outTemp = hsd.dirs.SpinupTemperatureDirectory();
        std::filesystem::create_directories(outXi);
        std::filesystem::create_directories(outTemp);
        outXiDir = QString::fromStdString(outXi.string());
        outTempDir = QString::fromStdString(outTemp.string());
    }

    QProgressDialog progress("Running thermal spin-up...", "Abort", 0, (int)N, this);
    progress.setWindowModality(Qt::WindowModal);
    QCoreApplication::processEvents();

    bool aborted = false;
    for (long long s = 1; s <= N; ++s) {
        if (progress.wasCanceled()) { aborted = true; break; }

        progress.setValue((int)(s - 1));
        progress.setLabelText(QString("Thermal spin-up: step %1/%2...").arg(s).arg(N));
        QCoreApplication::processEvents();

        preparer.ThermalSpinUpStep((int)(firstFrameIdx + s));

        if (renderFrames) {
            // Drives the "Day/Hour" text overlay (VisualRepresentation::UpdateTimeText,
            // called at the end of SynchronizeTopology/ChangeVisualizationOption below).
            // Zero reference is SimulationStartDate, same convention as
            // flowTimeSliderChanged -- negative here since spin-up runs before it.
            long long frameEpoch = hsd.windInterp.GetCARRA1Timestamp((int)(firstFrameIdx + s));
            representation.simulationTime = (double)(frameEpoch - hsd.prms.SimulationStartTime);

            preparer.ComputeIceStrengthCPU();
            CaptureSpinupFrame(VisOpt::pt_ice_strength, outXiDir, s);

            // Point the CARRA1 raster at this step's instant -- ThermalSpinUpStep
            // itself samples an exact frame (GetTemperatureFrameExact), while the
            // raster reads through windInterp's blended current-time state
            // (SetTime/GetTemperatureValue); aligning SetTime to the same nominal
            // timestamp keeps the rendered frame in step with what was just
            // computed, close enough for a visual sanity check.
            hsd.windInterp.SetTime(frameEpoch);
            CaptureSpinupFrame(VisOpt::carra1_temperature, outTempDir, s);
        }
    }
    progress.setValue((int)N);

    preparer.ThermalSpinUpSave();

    if (renderFrames) {
        representation.ChangeVisualizationOption(savedVisOpt);
        renderWindow->Render();
    }

    setEnabled(true);

    if (!aborted && statusBar()) {
        statusBar()->showMessage(
            QString("Thermal spin-up complete: %1 steps over %2 points; saved").arg(N).arg(numPoints), 5000);
    }
}

void MainWindow::LoadParameterFile(QString fileName)
{
    params.LoadParamsFile(fileName.toStdString());

    // Full path in the title so it stays visible in the taskbar/window list
    // for the whole (possibly long) prep + thermal spin-up run below.
    setWindowTitle("Preparer - " + QFileInfo(fileName).absoluteFilePath());

    // pt_thickness/grid_thickness visualization range: thickness is generated
    // within [ThicknessFrom, ThicknessTo], so mirror that into both options'
    // [from, to] color-mapping bounds (see VisOpt::range_from/range_to).
    // ParameterParser defaults both to 1.0 ("no scaling") when simulation.json
    // omits them, so only override when the JSON gives a real [min, max].
    if (params.ThicknessTo > params.ThicknessFrom) {
        VisOpt::range_from[VisOpt::pt_thickness] = VisOpt::range_from[VisOpt::grid_thickness] = params.ThicknessFrom;
        VisOpt::range_to[VisOpt::pt_thickness] = VisOpt::range_to[VisOpt::grid_thickness] = params.ThicknessTo;
    }

    // Loading a config involves grid/point prep plus CARRA1/GLO12 frame
    // loads (seconds of file I/O) -- disable input for the duration so the
    // window doesn't look interactive while mid-load (see
    // flowTimeSliderChanged for the same pattern, applied per-frame there).
    setEnabled(false);
    try {
        // Construct file paths from config file directory (images are in same dir as JSON)
        std::string configDir = params.ConfigFileDirectory;
        // Land mask is optional - only construct path if provided
        std::string landmaskPath = params.ImageLandMask.empty() ? "" : (configDir + "/" + params.ImageLandMask);
        std::string colorPath = configDir + "/" + params.ImageColor;
        std::string icemaskPath = params.ImageIceMask.empty() ? "" : (configDir + "/" + params.ImageIceMask);
        // Crushed mask is optional - only construct path if provided
        std::string crushedmaskPath = params.ImageCrushedMask.empty() ? "" : (configDir + "/" + params.ImageCrushedMask);
        // Cracked mask is optional - only construct path if provided
        std::string crackedmaskPath = params.ImageCrackedMask.empty() ? "" : (configDir + "/" + params.ImageCrackedMask);
        // Thickness mask is optional - only construct path if provided
        std::string thicknessmaskPath = params.ImageThicknessMask.empty() ? "" : (configDir + "/" + params.ImageThicknessMask);
        // Footprint (open boundary) mask is optional - only construct path if provided
        std::string footprintmaskPath = params.ImageFootprintMask.empty() ? "" : (configDir + "/" + params.ImageFootprintMask);
        // Analysis region mask is optional - only construct path if provided
        std::string analysismaskPath = params.AnalysisRegionMask.empty() ? "" : (configDir + "/" + params.AnalysisRegionMask);

        // --- Load projection + CARRA1/GLO12 paths (single consolidated
        // JSON file -- fileName -- shared with the ParameterParser fields
        // read above). Must happen BEFORE PrepareGridAndPoints, since
        // cellsize is now derived from the projection instead of a
        // separately specified physical dimension.
        std::filesystem::path configPath(fileName.toStdString());
        std::filesystem::path simConfigDir = configPath.parent_path();
        std::vector<std::string> carra1Files;
        std::map<std::string, std::string> simParseResult = hsd.prms.ParseFile(fileName.toStdString(), &carra1Files);
        // The one and only DirectoryManager instance for this run -- see
        // ParameterParser's header comment for why it doesn't have its own.
        hsd.dirs.Configure(simParseResult["RuntimeDirectory"], simParseResult["ProjectName"]);
        std::string projectDir = hsd.dirs.ProjectDirectory();
        std::filesystem::create_directories(projectDir);

        // Snapshot-to-load-instead-of-generating is optional - only construct path if
        // provided. Relative to this project's own snapshots directory
        // (RuntimeDirectory-derived), NOT the simulation.json directory -- mirrors
        // how model.cpp resolves the simulation-runtime "Snapshot" key.
        std::string snapshotToLoadPath;
        if (!params.LoadSnapshotForSpinUp.empty()) {
            std::filesystem::path p(params.LoadSnapshotForSpinUp);
            snapshotToLoadPath = (p.is_relative() ? (std::filesystem::path(hsd.dirs.SnapshotsDirectory()) / p) : p).string();
        }

        // --- Load GLO12 data BEFORE grid/point prep ---
        // Point generation no longer sources thickness from GLO12 (see
        // PopulatePoints_RAM_Optimized -- always the image mask now); GLO12
        // current/tide/thickness data is still loaded here for the live
        // flow-time-slider visualization (VisOpt::v_glo12_thickness etc).
        if (simParseResult.count("GLO12Data") && !simParseResult["GLO12Data"].empty()) {
            std::filesystem::path glo12Path = simConfigDir / simParseResult["GLO12Data"];
            if (std::filesystem::exists(glo12Path)) {
                hsd.currentInterp.SetGLO12Path(glo12Path.string());
                hsd.currentInterp.SetEnabled(true);
                spdlog::info("GLO12 Ocean Data loaded from: {}", glo12Path.string());
            } else {
                spdlog::warn("Warning: GLO12Data path not found: {}", glo12Path.string());
            }
        }

        if (simParseResult.count("GLO12Tides") && !simParseResult["GLO12Tides"].empty()) {
            std::filesystem::path glo12TidesPath = simConfigDir / simParseResult["GLO12Tides"];
            if (std::filesystem::exists(glo12TidesPath)) {
                hsd.currentInterp.SetGLO12TidesPath(glo12TidesPath.string());
                spdlog::info("GLO12 Tides Data loaded from: {}", glo12TidesPath.string());
            } else {
                spdlog::warn("Warning: GLO12Tides path not found: {}", glo12TidesPath.string());
            }
        }

        if (simParseResult.count("GLO12ThicknessData") && !simParseResult["GLO12ThicknessData"].empty()) {
            std::filesystem::path glo12ThicknessPath = simConfigDir / simParseResult["GLO12ThicknessData"];
            if (std::filesystem::exists(glo12ThicknessPath)) {
                hsd.currentInterp.SetGLO12ThicknessPath(glo12ThicknessPath.string());
                hsd.currentInterp.SetEnabled(true);
                spdlog::info("GLO12 Thickness Data loaded from: {}", glo12ThicknessPath.string());
            } else {
                spdlog::warn("Warning: GLO12ThicknessData path not found: {}", glo12ThicknessPath.string());
            }
        }

        // Unified grid and points preparation (loads images once, flips them, then processes)
        // Note: allocate_dense_grid = false to save memory (preparer doesn't need simulation grid state)
        DataPreparer preparer(hsd);
        preparer.PrepareGridAndPoints(landmaskPath, colorPath, icemaskPath, crushedmaskPath, crackedmaskPath,
                                 projectDir, params.PointsPerCell,
                                 params.ThicknessFrom, params.ThicknessTo,
                                 params.ProportionOfCrackedPoints, params.StdDevOfThickness,
                                 thicknessmaskPath,
                                 true, true, params.GeneratePoints,
                                 footprintmaskPath, analysismaskPath, snapshotToLoadPath);

        spdlog::info("Preparer: Grid and Points prepared successfully");

        // Update status bar with point count (with thousands separator)
        unsigned numPoints = hsd.hssoa.size;
        QLocale locale = QLocale::English;
        QString pointCountStr = locale.toString((int)numPoints);
        statusLabel->setText(QString("Points: %1").arg(pointCountStr));

        // --- Resolution/loading of CARRA1 Path(s) ---
        // Must happen AFTER PrepareGridAndPoints: WindInterpolator::
        // PrecomputeGridMapping() (triggered by SetCARRA1Path below) sizes
        // its mapping arrays from prms.GridXTotal/GridYTotal, which are only
        // set once DetermineExtents() runs inside PrepareGridAndPoints.
        // Loading CARRA1 any earlier sizes those arrays to 0 and crashes the
        // very next time a wind frame is loaded.
        //
        // 1. Check if the consolidated JSON provides CARRA1Data (one path or
        // an array of paths, in chronological order).
        std::vector<std::string> finalCARRA1Paths;
        bool useCARRA1 = false;

        for (const std::string& carra1File : carra1Files) {
            std::filesystem::path p(carra1File);
            if (p.is_relative()) {
                p = simConfigDir / p;
            }
            if (std::filesystem::exists(p)) {
                finalCARRA1Paths.push_back(p.string());
                useCARRA1 = true;
            } else {
                spdlog::warn("Warning: CARRA1Data path not found: {}", p.string());
            }
        }

        // 2. Apply FINAL decision
        hsd.prms.UseWindData = useCARRA1;
        if (useCARRA1) {
             for (const std::string& p : finalCARRA1Paths) {
                 hsd.windInterp.SetCARRA1Path(p);
                 spdlog::info("Preparer: CARRA1 Wind Data loaded from: {}", p);
             }
             hsd.windInterp.SetEnabled(true);
        }

        // Seeds every point's thermal spin-up state now, so a misconfigured window
        // fails at project load rather than only when spin-up is later triggered.
        if (hsd.prms.PreSimulationStartTime > 0) {
            preparer.InitThermalState();
        }

        // --- Data-visualization slider bounds ---
        // Span [PreSimulationStartDate, SimulationEndDate] so the slider can
        // exercise the full period a CPU pre-simulation would need (e.g. May 1
        // through the real simulation's end), not just the real sim's start.
        // Both fall back to SimulationStartTime when unset in the JSON, which
        // degenerates to a single-instant slider rather than crashing.
        sliderStartTime = hsd.prms.PreSimulationStartTime > 0 ? hsd.prms.PreSimulationStartTime : hsd.prms.SimulationStartTime;
        sliderEndTime = hsd.prms.SimulationEndDateTime > 0 ? hsd.prms.SimulationEndDateTime : hsd.prms.SimulationStartTime;
        if (sliderEndTime < sliderStartTime) sliderEndTime = sliderStartTime;

        // Default the slider to SimulationStartDate, not the span's start
        // (PreSimulationStartDate) -- that's the actual simulation start,
        // so it's the most useful one to see immediately rather than
        // having to scroll to it.
        long long initialTime = std::clamp(hsd.prms.SimulationStartTime, sliderStartTime, sliderEndTime);
        int initialSliderValue = 0;
        if (sliderEndTime > sliderStartTime) {
            double frac = (double)(initialTime - sliderStartTime) / (double)(sliderEndTime - sliderStartTime);
            initialSliderValue = std::clamp((int)std::lround(frac * 1000.0), 0, 1000);
        }
        flowTimeSlider->blockSignals(true);
        flowTimeSlider->setValue(initialSliderValue);
        flowTimeSlider->blockSignals(false);
        timeLabel->setText(FormatUtcDateTime(initialTime));

        // Ensure everything is synchronized at the slider's initial position
        // (SimulationStartDate -- see initialTime above)
        hsd.currentInterp.SetTime((double)initialTime);
        hsd.windInterp.SetTime((double)initialTime);
        representation.simulationTime = (double)(initialTime - hsd.prms.SimulationStartTime); // 0 when initialTime == SimulationStartDate

        // Update visualization
        representation.SynchronizeTopology();
        renderWindow->Render();
    } catch (const std::exception &e) {
        spdlog::error("Preparer error: {}", e.what());
    }
    setEnabled(true);
}

void MainWindow::comboboxIndexChanged_visualizations(int index)
{
    representation.ChangeVisualizationOption(index);
    qdsbValRangeFrom->blockSignals(true);
    qdsbValRangeTo->blockSignals(true);
    qdsbTransparency->blockSignals(true);
    qdsbValRangeFrom->setValue(VisOpt::range_from[index]);
    qdsbValRangeTo->setValue(VisOpt::range_to[index]);
    qdsbTransparency->setValue(VisOpt::transparency_coeffs[index]);
    qdsbValRangeFrom->blockSignals(false);
    qdsbValRangeTo->blockSignals(false);
    qdsbTransparency->blockSignals(false);
    renderWindow->Render();
}

void MainWindow::comboboxIndexChanged_hedgehog(int index)
{
    representation.HedgehogVariable = (VisOpt::HedgehogType)index;
    representation.SynchronizeTopology();
    qdsbHedgehogScale->blockSignals(true);
    qdsbHedgehogScale->setValue(VisOpt::hedgehog_ranges[index]);
    qdsbHedgehogScale->blockSignals(false);
    renderWindow->Render();
}

void MainWindow::limits_changed(double val)
{
    int idx = (int)representation.VisualizingVariable;
    VisOpt::range_from[idx] = qdsbValRangeFrom->value();
    VisOpt::range_to[idx] = qdsbValRangeTo->value();
    VisOpt::transparency_coeffs[idx] = qdsbTransparency->value();
    representation.SynchronizeTopology();
    renderWindow->Render();
}

void MainWindow::hedgehog_scale_changed(double val_)
{
    int idx = (int)representation.HedgehogVariable;
    VisOpt::hedgehog_ranges[idx] = qdsbHedgehogScale->value();
    representation.SynchronizeTopology();
    renderWindow->Render();
}

void MainWindow::arrowhead_size_changed(int size)
{
    representation.SetArrowheadSize(size * 10.0);
    renderWindow->Render();
}

QString MainWindow::FormatUtcDateTime(long long epoch)
{
    std::time_t tt = (std::time_t)epoch;
    char buf[32];
    std::tm tm_utc;
    gmtime_r(&tt, &tm_utc);
    std::strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M UTC", &tm_utc);
    return QString::fromStdString(buf);
}

void MainWindow::flowTimeSliderChanged(int value)
{
    // Convert slider value (0-1000) to an absolute Unix epoch spanning
    // [sliderStartTime, sliderEndTime] -- see LoadParameterFile, where these
    // are derived from PreSimulationStartDate/SimulationEndDate (falling
    // back to SimulationStartDate). SetTime's contract is an absolute epoch
    // matching CARRA1/GLO12 timestamps directly (see WindInterpolator::
    // ProcessCARRA1 / CurrentInterpolator::ProcessGLO12).
    double alpha = static_cast<double>(value) / 1000.0;
    double abs_t = sliderStartTime + alpha * (double)(sliderEndTime - sliderStartTime);

    QString dateStr = FormatUtcDateTime((long long)abs_t);
    LOGR("MainWindow::flowTimeSliderChanged: slider_value={}, abs_t={}, date={}", value, abs_t, dateStr.toStdString());

    // Loading a new frame can take a couple of seconds (file I/O). Disable
    // input widget-wide -- on top of on_loading_progress's event pumping --
    // as a second line of defense against reentrant slot calls, and show a
    // "loading" indicator so the wait doesn't look like a freeze.
    setEnabled(false);
    timeLabel->setText(dateStr + " (loading...)");
    QCoreApplication::processEvents(QEventLoop::ExcludeUserInputEvents);

    bool ocean_changed = hsd.currentInterp.SetTime(abs_t);
    bool wind_changed = hsd.windInterp.SetTime(abs_t);
    LOGR("MainWindow::flowTimeSliderChanged: ocean_changed={}; wind_changed={}", ocean_changed, wind_changed);

    setEnabled(true);

    // Update time display in status bar
    timeLabel->setText(dateStr);

    // Redraw with new flow field frame
    LOGR("MainWindow::flowTimeSliderChanged: Calling SynchronizeTopology and Render");
    // representation.simulationTime feeds UpdateTimeText()'s "Day/Hour"
    // display. Zero reference is SimulationStartDate (not the slider's own
    // start, PreSimulationStartDate) so the display reads "how far is this
    // reanalysis data from the real simulation's start" -- negative values
    // (pre-simulation dates) are expected and handled (see UpdateTimeText).
    representation.simulationTime = abs_t - (double)hsd.prms.SimulationStartTime;
    representation.SynchronizeTopology();
    renderWindow->Render();
    LOGR("MainWindow::flowTimeSliderChanged: Done");
}

void MainWindow::loadSettings()
{
    QFileInfo fi(settingsFileName);
    if (!fi.exists()) {
        return;  // File doesn't exist yet, use defaults
    }

    QSettings settings(settingsFileName, QSettings::IniFormat);

    QVariant vis_option = settings.value("vis_option");
    if (!vis_option.isNull()) {
        int option = vis_option.toInt();
        comboBox_visualizations->blockSignals(true);
        comboBox_visualizations->setCurrentIndex(option);
        comboBox_visualizations->blockSignals(false);

        qdsbValRangeFrom->blockSignals(true);
        qdsbValRangeFrom->setValue(VisOpt::range_from[option]);
        qdsbValRangeFrom->blockSignals(false);

        qdsbValRangeTo->blockSignals(true);
        qdsbValRangeTo->setValue(VisOpt::range_to[option]);
        qdsbValRangeTo->blockSignals(false);

        qdsbTransparency->blockSignals(true);
        qdsbTransparency->setValue(VisOpt::transparency_coeffs[option]);
        qdsbTransparency->blockSignals(false);

        representation.ChangeVisualizationOption(option);
        renderWindow->Render();
    }

    QVariant var = settings.value("hedgehog_option");
    if(!var.isNull())
    {
        comboBox_hedgehog->blockSignals(true);
        comboBox_hedgehog->setCurrentIndex(var.toInt());
        comboBox_hedgehog->blockSignals(false);
        representation.HedgehogVariable = (VisOpt::HedgehogType)var.toInt();

        qdsbHedgehogScale->blockSignals(true);
        qdsbHedgehogScale->setValue(VisOpt::hedgehog_ranges[var.toInt()]);
        qdsbHedgehogScale->blockSignals(false);
    }

    QVariant arr_size = settings.value("arrowhead_size");
    if (!arr_size.isNull()) {
        qsbArrowheadSize->blockSignals(true);
        qsbArrowheadSize->setValue(arr_size.toInt());
        qsbArrowheadSize->blockSignals(false);
        representation.SetArrowheadSize(arr_size.toInt() * 10.0);
    }

    bool showTitle = settings.value("show_title", true).toBool();
    showTitleAction->setChecked(showTitle);
}

void MainWindow::saveSettings()
{
    QSettings settings(settingsFileName, QSettings::IniFormat);
    settings.setValue("vis_option", comboBox_visualizations->currentIndex());
    settings.setValue("hedgehog_option", comboBox_hedgehog->currentIndex());
    settings.setValue("arrowhead_size", qsbArrowheadSize->value());
    settings.setValue("show_title", showTitleAction->isChecked());
}

void MainWindow::loadCameraState()
{
    QFileInfo fi(settingsFileName);
    if (!fi.exists()) {
        return;  // File doesn't exist yet, use default camera
    }

    QSettings settings(settingsFileName, QSettings::IniFormat);

    vtkCamera* camera = renderer->GetActiveCamera();
    renderer->ResetCamera();
    camera->ParallelProjectionOn();

    QVariant var = settings.value("camData");
    if (!var.isNull()) {
        double vec[7];
        const double* data = (const double*)var.toByteArray().constData();

        // Copy data safely
        for (int i = 0; i < 7; i++) {
            vec[i] = data[i];
        }

        camera->SetClippingRange(1e-1, 1e4);
        camera->SetViewUp(0.0, 1.0, 0.0);
        camera->SetPosition(vec[0], vec[1], vec[2]);
        camera->SetFocalPoint(vec[3], vec[4], vec[5]);
        camera->SetParallelScale(vec[6]);
        camera->Modified();

        spdlog::debug("Camera state loaded: pos ({}, {}, {}), focal ({}, {}, {}), scale {}",
                      vec[0], vec[1], vec[2], vec[3], vec[4], vec[5], vec[6]);
    }
}

void MainWindow::saveCameraState()
{
    QSettings settings(settingsFileName, QSettings::IniFormat);

    double data[7];
    renderer->GetActiveCamera()->GetPosition(&data[0]);
    renderer->GetActiveCamera()->GetFocalPoint(&data[3]);
    data[6] = renderer->GetActiveCamera()->GetParallelScale();

    QByteArray arr((char*)data, sizeof(data));
    settings.setValue("camData", arr);

    spdlog::debug("Camera state saved: pos ({}, {}, {}), focal ({}, {}, {}), scale {}",
                  data[0], data[1], data[2], data[3], data[4], data[5], data[6]);
}

void MainWindow::closeEvent(QCloseEvent *event)
{
    saveSettings();
    saveCameraState();
    QMainWindow::closeEvent(event);
}
