// pp_mainwindow.cpp

#include <QFileDialog>
#include <QList>
#include <QPointF>
#include <QCloseEvent>
#include <QStringList>
#include <QCoreApplication>
#include <QMessageBox>
#include <QProgressDialog>
#include <QMenuBar>

#include <algorithm>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <map>

#include <spdlog/spdlog.h>
#include <omp.h>

#include "visualizer_mainwindow.h"
#include "visual_representation.h"
#include "frame_utils.h"


PPMainWindow::~PPMainWindow() {}

void PPMainWindow::UpdateFlowInterpolatorsTime()
{
    hsd.currentInterp.SetTime(hsd.prms.SimulationTime + hsd.prms.SimulationStartTime);
    hsd.windInterp.SetTime(hsd.prms.SimulationTime + hsd.prms.SimulationStartTime);
}

PPMainWindow::PPMainWindow(QWidget *parent)
    : QMainWindow(parent), representation(hsd),
      windRose("Wind", {0.0, 5.0, 10.0, 15.0, 20.0},
               [](HostSideData& h, int i, int j){ return h.windInterp.GetWindValueGeographic(i, j); },
               0.26),
      currentRose("Current", {0.0, 0.15, 0.30, 0.45, 0.60},
                  [](HostSideData& h, int i, int j){ return h.currentInterp.GetOceanValueGeographic(i, j); },
                  0.70)
{
    hsd.currentInterp.SetEnabled(false);
    hsd.windInterp.SetEnabled(false);

    // The trackbar jumps around non-sequentially (and in both directions),
    // so background-preloading "the next frame" is usually wasted work here
    // -- unlike gplate/cplate, which always advance sequentially. Sticks
    // even across later SetEnabled(true) calls (only Reset()/SetEnabled(false)
    // touch ring-buffer state, never this flag).
    hsd.currentInterp.SetPreloadEnabled(false);
    hsd.windInterp.SetPreloadEnabled(false);
    
    setWindowTitle("MPM Post-Processor");

    // Create central widget with scroll area for VTK rendering
    scrollArea = new QScrollArea(this);
    scrollArea->setWidgetResizable(true);
    setCentralWidget(scrollArea);

    // Setup VTK rendering widget
    qt_vtk_widget = new QVTKOpenGLNativeWidget();
    qt_vtk_widget->setRenderWindow(renderWindow);
    scrollArea->setWidget(qt_vtk_widget);

    // Configure VTK renderer
    renderer->SetBackground(1.0, 1.0, 1.0);
    renderWindow->AddRenderer(renderer);
    renderWindow->GetInteractor()->SetInteractorStyle(interactor);

    // Add VTK actors to renderer
    renderer->AddActor(representation.textBgActor);
    renderer->AddActor(representation.scalarBarBgActor);
    renderer->AddActor(representation.dateBgActor);
    renderer->AddActor(representation.raster_actor);
    renderer->AddActor(representation.actor_region_boundary);

    // Flow Vis Actors
    renderer->AddActor(representation.hedgehog_actor_vectors);

    renderer->AddActor(representation.scalarBar);
    renderer->AddActor(representation.actorText);
    renderer->AddActor(representation.actorTextTitle);
    renderer->AddActor(representation.actorDateText);

    // Wind/current rose overlays -- self-contained, not part of VisualRepresentation.
    windRose.AddToRenderer(renderer);
    currentRose.AddToRenderer(renderer);


    // Create toolbar
    toolBar = addToolBar("Controls");

    // Visualization type selector
    comboBox_visualizations = new QComboBox();
    toolBar->addWidget(comboBox_visualizations);

    toolBar->addSeparator();

    comboBox_hedgehog = new QComboBox();
    toolBar->addWidget(comboBox_hedgehog);

    // Value range control -- [from, to] color-mapping bounds, see
    // VisOpt::range_from/range_to
    toolBar->addWidget(new QLabel("range:"));
    qdsbValRangeFrom = new QDoubleSpinBox();
    qdsbValRangeFrom->setRange(-1e7, 1e7);
    qdsbValRangeFrom->setValue(0);
    qdsbValRangeFrom->setDecimals(4);
    qdsbValRangeFrom->setSingleStep(0.1);
    toolBar->addWidget(qdsbValRangeFrom);

    toolBar->addWidget(new QLabel("to:"));
    qdsbValRangeTo = new QDoubleSpinBox();
    qdsbValRangeTo->setRange(-1e7, 1e7);
    qdsbValRangeTo->setValue(1);
    qdsbValRangeTo->setDecimals(4);
    qdsbValRangeTo->setSingleStep(0.1);
    toolBar->addWidget(qdsbValRangeTo);

    // Transparency spinner
    toolBar->addWidget(new QLabel(" Transparency:"));
    qdsbTransparency = new QDoubleSpinBox(this);
    qdsbTransparency->setRange(0.0, 1000.0);
    qdsbTransparency->setSingleStep(0.1);
    qdsbTransparency->setValue(1.0);
    qdsbTransparency->setDecimals(1);
    toolBar->addWidget(qdsbTransparency);

    toolBar->addWidget(new QLabel(" H_Sc:"));
    qdsbHedgehogScale = new QDoubleSpinBox(this);
    qdsbHedgehogScale->setRange(0.0, 1000.0);
    qdsbHedgehogScale->setSingleStep(0.1);
    qdsbHedgehogScale->setValue(1.0);
    qdsbHedgehogScale->setDecimals(2);
    toolBar->addWidget(qdsbHedgehogScale);

    toolBar->addWidget(new QLabel(" Arrow:"));
    qsbArrowheadSize = new QSpinBox(this);
    qsbArrowheadSize->setRange(0, 10000);
    qsbArrowheadSize->setValue(100);
    toolBar->addWidget(qsbArrowheadSize);

    addToolBarBreak();
    QToolBar *playbackToolBar = addToolBar("Playback");

    // Frame range selection
    qsbFrameFrom = new QSpinBox();
    qsbFrameTo = new QSpinBox();
    playbackToolBar->addWidget(qsbFrameFrom);
    playbackToolBar->addWidget(qsbFrameTo);

    playbackToolBar->addSeparator();
    qsbCurrentFrame = new QSpinBox();
    playbackToolBar->addWidget(qsbCurrentFrame);
    btnSubmitFrame = new QPushButton("Submit");
    playbackToolBar->addWidget(btnSubmitFrame);
    connect(btnSubmitFrame, &QPushButton::clicked, this, &PPMainWindow::submitFrame_triggered);

    playbackToolBar->addSeparator();

    slider2 = new QSlider(Qt::Horizontal);
    playbackToolBar->addWidget(slider2);
    slider2->setTracking(false);
    slider2->setMinimum(1);
    slider2->setMaximum(10000);
    connect(slider2, SIGNAL(valueChanged(int)), this, SLOT(sliderValueChanged(int)));

    btnForward24h = new QPushButton(">");
    btnForward24h->setToolTip("Forward 24 hours");
    btnForward24h->setMaximumWidth(30);
    playbackToolBar->addWidget(btnForward24h);
    connect(btnForward24h, &QPushButton::clicked, this, &PPMainWindow::forward24h_triggered);

    playbackToolBar->addSeparator();

    playbackToolBar->addWidget(new QLabel(" Jump to (s):"));
    qdsbJumpToTime = new QDoubleSpinBox();
    qdsbJumpToTime->setRange(0.0, 1e9);
    qdsbJumpToTime->setDecimals(1);
    qdsbJumpToTime->setSingleStep(60.0);
    playbackToolBar->addWidget(qdsbJumpToTime);
    btnJumpToTime = new QPushButton("Go");
    playbackToolBar->addWidget(btnJumpToTime);
    connect(btnJumpToTime, &QPushButton::clicked, this, &PPMainWindow::jumpToTime_triggered);
    connect(qdsbJumpToTime, &QDoubleSpinBox::editingFinished, this, &PPMainWindow::jumpToTime_triggered);

    // Populate visualization options
    for (const auto& [key, text] : VisOpt::descriptions) {
        comboBox_visualizations->addItem(QString::fromStdString(text.first),
                                         QVariant::fromValue(static_cast<int>(key)));
    }

    connect(comboBox_visualizations, QOverload<int>::of(&QComboBox::currentIndexChanged),
            [&](int index){ comboboxIndexChanged_visualizations(index); });

    for (const auto& [key, text] : VisOpt::hedgehog_descriptions) {
        comboBox_hedgehog->addItem(QString::fromStdString(text.first),
                                   QVariant::fromValue(static_cast<int>(key)));
    }

    connect(comboBox_hedgehog, QOverload<int>::of(&QComboBox::currentIndexChanged),
            [&](int index){ comboboxIndexChanged_hedgehog(index); });

    // Create status bar to show current frame
    statusBar = new QStatusBar(this);
    setStatusBar(statusBar);
    statusBar->showMessage("Ready");

    // Permanent widget (survives showMessage() calls) showing the full
    // simulated date/time, matching gplate's labelDateTime / preparer's timeLabel.
    dateLabel = new QLabel(this);
    dateLabel->setMinimumWidth(160);
    dateLabel->setAlignment(Qt::AlignRight | Qt::AlignVCenter);
    statusBar->addPermanentWidget(dateLabel);

    // Setup camera
    vtkCamera* camera = renderer->GetActiveCamera();
    renderer->ResetCamera();
    camera->ParallelProjectionOn();

    // Create render selector dialog (hidden by default)
    m_renderSelectorDialog = new RenderSelectorDialog(this);
    m_renderSelectorDialog->hide();

    // Load and restore user settings
    settingsFileName = QDir::currentPath() + "/ppcm.ini";
    QFileInfo fi(settingsFileName);

    if(fi.exists())
    {
        QSettings settings(settingsFileName, QSettings::IniFormat);
        QVariant var;

        // Restore camera position
        var = settings.value("camData");
        if(!var.isNull())
        {
            double *vec = (double*)var.toByteArray().constData();
            camera->SetClippingRange(1e-1, 1e4);
            camera->SetViewUp(0.0, 1.0, 0.0);
            camera->SetPosition(vec[0], vec[1], vec[2]);
            camera->SetFocalPoint(vec[3], vec[4], vec[5]);
            camera->SetParallelScale(vec[6]);
            camera->ParallelProjectionOn();
            camera->Modified();
        }



        // Restore visualization option
        var = settings.value("vis_option");
        if(!var.isNull())
        {
            comboBox_visualizations->setCurrentIndex(var.toInt());
        }

        var = settings.value("hedgehog_option");
        if(!var.isNull())
        {
            comboBox_hedgehog->setCurrentIndex(var.toInt());
        }

        var = settings.value("arrowhead_size");
        if (!var.isNull()) {
            qsbArrowheadSize->blockSignals(true);
            qsbArrowheadSize->setValue(var.toInt());
            qsbArrowheadSize->blockSignals(false);
            representation.SetArrowheadSize(var.toInt() * 10.0);
        } else {
            representation.SetArrowheadSize(100 * 10.0);
        }
    }
    else
    {
        cameraReset_triggered();
    }

    // Load render selector saved selections
    m_renderSelectorDialog->loadSelections(settingsFileName);
    int idx = comboBox_visualizations->currentIndex();
    qdsbValRangeFrom->setValue(VisOpt::range_from[idx]);
    qdsbValRangeTo->setValue(VisOpt::range_to[idx]);
    qdsbTransparency->setValue(VisOpt::transparency_coeffs[idx]);
    
    int h_idx = comboBox_hedgehog->currentIndex();
    qdsbHedgehogScale->setValue(VisOpt::hedgehog_ranges[h_idx]);

    // ========== MENUS ==========

    // File menu
    QMenu *fileMenu = menuBar()->addMenu("&File");

    QAction *openProjectAction = fileMenu->addAction("&Open Project...");
    openProjectAction->setShortcut(QKeySequence::Open);
    connect(openProjectAction, &QAction::triggered, this, &PPMainWindow::openProject_triggered);

    QAction *openFramesAction = fileMenu->addAction("&Open Frames...");
    connect(openFramesAction, &QAction::triggered, this, &PPMainWindow::openFrames_triggered);

    fileMenu->addSeparator();

    QAction *exitAction = fileMenu->addAction("E&xit");
    exitAction->setShortcut(QKeySequence::Quit);
    connect(exitAction, &QAction::triggered, this, &QWidget::close);

    // Tools menu
    QMenu *toolsMenu = menuBar()->addMenu("&Tools");

    QAction *resetCameraAction = toolsMenu->addAction("&Reset Camera");
    resetCameraAction->setShortcut(QKeySequence(Qt::CTRL | Qt::SHIFT | Qt::Key_R));
    connect(resetCameraAction, &QAction::triggered, this, &PPMainWindow::cameraReset_triggered);

    QAction *renderFrameAction = toolsMenu->addAction("&Render Frame");
    renderFrameAction->setShortcut(QKeySequence((int)Qt::CTRL | (int)Qt::Key_F));
    connect(renderFrameAction, &QAction::triggered, this, &PPMainWindow::render_frame_triggered);

    QAction *renderFullGridAction = toolsMenu->addAction("Render Full &Grid Snapshot");
    renderFullGridAction->setShortcut(QKeySequence(Qt::CTRL | Qt::SHIFT | Qt::Key_G));
    connect(renderFullGridAction, &QAction::triggered, this, &PPMainWindow::render_full_grid_snapshot_triggered);

    QAction *exportScalarBarAction = toolsMenu->addAction("Export Scalar &Bar (SVG)");
    connect(exportScalarBarAction, &QAction::triggered, this, &PPMainWindow::export_scalar_bar_svg_triggered);

    QAction *renderAllAction = toolsMenu->addAction("&Render All");
    renderAllAction->setShortcut(QKeySequence(Qt::Key_F5));
    connect(renderAllAction, &QAction::triggered, this, &PPMainWindow::render_all_triggered);

    QAction *renderAllAltAction = toolsMenu->addAction("Render All - &Alt");
    connect(renderAllAltAction, &QAction::triggered, this, &PPMainWindow::render_all_alt_triggered);
    
    QAction *renderToJpgAction = toolsMenu->addAction("Render to &JPG Extents");
    connect(renderToJpgAction, &QAction::triggered, this, &PPMainWindow::determine_crop_extents_triggered);

    toolsMenu->addSeparator();
    QAction *exportRegionDiagnosticsAction = toolsMenu->addAction("Export &Region Diagnostics...");
    connect(exportRegionDiagnosticsAction, &QAction::triggered, this, &PPMainWindow::exportRegionDiagnostics_triggered);

    // View menu
    QMenu *viewMenu = menuBar()->addMenu("&View");

    // Camera menu
    QMenu *cameraMenu = menuBar()->addMenu("&Camera");
    for(int i = 0; i < 5; ++i) {
        QAction *saveAction = cameraMenu->addAction(QString("Save Pos %1").arg(i+1));
        connect(saveAction, &QAction::triggered, this, [this, i](){
            captureCameraToSlot(i);
        });
    }
    cameraMenu->addSeparator();
    for(int i = 0; i < 5; ++i) {
        QAction *useAction = cameraMenu->addAction(QString("Use Slot %1").arg(i+1));
        useAction->setCheckable(true);
        connect(useAction, &QAction::toggled, this, [this, i](bool checked){
            m_useCameraSlot[i] = checked;
        });
    }

    cameraMenu->addSeparator();

    showTitleAction = cameraMenu->addAction("Show Title");
    showTitleAction->setCheckable(true);
    showTitleAction->setChecked(true); // Default is shown
    connect(showTitleAction, &QAction::toggled, this, [this](bool checked){
        representation.textBgActor->SetVisibility(checked);
        representation.actorText->SetVisibility(checked);
        representation.actorTextTitle->SetVisibility(checked);
        renderWindow->Render();
    });

    // NOW we can apply the saved settings to the actions
    if(fi.exists()) {
        QSettings settings(settingsFileName, QSettings::IniFormat);
        bool showTitle = settings.value("show_title", true).toBool();
        showTitleAction->setChecked(showTitle);
    }

    QAction *renderSelectorAction = viewMenu->addAction("&Render Selector");
    renderSelectorAction->setCheckable(true);
    renderSelectorAction->setChecked(false);
    connect(renderSelectorAction, &QAction::triggered, this, &PPMainWindow::toggleRenderSelector);

    QAction *loadFlowAction = viewMenu->addAction("Load Flow Data");
    loadFlowAction->setCheckable(true);
    loadFlowAction->setChecked(false);
    connect(loadFlowAction, &QAction::toggled, this, [&](bool checked){
        hsd.currentInterp.SetEnabled(checked);
        hsd.windInterp.SetEnabled(checked);
        UpdateFlowInterpolatorsTime();
        representation.SynchronizeTopology();
        renderWindow->Render();
    });

    QAction *windRoseAction = viewMenu->addAction("Wind/Current Rose");
    windRoseAction->setCheckable(true);
    windRoseAction->setChecked(false);
    connect(windRoseAction, &QAction::toggled, this, &PPMainWindow::windRoseToggled);

    viewMenu->addSeparator();

    showDateTimeAction = viewMenu->addAction("Show Date/Time Label");
    showDateTimeAction->setCheckable(true);
    showDateTimeAction->setChecked(true); // Default is shown
    connect(showDateTimeAction, &QAction::toggled, this, [this](bool checked){
        representation.actorDateText->SetVisibility(checked);
        representation.dateBgActor->SetVisibility(checked);
        renderWindow->Render();
    });

    showColorBarAction = viewMenu->addAction("Show Color Bar");
    showColorBarAction->setCheckable(true);
    showColorBarAction->setChecked(true); // Default is shown
    connect(showColorBarAction, &QAction::toggled, this, [this](bool checked){
        // Goes through ConfigureScalarBar() (rather than setting visibility
        // directly) since that function re-derives visibility from scratch
        // on every frame/variable change -- see userWantsScalarBarVisible.
        representation.userWantsScalarBarVisible = checked;
        representation.ConfigureScalarBar();
        renderWindow->Render();
    });

    if (fi.exists()) {
        QSettings settings(settingsFileName, QSettings::IniFormat);
        showDateTimeAction->setChecked(settings.value("show_date_time", true).toBool());
        showColorBarAction->setChecked(settings.value("show_color_bar", true).toBool());
    }

    viewMenu->addSeparator();

    // Off by default -- see the member declaration (visualizer_mainwindow.h)
    // and render_all_triggered for what this actually triggers. Lives in
    // the View menu rather than the Playback toolbar since it's a one-off
    // setting, not something toggled during routine playback.
    actionExportRegionDiagnosticsOnRenderAll = viewMenu->addAction("Also Export Region Diagnostics CSV on Render All");
    actionExportRegionDiagnosticsOnRenderAll->setCheckable(true);
    actionExportRegionDiagnosticsOnRenderAll->setChecked(false);
    actionExportRegionDiagnosticsOnRenderAll->setToolTip(
        "When checked, \"Render All\" also writes region_diagnostics.csv "
        "(same data as the \"Export Region Diagnostics\" menu action) "
        "in the same pass over the frame data, instead of a second, "
        "separate pass over the same frames.");

    // Connect value range changes
    connect(qdsbValRangeFrom, QOverload<double>::of(&QDoubleSpinBox::valueChanged),
            this, &PPMainWindow::limits_changed);
    connect(qdsbValRangeTo, QOverload<double>::of(&QDoubleSpinBox::valueChanged),
            this, &PPMainWindow::limits_changed);
    connect(qdsbTransparency, QOverload<double>::of(&QDoubleSpinBox::valueChanged),
            this, &PPMainWindow::limits_changed);
    connect(qdsbHedgehogScale, QOverload<double>::of(&QDoubleSpinBox::valueChanged),
            this, &PPMainWindow::hedgehog_scale_changed);
    connect(qsbArrowheadSize, QOverload<int>::of(&QSpinBox::valueChanged),
            this, &PPMainWindow::arrowhead_size_changed);



    connect(comboBox_visualizations, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &PPMainWindow::comboboxIndexChanged_visualizations);

    qDebug() << "PPMainWindow constructor done";
}




void PPMainWindow::closeEvent(QCloseEvent* event)
{
    qDebug() << "close event";

    QSettings settings(settingsFileName,QSettings::IniFormat);
    qDebug() << "PPMainWindow: closing main window; " << settings.fileName();

    double data[10];
    renderer->GetActiveCamera()->GetPosition(&data[0]);
    renderer->GetActiveCamera()->GetFocalPoint(&data[3]);
    data[6] = renderer->GetActiveCamera()->GetParallelScale();

    qDebug() << "cam pos " << data[0] << "," << data[1] << "," << data[2];
    qDebug() << "cam focal pt " << data[3] << "," << data[4] << "," << data[5];
    qDebug() << "cam par scale " << data[6];

    QByteArray arr((char*)data, sizeof(data));
    settings.setValue("camData", arr);



    settings.setValue("vis_option", comboBox_visualizations->currentIndex());
    settings.setValue("hedgehog_option", comboBox_hedgehog->currentIndex());
    settings.setValue("arrowhead_size", qsbArrowheadSize->value());
    
    if (showTitleAction) {
        settings.setValue("show_title", showTitleAction->isChecked());
    }
    if (showDateTimeAction) {
        settings.setValue("show_date_time", showDateTimeAction->isChecked());
    }
    if (showColorBarAction) {
        settings.setValue("show_color_bar", showColorBarAction->isChecked());
    }

    // Save render selector state
    m_renderSelectorDialog->saveSelections(settingsFileName);

    if (!currentProjectDirectory.empty()) {
        QString frameFile = QString::fromStdString(currentProjectDirectory) + "/last_frame.txt";
        QFile file(frameFile);
        if (file.open(QIODevice::WriteOnly | QIODevice::Text)) {
            QTextStream out(&file);
            out << slider2->value();
            file.close();
        }
    }

    event->accept();
}


void PPMainWindow::limits_changed(double val_)
{
    qDebug() << "limits_changed";
    int idx = (int)representation.VisualizingVariable;
    VisOpt::range_from[idx] = qdsbValRangeFrom->value();
    VisOpt::range_to[idx] = qdsbValRangeTo->value();
    VisOpt::transparency_coeffs[idx] = qdsbTransparency->value();
    // Frame will be redrawn only after pressing 'submit' button
}

void PPMainWindow::comboboxIndexChanged_hedgehog(int index)
{
    representation.HedgehogVariable = (VisOpt::HedgehogType)index;
    qdsbHedgehogScale->blockSignals(true);
    qdsbHedgehogScale->setValue(VisOpt::hedgehog_ranges[index]);
    qdsbHedgehogScale->blockSignals(false);
    
    // Auto-enable flow interpolation if a flow visualization is selected
    if (index != VisOpt::hedgehog_none) 
    {
        if (!hsd.currentInterp.interpolation_enabled || !hsd.windInterp.interpolation_enabled) {
            hsd.currentInterp.SetEnabled(true);
            hsd.windInterp.SetEnabled(true);
            UpdateFlowInterpolatorsTime();
        }
    }
    
    representation.SynchronizeTopology();
    renderWindow->Render();
}

void PPMainWindow::hedgehog_scale_changed(double val_)
{
    qDebug() << "hedgehog_scale_changed";
    int idx = (int)representation.HedgehogVariable;
    VisOpt::hedgehog_ranges[idx] = qdsbHedgehogScale->value();
    // Frame will be redrawn only after pressing 'submit' button
}

void PPMainWindow::arrowhead_size_changed(int val)
{
    qDebug() << "arrowhead_size_changed";
    representation.SetArrowheadSize(val *10.0);
    renderWindow->Render();
}

void PPMainWindow::submitFrame_triggered()
{
    int targetFrame = qsbCurrentFrame->value();
    if (targetFrame != slider2->value()) {
        slider2->setValue(targetFrame); // This triggers sliderValueChanged, loading and rendering
    } else {
        // Redraw frame manually, e.g. if the limits or representations changed
        this->representation.SynchronizeTopology();
        renderWindow->Render();
    }
}

void PPMainWindow::forward24h_triggered()
{
    if (hsd.prms.AnimationFramePeriod <= 0) return;
    int frames_per_24h = std::round((24.0 * 3600.0) / hsd.prms.AnimationFramePeriod);
    if (frames_per_24h <= 0) frames_per_24h = 1;
    
    int next_frame = slider2->value() + frames_per_24h;
    if (next_frame > slider2->maximum()) {
        next_frame = slider2->maximum();
    }
    slider2->setValue(next_frame);
}

void PPMainWindow::keyPressEvent(QKeyEvent *event)
{
    // Left/Right steps the frame slider by ~1 simulated hour, same
    // frames-per-interval computation as forward24h_triggered (just 3600s
    // instead of 24*3600s), stepping in either direction and clamped to the
    // slider's range.
    if ((event->key() == Qt::Key_Left || event->key() == Qt::Key_Right) &&
        slider2->isEnabled() && hsd.prms.AnimationFramePeriod > 0) {
        int frames_per_hour = std::round(3600.0 / hsd.prms.AnimationFramePeriod);
        if (frames_per_hour <= 0) frames_per_hour = 1;

        int delta = (event->key() == Qt::Key_Right) ? frames_per_hour : -frames_per_hour;
        int next_frame = std::clamp(slider2->value() + delta, slider2->minimum(), slider2->maximum());
        slider2->setValue(next_frame);
        event->accept();
        return;
    }
    QMainWindow::keyPressEvent(event);
}

void PPMainWindow::jumpToTime_triggered()
{
    if (hsd.prms.AnimationFramePeriod <= 0) return;

    // Frames are nominally saved every AnimationFramePeriod seconds (frame N
    // is saved at simulation time N * AnimationFramePeriod); no interpolation,
    // just snap to the nearest saved frame.
    int nearestFrame = (int)std::round(qdsbJumpToTime->value() / hsd.prms.AnimationFramePeriod);
    nearestFrame = std::clamp(nearestFrame, slider2->minimum(), slider2->maximum());
    slider2->setValue(nearestFrame); // triggers sliderValueChanged() -> loads frame, updates status bar
}



void PPMainWindow::cameraReset_triggered()
{
    qDebug() << "MainWindow::on_action_camera_reset_triggered()";
    vtkCamera* camera = renderer->GetActiveCamera();
    renderer->ResetCamera();
    camera->ParallelProjectionOn();
    camera->SetClippingRange(1e-1,1e3);

    const double dx = hsd.prms.cellsize*hsd.prms.InitializationImageSizeX/2;
    const double dy = hsd.prms.cellsize*hsd.prms.InitializationImageSizeY/2;

    qDebug() << "dx " << dx << "\ndy " << dy;

    camera->SetPosition(dx, dy, 50.);
    camera->SetFocalPoint(dx, dy, 0.);
    camera->SetViewUp(0.0, 1.0, 0.0);
    camera->SetParallelScale(std::min(dx,dy)*1.1);

    camera->Modified();
    renderWindow->Render();
}

void PPMainWindow::determine_crop_extents_triggered()
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
    
    QString filename = "crop_extents.txt";
    if (!currentProjectDirectory.empty()) {
        filename = QString::fromStdString(currentProjectDirectory) + "/crop_extents.txt";
    }
    
    QFile file(filename);
    if (file.open(QIODevice::WriteOnly | QIODevice::Text)) {
        QTextStream out(&file);
        out << extentsMsg << "\n";
        file.close();
        statusBar->showMessage(QString("Saved extents to %1").arg(filename), 5000);
    } else {
        qWarning() << "Failed to save extents to" << filename;
        statusBar->showMessage("Failed to save extents", 5000);
    }
    
    VisOpt::Type visOpt = representation.VisualizingVariable;
    hsd.RenderAsJPG(slider2->value(), visOpt, offsetX, offsetY, crop_width, crop_height);
}

// ==================================================

void PPMainWindow::LoadParametersFile(QString fileName)
{
    namespace fs = std::filesystem;

    // Extract the directory containing the JSON file
    fs::path jsonPath(fileName.toStdString());
    fs::path jsonFileDir = jsonPath.parent_path();
    if (jsonFileDir.empty()) {
        jsonFileDir = ".";
    }

    LOGR("\n=== Loading project configuration ===");
    LOGR("JSON file: {}", jsonPath.string());
    LOGR("Project directory: {}", jsonFileDir.string());

    // Read the JSON configuration file to find where the grid data is stored
    LOGR("Parsing JSON configuration...");
    std::vector<std::string> carra1Files;
    std::map<std::string, std::string> parseResult = hsd.prms.ParseFile(jsonPath.string(), &carra1Files);
    LOGR("Project title: {}", parseResult["SimulationTitle"]);
    hsd.dirs.Configure(parseResult["RuntimeDirectory"], parseResult["ProjectName"]);

    // grid.h5/output are read from RuntimeDirectory/input/<ProjectName>/
    // when configured -- must match where preparer/gplate/cplate read and
    // write them (see DirectoryManager). Only CARRA1Data/GLO12Data/etc.
    // (the raw forcing data) stay resolved against jsonFileDir.
    fs::path projectDir = hsd.dirs.ProjectDirectory();

    // Remember this (runtime, not JSON-file) directory -- used for loading
    // frames and for last_frame.txt/crop_extents.txt/region_diagnostics.csv,
    // which must land in RuntimeDirectory too, not back in the git-tracked
    // input config tree.
    currentProjectDirectory = projectDir.string();

    // Load the grid structure that was created by plate_preparer
    // This contains grid dimensions, cell size, and other geometric information
    LOGR("Loading grid structure...");
    fs::path gridPath = projectDir / parseResult["GridData"];
    LOGR("Grid file: {}", gridPath.string());
    if (!fs::exists(gridPath)) {
        throw std::runtime_error(fmt::format("Grid file not found: {}", gridPath.string()));
    }
    hsd.LoadGridDataFromFile(gridPath.string());
    LOGR("Grid loaded successfully");

    // Subsampled modeled-area cell list for the wind/current rose overlays --
    // depends only on grid dimensions/landmask, so rebuilt once per project
    // load, not per frame (see wind_rose_widget.h).
    windRose.RebuildSampleCells(hsd);
    currentRose.RebuildSampleCells(hsd);

    // Load wind data if available (for wind visualization). CARRA1Data can be
    // multiple files, so ParseFile returns it via carra1Files (above), not
    // parseResult -- matches Model::LoadParameterFile's pattern.
    for (const std::string& rawPath : carra1Files) {
        fs::path windPath(rawPath);
        if (windPath.is_relative()) {
            windPath = jsonFileDir / windPath;
        }
        LOGR("Loading wind data: {}", windPath.string());
        hsd.windInterp.SetCARRA1Path(windPath.string());
        LOGR("Wind data loaded successfully");
    }

    // Load GLO12 data if available
    if (parseResult.count("GLO12Data") && !parseResult["GLO12Data"].empty()) {
        fs::path glo12Path = jsonFileDir / parseResult["GLO12Data"];
        LOGR("Loading GLO12 data: {}", glo12Path.string());
        if (fs::exists(glo12Path)) {
            hsd.currentInterp.SetGLO12Path(glo12Path.string());
            LOGR("GLO12 data loaded successfully");
        } else {
            LOGR("Warning: GLO12 data file not found: {}", glo12Path.string());
        }
    }

    if (parseResult.count("GLO12Tides") && !parseResult["GLO12Tides"].empty()) {
        fs::path glo12TidesPath = jsonFileDir / parseResult["GLO12Tides"];
        if (fs::exists(glo12TidesPath)) {
            hsd.currentInterp.SetGLO12TidesPath(glo12TidesPath.string());
            LOGR("GLO12 Tides data loaded from: {}", glo12TidesPath.string());
        } else {
            LOGR("Warning: GLO12Tides path not found: {}", glo12TidesPath.string());
        }
    }

    // sithick (ice thickness) lives in its own GLO12 file in the real Kane
    // Basin deployment (same lat/lon/time grid as currents, different .nc).
    if (parseResult.count("GLO12ThicknessData") && !parseResult["GLO12ThicknessData"].empty()) {
        fs::path glo12ThicknessPath = jsonFileDir / parseResult["GLO12ThicknessData"];
        if (fs::exists(glo12ThicknessPath)) {
            hsd.currentInterp.SetGLO12ThicknessPath(glo12ThicknessPath.string());
            LOGR("GLO12 Thickness data loaded from: {}", glo12ThicknessPath.string());
        } else {
            LOGR("Warning: GLO12ThicknessData path not found: {}", glo12ThicknessPath.string());
        }
    }

    // Note: We only visualize grid data, not particle data, so we skip loading snapshots

    // Set up the output directory where simulation frames are saved. Must
    // match how gplate/cplate resolve their own output directory (see
    // Model::LoadParameterFile).
    // This is where we'll find f00000.h5, f00001.h5, etc.
    LOGR("Checking for simulation output directory...");
    fs::path outputDir = hsd.dirs.OutputDirectory();
    fs::path framesDir = hsd.dirs.FramesDirectory();
    if (fs::exists(framesDir)) {
        LOGR("Found frames directory: {}", framesDir.string());
    }
    hsd.output_directory = outputDir.string();

    setWindowTitle(QString("MPM Post-Processor - %1").arg(fileName));

    LOGR("Project configuration loaded successfully\n");
}

void PPMainWindow::LoadFramesDirectory(QString framesDirectory)
{
    qDebug() << "PPMainWindow::LoadFramesDirectory: " << framesDirectory;
    currentFrameDirectory = framesDirectory.toStdString();

    int countFrames = frame_utils::ScanFrameDirectory(currentFrameDirectory);

    if(countFrames <= 0)
    {
        slider2->setEnabled(false);
        QMessageBox::warning(this, "No Frames Found", "The specified directory does not contain any valid frame files.");
        return;
    }

    // Configure slider
    slider2->setEnabled(true);
    slider2->setMaximum(countFrames);

    // Setup frame range controls
    qsbFrameTo->setMaximum(countFrames);
    qsbFrameTo->setValue(countFrames);

    qsbFrameFrom->setMaximum(countFrames);
    qsbFrameFrom->setValue(1);

    qsbCurrentFrame->setMaximum(countFrames);
    qsbCurrentFrame->setValue(countFrames);

    int initialFrame = countFrames;
    if (!currentProjectDirectory.empty()) {
        QString frameFile = QString::fromStdString(currentProjectDirectory) + "/last_frame.txt";
        QFile file(frameFile);
        if (file.open(QIODevice::ReadOnly | QIODevice::Text)) {
            QTextStream in(&file);
            int savedFrame = 0;
            in >> savedFrame;
            if (savedFrame >= 1 && savedFrame <= countFrames) {
                initialFrame = savedFrame;
            }
            file.close();
        }
    }

    // Load and display the initial frame as a starting point
    slider2->setValue(initialFrame);

    statusBar->showMessage(QString("Loaded %1 frames").arg(countFrames));
}

void PPMainWindow::sliderValueChanged(int val)
{
    qsbCurrentFrame->blockSignals(true);
    qsbCurrentFrame->setValue(val);
    qsbCurrentFrame->blockSignals(false);
    
    try {
        // Load the frame data from disk
        // Load the frame data from disk
        // std::string framePath = frame_utils::GetFramePath(currentFrameDirectory, val);
        hsd.LoadFrameData(val, currentFrameDirectory);

        representation.simulationTime = hsd.prms.SimulationTime;
        qDebug() << "time set to " << hsd.prms.SimulationTime;

        // Update WACI time to match current frame (for v_eta and v_norm visualization)
        UpdateFlowInterpolatorsTime();

        // Re-bin the wind/current roses for the new frame's time (no-op if hidden)
        windRose.Update(hsd);
        currentRose.Update(hsd);

        // Update the visualization with new frame data
        this->representation.SynchronizeTopology();


        // Update status bar with current frame number
        statusBar->showMessage(QString("Frame %1 | Time: %2").arg(val).arg(hsd.prms.SimulationTime, 0, 'f', 2));
        dateLabel->setText(FormatUtcDateTime(hsd.prms.SimulationStartTime + (long long)hsd.prms.SimulationTime));

        // Render the updated frame
        renderWindow->Render();
    } catch (const std::exception& e) {
        LOGR("Failed to load and display frame {}: {}", val, e.what());
    }
}

QString PPMainWindow::FormatUtcDateTime(long long epoch)
{
    std::time_t tt = (std::time_t)epoch;
    char buf[32];
    std::tm tm_utc;
    gmtime_r(&tt, &tm_utc);
    std::strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M UTC", &tm_utc);
    return QString::fromStdString(buf);
}


void PPMainWindow::comboboxIndexChanged_visualizations(int index)
{
    VisOpt::Type opt = (VisOpt::Type)index;

    // Flow visualizations render live from currentInterp/windInterp rather than a grid array -- keep them enabled and re-synced to the current frame's time whenever selected (SetEnabled is a no-op if already enabled).
    if (opt == VisOpt::glo12_ocean ||
        opt == VisOpt::carra1_wind ||
        opt == VisOpt::carra1_temperature ||
        opt == VisOpt::v_glo12_thickness)
    {
        hsd.currentInterp.SetEnabled(true);
        hsd.windInterp.SetEnabled(true);
        UpdateFlowInterpolatorsTime();
    }

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



void PPMainWindow::render_frame_triggered()
{
    qDebug() << "render_frame_triggered()";
    statusBar->showMessage("Rendering frame...");
    QCoreApplication::processEvents();

    std::string visName = VisOpt::descriptions.at(static_cast<VisOpt::Type>(representation.VisualizingVariable)).first;
    const int selectedFrame = slider2->value();
    std::string filename = visName + "_" + std::to_string(selectedFrame) + ".jpg";

    const QSizePolicy originalPolicy = qt_vtk_widget->sizePolicy();

    scrollArea->setWidgetResizable(false);
    
    qt_vtk_widget->setFixedSize(1920, 1080);
    
    QCoreApplication::processEvents();

    windowToImageFilter->SetInput(renderWindow);
    writer->SetInputConnection(windowToImageFilter->GetOutputPort());

    renderWindow->DoubleBufferOff();
    renderWindow->Render();
    windowToImageFilter->SetInputBufferTypeToRGB();
    renderWindow->WaitForCompletion();

    windowToImageFilter->Modified();
    writer->SetFileName(filename.c_str());
    writer->Write();

    renderWindow->DoubleBufferOn();

    qt_vtk_widget->setMinimumSize(0, 0);
    qt_vtk_widget->setMaximumSize(QWIDGETSIZE_MAX, QWIDGETSIZE_MAX);
    qt_vtk_widget->setSizePolicy(originalPolicy);
    scrollArea->setWidgetResizable(true);
    QCoreApplication::processEvents();

    statusBar->showMessage(QString("Rendered frame %1 as %2").arg(selectedFrame).arg(QString::fromStdString(visName)), 3000);
}



void PPMainWindow::captureCameraToSlot(int index)
{
    if (index < 0 || index >= 5) return;
    vtkCamera* cam = renderer->GetActiveCamera();
    cam->GetPosition(&m_cameraSlots[index][0]);
    cam->GetFocalPoint(&m_cameraSlots[index][3]);
    m_cameraSlots[index][6] = cam->GetParallelScale();
    statusBar->showMessage(QString("Camera position saved to slot %1").arg(index + 1), 2000);
}

void PPMainWindow::render_all_triggered()
{
    qDebug() << "render_all_triggered() started.";

    // Get the frame range from the UI controls
    const int frameFrom = qsbFrameFrom->value();
    const int frameTo = qsbFrameTo->value();

    // Get selected visualization options from dialog
    std::vector<VisOpt::Type> visOptsToRender =
        m_renderSelectorDialog->getSelectedOptions();

    // Check if any flow visualization is selected
    bool needFlow = false;
    for (const auto& opt : visOptsToRender) {
        if (opt == VisOpt::Type::glo12_ocean ||
            opt == VisOpt::Type::carra1_wind ||
            opt == VisOpt::Type::carra1_temperature ||
            opt == VisOpt::Type::v_glo12_thickness)
        {
            needFlow = true;
            break;
        }
    }
    
    // The wind/current roses are real VTK actors captured along with
    // everything else in this function's on-screen render (unlike
    // render_all_alt_triggered's separate CPU rasterizer, which can never
    // show them) -- but they need the interpolators enabled too, and
    // "needFlow" above only looks at the selected raster VisOpts, with no
    // idea the roses exist.
    if (windRose.GetVisible() || currentRose.GetVisible())
        needFlow = true;

    // Optional in-line region_diagnostics.csv export (see
    // actionExportRegionDiagnosticsOnRenderAll) -- also needs the flow
    // interpolators, and its own wind_speed_mean/current_speed_mean
    // columns depend on them same as the roses/raster VisOpts above.
    // Validated and set up here (once) rather than inside the frame loop;
    // a missing/empty mask only disables the CSV export for this run --
    // it does not abort the render, since the checkbox is an opt-in extra
    // on top of the render the user actually asked for.
    bool exportDiagnostics = actionExportRegionDiagnosticsOnRenderAll->isChecked();
    std::vector<std::pair<int,int>> diagnosticsMaskedCells;
    std::ofstream diagnosticsCsv;
    std::filesystem::path diagnosticsCsvPath;
    int diagnosticsFramesWritten = 0;
    if (exportDiagnostics) {
        if (!HasAnalysisRegionMaskCells()) {
            QMessageBox::warning(this, "No Analysis Region Mask",
                "\"Also export region diagnostics CSV\" is checked, but no AnalysisRegionMask cells were "
                "loaded from grid.h5 (dataset missing, or it selects zero cells).\n\n"
                "Rendering will continue without the CSV export.");
            exportDiagnostics = false;
        } else {
            diagnosticsMaskedCells = CollectMaskedCells();
            diagnosticsCsvPath = std::filesystem::path(currentProjectDirectory) / "region_diagnostics.csv";
            diagnosticsCsv.open(diagnosticsCsvPath);
            if (!diagnosticsCsv) {
                QMessageBox::warning(this, "Export Failed",
                    QString("Could not open %1 for writing.\n\nRendering will continue without the CSV export.")
                        .arg(QString::fromStdString(diagnosticsCsvPath.string())));
                exportDiagnostics = false;
            } else {
                diagnosticsCsv << "frame,epoch_utc,datetime_utc,ice_strength_mean,ice_strength_max,"
                                  "wind_speed_mean,wind_speed_max,current_speed_mean,current_speed_max,"
                                  "ice_speed_mean,ice_speed_max,fractured_fraction,"
                                  "wind_drag_mean,wind_drag_max,current_drag_mean,current_drag_max\n";
                needFlow = true;
            }
        }
    }

    if (needFlow && (!hsd.currentInterp.interpolation_enabled || !hsd.windInterp.interpolation_enabled)) {
        hsd.currentInterp.SetEnabled(true);
        hsd.windInterp.SetEnabled(true);
    }

    // Validate that at least one option is selected
    if (visOptsToRender.empty()) {
        QMessageBox::warning(this, "No Visualizations Selected",
            "Please select at least one visualization option in the Render Selector (View → Render Selector).");
        return;
    }

    // Identify active camera slots (including default)
    struct RenderTarget {
        std::string dirName;
        int slotIndex; // -1 for default
    };
    std::vector<RenderTarget> targets;
    targets.push_back({"raster", -1}); // Default
    for(int i = 0; i < 5; ++i) {
        if(m_useCameraSlot[i]) {
            targets.push_back({"raster_" + std::to_string(i+1), i});
        }
    }

    // Validate that the frame range is valid
    if (frameTo < frameFrom) {
        QMessageBox::warning(this, "Invalid Range", "Frame 'To' must be greater than or equal to Frame 'From'.");
        return;
    }
    const int totalFrames = (frameTo - frameFrom) + 1;

    statusBar->showMessage(QString("Rendering %1 frames...").arg(totalFrames));

    // Create a progress dialog to show rendering status
    const int totalOperations = totalFrames * visOptsToRender.size() * targets.size();
    QProgressDialog progress("Rendering all frames...", "Abort", 0, totalOperations, this);
    progress.setWindowModality(Qt::WindowModal);
    QCoreApplication::processEvents();

    // Set up the rendering window to a fixed size for consistent output
    const QSizePolicy originalPolicy = qt_vtk_widget->sizePolicy();
    scrollArea->setWidgetResizable(false);
    qt_vtk_widget->setFixedSize(1920, 1080);
    QCoreApplication::processEvents();

    // Configure VTK rendering and image capture
    renderWindow->DoubleBufferOff();
    windowToImageFilter->SetInput(renderWindow);
    windowToImageFilter->SetInputBufferTypeToRGB();
    writer->SetInputConnection(windowToImageFilter->GetOutputPort());

    // Save initial camera state to restore later
    vtkCamera* cam = renderer->GetActiveCamera();
    std::array<double, 7> initialCamState;
    cam->GetPosition(&initialCamState[0]);
    cam->GetFocalPoint(&initialCamState[3]);
    initialCamState[6] = cam->GetParallelScale();

    // Main loop: process each frame
    int operationCount = 0;
    for (int frameNum = frameFrom; frameNum <= frameTo; ++frameNum) {
        if (progress.wasCanceled()) break;

        progress.setLabelText(QString("Loading frame %1...").arg(frameNum));
        QCoreApplication::processEvents();

        // Load the frame data from HDF5 file (ONCE per frame)
        try {
            hsd.LoadFrameData(frameNum, currentFrameDirectory);
            representation.simulationTime = hsd.prms.SimulationTime;
            // Update WACI time to match current frame. (Previously
            // SimulationTime*2+SimulationStartTime here -- a copy-paste
            // divergence from every other call site in this file, which all
            // use UpdateFlowInterpolatorsTime()'s SimulationTime+SimulationStartTime.
            // Fixed to match: for any frame past the first, the doubled
            // offset was sampling CARRA1/GLO12 well past the frame's actual
            // time, growing worse deeper into the run.)
            UpdateFlowInterpolatorsTime();
            // Re-bin the wind/current roses for this frame -- no-op if
            // hidden, but without this they'd stay frozen at whatever they
            // last showed interactively and get baked identically into
            // every exported frame.
            windRose.Update(hsd);
            currentRose.Update(hsd);

            if (exportDiagnostics) {
                WriteRegionDiagnosticsRow(diagnosticsCsv, frameNum, diagnosticsMaskedCells);
                ++diagnosticsFramesWritten;
            }
        } catch (const std::exception& e) {
            LOGR("Failed to load frame {}: {}", frameNum, e.what());
            operationCount += visOptsToRender.size() * targets.size();
            progress.setValue(operationCount);
            continue;
        }

        // Render each visualization option for this frame
        for (const auto& visOpt : visOptsToRender) {
            if (progress.wasCanceled()) break;

            const QString visName = QString::fromStdString(VisOpt::descriptions.at(visOpt).first);

            // Update visualization (ONCE per VisOpt per frame)
            representation.ChangeVisualizationOption(visOpt);

            // Render for each camera position
            for (const auto& target : targets) {
                 if (progress.wasCanceled()) break;
                 operationCount++;

                 progress.setValue(operationCount);
                 progress.setLabelText(QString("Frame %1/%2 : %3 [%4]")
                                           .arg(frameNum - frameFrom + 1)
                                           .arg(totalFrames)
                                           .arg(visName)
                                           .arg(QString::fromStdString(target.dirName)));
                 
                 // Apply camera state
                 if (target.slotIndex == -1) {
                     // Default: use initial state (or should we track the "live" camera? 
                     // The requirement says "The current version works well... current camera state is used".
                     // So we should restore the state the user had before clicking Render.
                     cam->SetPosition(initialCamState[0], initialCamState[1], initialCamState[2]);
                     cam->SetFocalPoint(initialCamState[3], initialCamState[4], initialCamState[5]);
                     cam->SetParallelScale(initialCamState[6]);
                 } else {
                     // Apply slot state
                     const auto& slot = m_cameraSlots[target.slotIndex];
                     cam->SetPosition(slot[0], slot[1], slot[2]);
                     cam->SetFocalPoint(slot[3], slot[4], slot[5]);
                     cam->SetParallelScale(slot[6]);
                 }
                 cam->Modified();
                 
                 // Render
                 renderWindow->Render();
                 windowToImageFilter->Modified();

                 // Create output directory structure and save rendered image
                 QDir frameDir(QString::fromStdString(currentFrameDirectory));
                 frameDir.cdUp(); // Navigate to parent directory (output/)
                 const QString rasterBasePath = frameDir.filePath(QString::fromStdString(target.dirName));
                 const QString subDir = QString("%1/%2").arg(rasterBasePath).arg(visName);
                 QDir().mkpath(subDir); // Create full directory path
                 const QString outputPath = QString("%1/%2.jpg").arg(subDir).arg(frameNum, 5, 10, QChar('0'));

                 // Capture and write the image to disk
                 writer->SetFileName(outputPath.toStdString().c_str());
                 writer->Write();
            }
        }
    }

    // Restore the GUI to its original state
    cam->SetPosition(initialCamState[0], initialCamState[1], initialCamState[2]);
    cam->SetFocalPoint(initialCamState[3], initialCamState[4], initialCamState[5]);
    cam->SetParallelScale(initialCamState[6]);
    cam->Modified();

    qt_vtk_widget->setMinimumSize(0, 0);
    qt_vtk_widget->setMaximumSize(QWIDGETSIZE_MAX, QWIDGETSIZE_MAX);
    qt_vtk_widget->setSizePolicy(originalPolicy);
    scrollArea->setWidgetResizable(true);
    renderWindow->DoubleBufferOn();
    QCoreApplication::processEvents();

    // Generate scripts for all populated targets
    for(const auto& target : targets) {
        generate_ffmpeg_script(frameFrom, frameTo, target.dirName);
    }

    // Refresh the display to show the last rendered frame
    renderWindow->Render();

    QString completionMessage = "Batch rendering has finished or was canceled.";
    if (exportDiagnostics) {
        diagnosticsCsv.close();
        LOGR("render_all_triggered: wrote {} region diagnostics row(s) to {}",
             diagnosticsFramesWritten, diagnosticsCsvPath.string());
        completionMessage += QString("\n\nAlso wrote %1 region diagnostics row(s) to %2")
            .arg(diagnosticsFramesWritten)
            .arg(QString::fromStdString(diagnosticsCsvPath.string()));
    }

    // Show completion message
    progress.setValue(totalOperations);
    QMessageBox::information(this, "Rendering Complete", completionMessage);
}

void PPMainWindow::render_all_alt_triggered()
{
    qDebug() << "render_all_alt_triggered() started.";

    // Get the frame range from the UI controls
    const int frameFrom = qsbFrameFrom->value();
    const int frameTo = qsbFrameTo->value();

    // Get selected visualization options from dialog
    std::vector<VisOpt::Type> visOptsToRender =
        m_renderSelectorDialog->getSelectedOptions();

    // Check if any flow visualization is selected
    bool needFlow = false;
    for (const auto& opt : visOptsToRender) {
        if (opt == VisOpt::Type::glo12_ocean ||
            opt == VisOpt::Type::carra1_wind ||
            opt == VisOpt::Type::carra1_temperature ||
            opt == VisOpt::Type::v_glo12_thickness)
        {
            needFlow = true;
            break;
        }
    }
    
    if (needFlow && (!hsd.currentInterp.interpolation_enabled || !hsd.windInterp.interpolation_enabled)) {
        hsd.currentInterp.SetEnabled(true);
        hsd.windInterp.SetEnabled(true);
    }

    // Validate that at least one option is selected
    if (visOptsToRender.empty()) {
        QMessageBox::warning(this, "No Visualizations Selected",
            "Please select at least one visualization option in the Render Selector (View → Render Selector).");
        return;
    }

    // Identify active camera slots (including default)
    struct RenderTarget {
        std::string dirName;
        int slotIndex; // -1 for default
    };
    std::vector<RenderTarget> targets;
    targets.push_back({"raster", -1}); // Default
    for(int i = 0; i < 5; ++i) {
        if(m_useCameraSlot[i]) {
            targets.push_back({"raster_" + std::to_string(i+1), i});
        }
    }

    // Validate that the frame range is valid
    if (frameTo < frameFrom) {
        QMessageBox::warning(this, "Invalid Range", "Frame 'To' must be greater than or equal to Frame 'From'.");
        return;
    }
    const int totalFrames = (frameTo - frameFrom) + 1;

    statusBar->showMessage(QString("Rendering %1 frames via Alt Engine...").arg(totalFrames));

    // Create a progress dialog to show rendering status
    const int totalOperations = totalFrames * visOptsToRender.size() * targets.size();
    QProgressDialog progress("Rendering all frames via Raster API...", "Abort", 0, totalOperations, this);
    progress.setWindowModality(Qt::WindowModal);
    QCoreApplication::processEvents();

    // Save initial camera state to restore later
    vtkCamera* cam = renderer->GetActiveCamera();
    std::array<double, 7> initialCamState;
    cam->GetPosition(&initialCamState[0]);
    cam->GetFocalPoint(&initialCamState[3]);
    initialCamState[6] = cam->GetParallelScale();

    int operationCount = 0;
    for (int frameNum = frameFrom; frameNum <= frameTo; ++frameNum) {
        if (progress.wasCanceled()) break;

        progress.setLabelText(QString("Loading frame %1...").arg(frameNum));
        QCoreApplication::processEvents();

        // Load the frame data from HDF5 file (ONCE per frame)
        try {
            hsd.LoadFrameData(frameNum, currentFrameDirectory);
            representation.simulationTime = hsd.prms.SimulationTime;
            // Update WACI time to match current frame -- see the identical
            // fix/comment in render_all_triggered.
            UpdateFlowInterpolatorsTime();
        } catch (const std::exception& e) {
            LOGR("Failed to load frame {}: {}", frameNum, e.what());
            operationCount += visOptsToRender.size() * targets.size();
            progress.setValue(operationCount);
            continue;
        }

        // Render each visualization option for this frame
        for (const auto& visOpt : visOptsToRender) {
            if (progress.wasCanceled()) break;

            const QString visName = QString::fromStdString(VisOpt::descriptions.at(visOpt).first);

            for (const auto& target : targets) {
                 if (progress.wasCanceled()) break;
                 operationCount++;

                 progress.setValue(operationCount);
                 progress.setLabelText(QString("Frame %1/%2 : %3 [%4]")
                                           .arg(frameNum - frameFrom + 1)
                                           .arg(totalFrames)
                                           .arg(visName)
                                           .arg(QString::fromStdString(target.dirName)));
                 
                 // Apply camera state
                 if (target.slotIndex == -1) {
                     cam->SetPosition(initialCamState[0], initialCamState[1], initialCamState[2]);
                     cam->SetFocalPoint(initialCamState[3], initialCamState[4], initialCamState[5]);
                     cam->SetParallelScale(initialCamState[6]);
                 } else {
                     const auto& slot = m_cameraSlots[target.slotIndex];
                     cam->SetPosition(slot[0], slot[1], slot[2]);
                     cam->SetFocalPoint(slot[3], slot[4], slot[5]);
                     cam->SetParallelScale(slot[6]);
                 }
                 cam->Modified();
                 
                 // Compute crop bounds natively from the applied camera
                 double focalPoint[3];
                 cam->GetFocalPoint(focalPoint);
                 
                 double parallelScale = cam->GetParallelScale();
                 double cellsize = hsd.prms.cellsize;
                 
                 double visibleHeightGrid = (2.0 * parallelScale) / cellsize;
                 
                 // Default bounding constraint mapping dynamically off full res
                 int winSize0 = 1920; 
                 int winSize1 = 1080;
                 // Since rendering alt is offscreen, we don't grab current renderWindow aspect,
                 // we force the 16:9 standard expected aspect ratio layout.
                 double winAspect = (double)winSize0 / (double)winSize1;
                 double visibleWidthGrid = visibleHeightGrid * winAspect;
                 
                 int k_h = std::round(visibleHeightGrid / 9.0);
                 int k_w = std::round(visibleWidthGrid / 16.0);
                 int k = std::max(k_h, k_w);
                 if (k < 1) k = 1;
                 
                 int crop_width = 16 * k;
                 int crop_height = 9 * k;
                 
                 int cx = std::round(focalPoint[0] / cellsize);
                 int cy = std::round(focalPoint[1] / cellsize);
                 
                 int offsetX = cx - crop_width / 2;
                 int offsetY = cy - crop_height / 2;

                 hsd.RenderAsJPG(frameNum, visOpt, offsetX, offsetY, crop_width, crop_height, target.dirName);
            }
        }
    }

    // Restore GUI state
    cam->SetPosition(initialCamState[0], initialCamState[1], initialCamState[2]);
    cam->SetFocalPoint(initialCamState[3], initialCamState[4], initialCamState[5]);
    cam->SetParallelScale(initialCamState[6]);
    cam->Modified();

    representation.ChangeVisualizationOption(representation.VisualizingVariable);
    renderWindow->Render();

    for(const auto& target : targets) {
        generate_ffmpeg_script(frameFrom, frameTo, target.dirName);
    }

    progress.setValue(totalOperations);
    QMessageBox::information(this, "Rendering Complete", "Batch raster (Alt API) rendering has finished or was canceled.");
}


bool PPMainWindow::HasAnalysisRegionMaskCells() const
{
    return std::any_of(hsd.analysis_mask_buffer.begin(), hsd.analysis_mask_buffer.end(),
                        [](uint8_t v) { return v != 0; });
}

std::vector<std::pair<int,int>> PPMainWindow::CollectMaskedCells() const
{
    const int gx = hsd.prms.GridXTotal;
    const int gy = hsd.prms.GridYTotal;
    std::vector<std::pair<int,int>> maskedCells;
    for (int i = 0; i < gx; ++i)
        for (int j = 0; j < gy; ++j)
            if (hsd.analysis_mask_buffer[(size_t)j + (size_t)i * gy])
                maskedCells.push_back({i, j});
    return maskedCells;
}

// See the header comment: hsd must already have frameNum loaded, and the
// flow interpolators already set to that frame's time, before this runs.
void PPMainWindow::WriteRegionDiagnosticsRow(std::ofstream& csv, int frameNum,
                                              const std::vector<std::pair<int,int>>& maskedCells)
{
    const int gy = hsd.prms.GridYTotal;

    const float* ptr_ice_strength = hsd.GetGridBufferPointer(SimParams::HostGridArrayIndex::grid_idx_vis_ice_strength);
    const float* ptr_px = hsd.GetGridBufferPointer(SimParams::HostGridArrayIndex::grid_idx_px);
    const float* ptr_py = hsd.GetGridBufferPointer(SimParams::HostGridArrayIndex::grid_idx_py);
    const float* ptr_mass = hsd.GetGridBufferPointer(SimParams::HostGridArrayIndex::host_grid_idx_mass);
    // Fraction of cracked material at each node (uint8 [0,255] -> float [0,1] on load,
    // see HostSideData::LoadFrameData) -- mass-weighted, not a hard per-cell classification.
    const float* ptr_cracked = hsd.GetGridBufferPointer(SimParams::HostGridArrayIndex::grid_idx_vis_cracked);

    double strengthSum = 0.0, strengthMax = 0.0;
    double windSum = 0.0, windMax = 0.0;
    double currentSum = 0.0, currentMax = 0.0;
    double iceSpeedSum = 0.0, iceSpeedMax = 0.0;
    double crackedSum = 0.0;

    // Mass-weighted wind/current drag stress (traction, Pa), same quadratic drag law
    // as kernels.cu's node update -- zero-mass cells contribute nothing to the mean.
    double massSum = 0.0;
    double windDragSumX = 0.0, windDragSumY = 0.0, windDragMax = 0.0;
    double currentDragSumX = 0.0, currentDragSumY = 0.0, currentDragMax = 0.0;

    for (const auto& [i, j] : maskedCells) {
        const size_t idx = (size_t)j + (size_t)i * gy;

        if (ptr_ice_strength) {
            const double val = ptr_ice_strength[idx];
            strengthSum += val;
            strengthMax = std::max(strengthMax, val);
        }

        const auto [wx, wy] = hsd.windInterp.GetWindValue(i, j);
        const double wspeed = std::sqrt(wx*wx + wy*wy);
        windSum += wspeed;
        windMax = std::max(windMax, wspeed);

        const auto [cx, cy] = hsd.currentInterp.GetOceanValue(i, j);
        const double cspeed = std::sqrt(cx*cx + cy*cy);
        currentSum += cspeed;
        currentMax = std::max(currentMax, cspeed);

        if (ptr_px && ptr_py) {
            const double ix = ptr_px[idx], iy = ptr_py[idx];
            const double ispeed = std::sqrt(ix*ix + iy*iy);
            iceSpeedSum += ispeed;
            iceSpeedMax = std::max(iceSpeedMax, ispeed);

            if (ptr_mass && ptr_mass[idx] > 0) {
                const double mass = ptr_mass[idx];

                const double urelWindX = wx - ix, urelWindY = wy - iy;
                const double urelWindMag = std::sqrt(urelWindX*urelWindX + urelWindY*urelWindY);
                const double tauWindX = hsd.prms.windDragNormalized * SimParams::rho_air * urelWindMag * urelWindX;
                const double tauWindY = hsd.prms.windDragNormalized * SimParams::rho_air * urelWindMag * urelWindY;

                const double urelCurX = cx - ix, urelCurY = cy - iy;
                const double urelCurMag = std::sqrt(urelCurX*urelCurX + urelCurY*urelCurY);
                const double tauCurX = hsd.prms.waterDragNormalized * SimParams::rho_water * urelCurMag * urelCurX;
                const double tauCurY = hsd.prms.waterDragNormalized * SimParams::rho_water * urelCurMag * urelCurY;

                massSum += mass;
                windDragSumX += mass*tauWindX;
                windDragSumY += mass*tauWindY;
                windDragMax = std::max(windDragMax, std::sqrt(tauWindX*tauWindX + tauWindY*tauWindY));

                currentDragSumX += mass*tauCurX;
                currentDragSumY += mass*tauCurY;
                currentDragMax = std::max(currentDragMax, std::sqrt(tauCurX*tauCurX + tauCurY*tauCurY));
            }
        }

        if (ptr_cracked) {
            crackedSum += ptr_cracked[idx];
        }
    }

    const double cellCount = (double)maskedCells.size();
    const long long epoch = hsd.prms.SimulationStartTime + (long long)hsd.prms.SimulationTime;

    // Magnitude of the mass-weighted mean vector (not mean of magnitudes) -- lets
    // opposing-direction contributions within the region partially cancel, matching
    // how a net regional drag force is normally reported.
    const double windDragMean = massSum > 0 ? std::sqrt(windDragSumX*windDragSumX + windDragSumY*windDragSumY) / massSum : 0.0;
    const double currentDragMean = massSum > 0 ? std::sqrt(currentDragSumX*currentDragSumX + currentDragSumY*currentDragSumY) / massSum : 0.0;

    csv << frameNum << "," << epoch << "," << FormatUtcDateTime(epoch).toStdString() << ","
        << (strengthSum / cellCount) << "," << strengthMax << ","
        << (windSum / cellCount) << "," << windMax << ","
        << (currentSum / cellCount) << "," << currentMax << ","
        << (iceSpeedSum / cellCount) << "," << iceSpeedMax << ","
        << (crackedSum / cellCount) << ","
        << windDragMean << "," << windDragMax << ","
        << currentDragMean << "," << currentDragMax << "\n";
}

void PPMainWindow::exportRegionDiagnostics_triggered()
{
    namespace fs = std::filesystem;

    if (!HasAnalysisRegionMaskCells()) {
        QMessageBox::warning(this, "No Analysis Region Mask",
            "No AnalysisRegionMask cells were loaded from grid.h5 (dataset missing, or it selects zero cells).\n\n"
            "Set \"AnalysisRegionMask\" in the config and re-run preparer (GeneratePoints=false is enough to "
            "refresh grid.h5 without regenerating points) before exporting.");
        return;
    }

    const int frameFrom = qsbFrameFrom->value();
    const int frameTo = qsbFrameTo->value();
    if (frameTo < frameFrom) {
        QMessageBox::warning(this, "Invalid Range", "Frame 'To' must be greater than or equal to Frame 'From'.");
        return;
    }

    if (!hsd.currentInterp.interpolation_enabled || !hsd.windInterp.interpolation_enabled) {
        hsd.currentInterp.SetEnabled(true);
        hsd.windInterp.SetEnabled(true);
    }

    // Masked (i,j) grid cells, collected once -- reused every frame.
    const std::vector<std::pair<int,int>> maskedCells = CollectMaskedCells();

    const fs::path csvPath = fs::path(currentProjectDirectory) / "region_diagnostics.csv";
    std::ofstream csv(csvPath);
    if (!csv) {
        QMessageBox::critical(this, "Export Failed",
            QString("Could not open %1 for writing.").arg(QString::fromStdString(csvPath.string())));
        return;
    }
    csv << "frame,epoch_utc,datetime_utc,ice_strength_mean,ice_strength_max,"
           "wind_speed_mean,wind_speed_max,current_speed_mean,current_speed_max,"
           "ice_speed_mean,ice_speed_max,fractured_fraction,"
           "wind_drag_mean,wind_drag_max,current_drag_mean,current_drag_max\n";

    QProgressDialog progress("Exporting region diagnostics...", "Abort", frameFrom, frameTo, this);
    progress.setWindowModality(Qt::WindowModal);

    int framesWritten = 0;
    for (int frameNum = frameFrom; frameNum <= frameTo; ++frameNum) {
        if (progress.wasCanceled()) break;
        progress.setValue(frameNum);
        QCoreApplication::processEvents();

        try {
            hsd.LoadFrameData(frameNum, currentFrameDirectory);
        } catch (const std::exception& e) {
            LOGR("exportRegionDiagnostics: failed to load frame {}: {}", frameNum, e.what());
            continue;
        }
        UpdateFlowInterpolatorsTime();

        WriteRegionDiagnosticsRow(csv, frameNum, maskedCells);
        ++framesWritten;
    }
    csv.close();
    progress.setValue(frameTo);

    LOGR("exportRegionDiagnostics: wrote {} rows to {}", framesWritten, csvPath.string());
    statusBar->showMessage(QString("Region diagnostics exported to %1").arg(QString::fromStdString(csvPath.string())), 5000);
    QMessageBox::information(this, "Export Complete",
        QString("Wrote %1 frame(s) to %2").arg(framesWritten).arg(QString::fromStdString(csvPath.string())));
}


void PPMainWindow::render_full_grid_snapshot_triggered()
{
    qDebug() << "render_full_grid_snapshot_triggered()";
    statusBar->showMessage("Rendering full resolution snapshot...");
    QCoreApplication::processEvents();

    VisOpt::Type visOpt = representation.VisualizingVariable;
    int currentFrame = slider2->value();

    // Pass -1 for sizeX, sizeY to get full InitializationImageSize
    // Pass -1 for finalW, finalH to keep it at that resolution (no resizing)
    hsd.RenderAsJPG(currentFrame, visOpt, 0, 0, -1, -1, "snapshots_full", -1, -1);

    statusBar->showMessage("Full resolution snapshot saved.", 3000);
}


void PPMainWindow::export_scalar_bar_svg_triggered()
{
    namespace fs = std::filesystem;
    qDebug() << "export_scalar_bar_svg_triggered()";

    VisOpt::Type visOpt = representation.VisualizingVariable;
    int currentFrame = slider2->value();

    std::string out_dir = hsd.output_directory.empty() ? "output" : hsd.output_directory;
    std::string visName = VisOpt::descriptions.at(visOpt).first;
    fs::path barDir = fs::path(out_dir) / "snapshots_full" / visName;
    fs::create_directories(barDir);
    std::string svgPath = (barDir / fmt::format("{:05d}_scalarbar.svg", currentFrame)).string();

    representation.ExportScalarBarSVG(svgPath);

    statusBar->showMessage(QString("Scalar bar saved to %1").arg(QString::fromStdString(svgPath)), 5000);
}




// --- Core Logic Function (Rewritten) ---
// This is the clean, reusable implementation.
void PPMainWindow::generate_ffmpeg_script(int frameFrom, int frameTo, std::string dirName)
{
    hsd.generate_ffmpeg_script(frameFrom, frameTo, dirName, currentFrameDirectory, m_renderSelectorDialog->getSelectedOptions());
}


void PPMainWindow::toggleRenderSelector()
{
    if (m_renderSelectorDialog->isVisible()) {
        m_renderSelectorDialog->hide();
    } else {
        m_renderSelectorDialog->show();
        m_renderSelectorDialog->raise();
        m_renderSelectorDialog->activateWindow();
    }
}


void PPMainWindow::TryLoadDefaultFrames()
{
    if(currentProjectDirectory.empty())
    {
        LOGR("No project directory set, cannot auto-load frames");
        return;
    }

    // Try to load frames from default location: [project_dir]/output/frames
    std::string defaultFramesPath = currentProjectDirectory + "/output/frames";

    if(!std::filesystem::exists(defaultFramesPath))
    {
        LOGR("Default frames directory does not exist: {}", defaultFramesPath);
        return;
    }

    LOGR("Found default frames directory, auto-loading from: {}", defaultFramesPath);
    LoadFramesDirectory(QString::fromStdString(defaultFramesPath));
}


void PPMainWindow::openProject_triggered()
{
    qDebug() << "openProject_triggered()";

    QString fileName = QFileDialog::getOpenFileName(this, "Open Project Configuration", "",
                                                    "JSON Files (*.json);;All Files (*)");
    if(!fileName.isEmpty())
    {
        LoadParametersFile(fileName);
        // After loading project, try to auto-load frames from default location
        TryLoadDefaultFrames();
    }
}


void PPMainWindow::openFrames_triggered()
{
    qDebug() << "openFrames_triggered()";

    if(currentProjectDirectory.empty())
    {
        QMessageBox::warning(this, "No Project Loaded",
                            "Please open a project first before selecting frames directory.");
        return;
    }

    QString framesDir = QFileDialog::getExistingDirectory(this, "Select Frames Directory",
                                                         QString::fromStdString(currentProjectDirectory));
    if(!framesDir.isEmpty())
    {
        LoadFramesDirectory(framesDir);
    }
}

void PPMainWindow::toggleContours(bool checked)
{
    qDebug() << "toggleContours: " << checked;
    // representation.ShowEtaContours = checked;
    representation.SynchronizeTopology(); // Will update visibility and data
    renderWindow->Render();
}

void PPMainWindow::contourIntervalChanged(double val)
{
    // representation.ContourInterval = val;
    representation.SynchronizeTopology(); // Trigger update if contours are shown
    renderWindow->Render();
}

void PPMainWindow::windRoseToggled(bool checked)
{
    // Single toggle controls both the wind and current roses.
    windRose.SetVisible(checked);
    currentRose.SetVisible(checked);
    if (checked) {
        // Flow visualizations render live from windInterp/currentInterp
        // rather than a grid array -- force both enabled/synced regardless
        // of which raster visualization option is currently selected
        // (SetEnabled is a no-op if already enabled).
        hsd.windInterp.SetEnabled(true);
        hsd.currentInterp.SetEnabled(true);
        UpdateFlowInterpolatorsTime();
        windRose.Update(hsd);
        currentRose.Update(hsd);
    }
    renderWindow->Render();
}
