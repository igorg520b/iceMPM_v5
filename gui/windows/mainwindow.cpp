#include "mainwindow.h"
#include "./ui_mainwindow.h"
#include <QCloseEvent>
#include <QFileDialog>
#include <QList>
#include <QPointF>
#include <QStringList>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <ctime>
#include <fstream>

MainWindow::~MainWindow() { delete ui; }

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent), ui(new Ui::MainWindow),
      representation(model.sim_data) {
  ui->setupUi(this);

  worker = new BackgroundWorker(&model);

  // VTK
  qt_vtk_widget = new QVTKOpenGLNativeWidget();
  qt_vtk_widget->setRenderWindow(renderWindow);

  renderer->SetBackground(1.0, 1.0, 1.0);
  renderWindow->AddRenderer(renderer);
  renderWindow->GetInteractor()->SetInteractorStyle(interactorStyle);

  setCentralWidget(qt_vtk_widget);

  // toolbar - combobox
  comboBox_visualizations = new QComboBox();
  ui->toolBar->addWidget(comboBox_visualizations);

  ui->toolBar->addSeparator();

  comboBox_hedgehog = new QComboBox();
  ui->toolBar->addWidget(comboBox_hedgehog);

  QLabel *lbl1 = new QLabel("range:");
  ui->toolBar->addWidget(lbl1);

  // [from, to] color-mapping bounds -- see VisOpt::range_from/range_to
  qdsbValRangeFrom = new QDoubleSpinBox();
  qdsbValRangeFrom->setRange(-1e7, 1e7);
  qdsbValRangeFrom->setValue(0);
  qdsbValRangeFrom->setDecimals(4);
  qdsbValRangeFrom->setSingleStep(0.1);
  ui->toolBar->addWidget(qdsbValRangeFrom);

  QLabel *lblRangeTo = new QLabel("to:");
  ui->toolBar->addWidget(lblRangeTo);

  qdsbValRangeTo = new QDoubleSpinBox();
  qdsbValRangeTo->setRange(-1e7, 1e7);
  qdsbValRangeTo->setValue(1);
  qdsbValRangeTo->setDecimals(4);
  qdsbValRangeTo->setSingleStep(0.1);
  ui->toolBar->addWidget(qdsbValRangeTo);

  QLabel *lbl2 = new QLabel("tr:");
  ui->toolBar->addWidget(lbl2);

  qdsbTransparency = new QDoubleSpinBox();
  qdsbTransparency->setRange(0, 1000);
  qdsbTransparency->setValue(0);
  qdsbTransparency->setDecimals(1);
  qdsbTransparency->setSingleStep(0.1);
  ui->toolBar->addWidget(qdsbTransparency);

  QLabel *lblHScale = new QLabel(" h_sc:");
  ui->toolBar->addWidget(lblHScale);

  qdsbHedgehogScale = new QDoubleSpinBox();
  qdsbHedgehogScale->setRange(0, 1000);
  qdsbHedgehogScale->setValue(1.0);
  qdsbHedgehogScale->setDecimals(2);
  qdsbHedgehogScale->setSingleStep(0.1);
  ui->toolBar->addWidget(qdsbHedgehogScale);

  QLabel *lblArrow = new QLabel(" arrow:");
  ui->toolBar->addWidget(lblArrow);

  qsbArrowheadSize = new QSpinBox();
  qsbArrowheadSize->setRange(0, 10000);
  qsbArrowheadSize->setValue(100);
  ui->toolBar->addWidget(qsbArrowheadSize);

  QLabel *lblPtSize = new QLabel(" pts:");
  ui->toolBar->addWidget(lblPtSize);

  qsbParticleSize = new QSpinBox();
  qsbParticleSize->setRange(1, 100);
  qsbParticleSize->setValue(1);
  ui->toolBar->addWidget(qsbParticleSize);

  QLabel *lbl3 = new QLabel("sldn:");
  ui->toolBar->addWidget(lbl3);

  qsbIntentionalSlowdown = new QSpinBox();
  qsbIntentionalSlowdown->setRange(0, 1000);
  qsbIntentionalSlowdown->setValue(0);
  ui->toolBar->addWidget(qsbIntentionalSlowdown);

  // statusbar
  statusLabel = new QLabel();
  labelElapsedTime = new QLabel();
  labelStepCount = new QLabel();
  labelDateTime = new QLabel();

  QSizePolicy sp;
  const int status_width = 90;
  sp.setHorizontalPolicy(QSizePolicy::Fixed);
  labelStepCount->setSizePolicy(sp);
  labelStepCount->setFixedWidth(status_width);
  labelElapsedTime->setSizePolicy(sp);
  labelElapsedTime->setFixedWidth(status_width);
  labelDateTime->setSizePolicy(sp);
  labelDateTime->setFixedWidth(160);

  ui->statusbar->addWidget(statusLabel);
  ui->statusbar->addPermanentWidget(labelDateTime);
  ui->statusbar->addPermanentWidget(labelElapsedTime);
  ui->statusbar->addPermanentWidget(labelStepCount);

  // anything that includes the Model
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

  renderer->AddActor(representation.hedgehog_actor_vectors);

  // populate combobox
  for (const auto &[key, text] : VisOpt::descriptions) {
    comboBox_visualizations->addItem(
        QString::fromStdString(text.first),
        QVariant::fromValue(static_cast<int>(key)));
  }

  connect(comboBox_visualizations,
          QOverload<int>::of(&QComboBox::currentIndexChanged),
          [&](int index) { comboboxIndexChanged_visualizations(index); });

  for (const auto &[key, text] : VisOpt::hedgehog_descriptions) {
    comboBox_hedgehog->addItem(QString::fromStdString(text.first),
                               QVariant::fromValue(static_cast<int>(key)));
  }

  connect(comboBox_hedgehog,
          QOverload<int>::of(&QComboBox::currentIndexChanged),
          [&](int index) { comboboxIndexChanged_hedgehog(index); });

  // read/restore saved settings
  settingsFileName = QDir::currentPath() + "/cm.ini";
  QFileInfo fi(settingsFileName);

  if (fi.exists()) {
    QSettings settings(settingsFileName, QSettings::IniFormat);
    QVariant var;

    vtkCamera *camera = renderer->GetActiveCamera();
    renderer->ResetCamera();
    camera->ParallelProjectionOn();

    var = settings.value("camData");
    if (!var.isNull()) {
      double *vec = (double *)var.toByteArray().constData();
      camera->SetClippingRange(1e-1, 1e4);
      camera->SetViewUp(0.0, 1.0, 0.0);
      camera->SetPosition(vec[0], vec[1], vec[2]);
      camera->SetFocalPoint(vec[3], vec[4], vec[5]);
      camera->SetParallelScale(vec[6]);
      camera->Modified();
    }

    comboBox_visualizations->setCurrentIndex(
        settings.value("vis_option").toInt());

    var = settings.value("hedgehog_option");
    if (!var.isNull()) {
      comboBox_hedgehog->setCurrentIndex(var.toInt());
    }

    QVariant arr_size = settings.value("arrowhead_size");
    if (!arr_size.isNull()) {
      qsbArrowheadSize->blockSignals(true);
      qsbArrowheadSize->setValue(arr_size.toInt());
      qsbArrowheadSize->blockSignals(false);
      representation.SetArrowheadSize(arr_size.toInt() * 10.0);
    } else {
      representation.SetArrowheadSize(100 * 10.0);
    }

    QVariant pt_size = settings.value("particle_size");
    if (!pt_size.isNull()) {
      qsbParticleSize->blockSignals(true);
      qsbParticleSize->setValue(pt_size.toInt());
      qsbParticleSize->blockSignals(false);
      representation.SetParticleViewSize(pt_size.toInt());
    } else {
      representation.SetParticleViewSize(1);
    }

    var = settings.value("vis_option");
    if (!var.isNull()) {
      comboBox_visualizations->setCurrentIndex(var.toInt());
      qdsbValRangeFrom->setValue(VisOpt::range_from[var.toInt()]);
      qdsbValRangeTo->setValue(VisOpt::range_to[var.toInt()]);
    }

    var = settings.value("hedgehog_option");
    if (!var.isNull()) {
      comboBox_hedgehog->setCurrentIndex(var.toInt());
      qdsbHedgehogScale->setValue(VisOpt::hedgehog_ranges[var.toInt()]);
    }
  } else {
    cameraNeedsReset = true;
    int idx = comboBox_visualizations->findData(QVariant::fromValue(static_cast<int>(VisOpt::grid_fracture_type)));
    if (idx != -1) {
        comboBox_visualizations->setCurrentIndex(idx);
    }
  }

  connect(ui->action_quit, &QAction::triggered, this,
          &MainWindow::quit_triggered);
  connect(ui->action_camera_reset, &QAction::triggered, this,
          &MainWindow::cameraReset_triggered);
  connect(ui->actionStart_Pause, &QAction::triggered, this,
          &MainWindow::simulation_start_pause);
  connect(ui->actionLoad_Parameters, &QAction::triggered, this,
          &MainWindow::load_parameter_triggered);
  connect(ui->actionView_ScalarBar, &QAction::triggered, this,
          &MainWindow::toggle_scalarbar);

  connect(qdsbValRangeFrom, QOverload<double>::of(&QDoubleSpinBox::valueChanged),
          this, &MainWindow::limits_changed);
  connect(qdsbValRangeTo, QOverload<double>::of(&QDoubleSpinBox::valueChanged),
          this, &MainWindow::limits_changed);
  connect(qdsbTransparency,
          QOverload<double>::of(&QDoubleSpinBox::valueChanged), this,
          &MainWindow::limits_changed);
  connect(qdsbHedgehogScale,
          QOverload<double>::of(&QDoubleSpinBox::valueChanged), this,
          &MainWindow::hedgehog_scale_changed);
  connect(qsbArrowheadSize, QOverload<int>::of(&QSpinBox::valueChanged), this,
          &MainWindow::arrowhead_size_changed);
  connect(qsbParticleSize, QOverload<int>::of(&QSpinBox::valueChanged), this,
          &MainWindow::particle_size_changed);
  connect(qsbIntentionalSlowdown, QOverload<int>::of(&QSpinBox::valueChanged),
          this, &MainWindow::spinbox_slowdown_value_changed);

  connect(worker, SIGNAL(workerPaused()), SLOT(background_worker_paused()));
  connect(worker, SIGNAL(stepCompleted()), SLOT(simulation_data_ready()));

  qDebug() << "MainWindow constructor done";
}

void MainWindow::closeEvent(QCloseEvent *event) {
  quit_triggered();
  event->accept();
}

void MainWindow::quit_triggered() {
  qDebug() << "MainWindow::quit_triggered() ";
  worker->Finalize();
  // save settings and stop simulation
  QSettings settings(settingsFileName, QSettings::IniFormat);
  qDebug() << "MainWindow: closing main window; " << settings.fileName();

  double data[10];
  renderer->GetActiveCamera()->GetPosition(&data[0]);
  renderer->GetActiveCamera()->GetFocalPoint(&data[3]);
  data[6] = renderer->GetActiveCamera()->GetParallelScale();

  qDebug() << "cam pos " << data[0] << "," << data[1] << "," << data[2];
  qDebug() << "cam focal pt " << data[3] << "," << data[4] << "," << data[5];
  qDebug() << "cam par scale " << data[6];

  QByteArray arr((char *)data, sizeof(data));
  settings.setValue("camData", arr);

  settings.setValue("vis_option", comboBox_visualizations->currentIndex());
  settings.setValue("hedgehog_option", comboBox_hedgehog->currentIndex());
  settings.setValue("arrowhead_size", qsbArrowheadSize->value());
  settings.setValue("particle_size", qsbParticleSize->value());

  QApplication::quit();
}

void MainWindow::comboboxIndexChanged_visualizations(int index) {
  if (model.prms.GridXTotal <= 0)
    return;

  {
    std::lock_guard<std::mutex> lg(model.lock_data_for_GUI);
    representation.ChangeVisualizationOption(index);
  }
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

void MainWindow::comboboxIndexChanged_hedgehog(int index) {
  {
    std::lock_guard<std::mutex> lg(model.lock_data_for_GUI);
    representation.HedgehogVariable = (VisOpt::HedgehogType)index;
    representation.SynchronizeTopology();
  }
  qdsbHedgehogScale->blockSignals(true);
  qdsbHedgehogScale->setValue(VisOpt::hedgehog_ranges[index]);
  qdsbHedgehogScale->blockSignals(false);
  renderWindow->Render();
}

void MainWindow::limits_changed(double val_) {
  int idx = (int)representation.VisualizingVariable;
  VisOpt::range_from[idx] = qdsbValRangeFrom->value();
  VisOpt::range_to[idx] = qdsbValRangeTo->value();
  VisOpt::transparency_coeffs[idx] = qdsbTransparency->value();
  std::lock_guard<std::mutex> lg(model.lock_data_for_GUI);
  representation.SynchronizeTopology();
  renderWindow->Render();
}

void MainWindow::hedgehog_scale_changed(double val_) {
  int idx = (int)representation.HedgehogVariable;
  VisOpt::hedgehog_ranges[idx] = qdsbHedgehogScale->value();
  std::lock_guard<std::mutex> lg(model.lock_data_for_GUI);
  representation.SynchronizeTopology();
  renderWindow->Render();
}

void MainWindow::cameraReset_triggered() {
  qDebug() << "MainWindow::on_action_camera_reset_triggered()";
  vtkCamera *camera = renderer->GetActiveCamera();
  renderer->ResetCamera();
  camera->ParallelProjectionOn();
  camera->SetClippingRange(1e-1, 1e3);

  const double dx =
      model.prms.cellsize * model.prms.InitializationImageSizeX / 2;
  const double dy =
      model.prms.cellsize * model.prms.InitializationImageSizeY / 2;

  camera->SetPosition(dx, dy, 50.);
  camera->SetFocalPoint(dx, dy, 0.);
  camera->SetViewUp(0.0, 1.0, 0.0);
  camera->SetParallelScale(std::min(dx, dy) * 1.1);

  camera->Modified();
  renderWindow->Render();
}

void MainWindow::load_parameter_triggered() {
  QString qFileName = QFileDialog::getOpenFileName(
      this, "Load Parameters", QDir::currentPath(), "JSON Files (*.json)");
  if (qFileName.isNull())
    return;
  LoadParameterFile(qFileName);
}

void MainWindow::simulation_data_ready() { updateGUI(); }

void MainWindow::updateGUI() {
  // LOGV("updateGUI");
  labelStepCount->setText(QString::number(model.prms.SimulationStep));
  labelElapsedTime->setText(
      QString("%1 s").arg(model.prms.SimulationTime, 0, 'f', 0));
  labelDateTime->setText(FormatUtcDateTime(
      model.prms.SimulationStartTime + (long long)model.prms.SimulationTime));

  // statusLabel->setText(QString("per cycle: %1
  // ms").arg(model.compute_time_per_cycle,0,'f',3));

  {
    std::lock_guard<std::mutex> lg(model.lock_data_for_GUI);
    model.SyncTopologyRequired = false;
    representation.simulationTime = model.prms.SimulationTime;
    representation.UpdateTimeText();
    representation.SynchronizeTopology();
  }

  renderWindow->Render();
  worker->visual_update_requested = false;
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

void MainWindow::simulation_start_pause(bool checked) {
  if (!worker->running && checked) {
    qDebug() << "starting simulation via GUI";
    statusLabel->setText("starting simulation");
    worker->Resume();
  } else if (worker->running && !checked) {
    qDebug() << "pausing simulation via GUI";
    statusLabel->setText("pausing simulation");
    model.pause_requested = true;
    ui->actionStart_Pause->setEnabled(false);
  }
}

void MainWindow::background_worker_paused() {
  ui->actionStart_Pause->blockSignals(true);
  ui->actionStart_Pause->setEnabled(true);
  ui->actionStart_Pause->setChecked(false);
  ui->actionStart_Pause->blockSignals(false);
  statusLabel->setText("simulation stopped");
}

void MainWindow::LoadParameterFile(QString qFileName) {
  qDebug() << "MainWindow::LoadParameterFile " << qFileName;
  model.LoadParameterFile(qFileName.toStdString());

  this->setWindowTitle(qFileName);

  // Apply the selected visualization option now that the model is loaded
  comboboxIndexChanged_visualizations(comboBox_visualizations->currentIndex());

  updateGUI();

  if (cameraNeedsReset) {
    cameraReset_triggered();
    cameraNeedsReset = false;
  }
}

void MainWindow::spinbox_slowdown_value_changed(int val) {
  model.intentionalSlowdown = val;
}

void MainWindow::toggle_scalarbar(bool checked) {
  representation.scalarBar->SetVisibility(checked);
  renderWindow->Render();
}

void MainWindow::arrowhead_size_changed(int size) {
  std::lock_guard<std::mutex> lg(model.lock_data_for_GUI);
  representation.SetArrowheadSize(size * 10.0);
  renderWindow->Render();
}

void MainWindow::particle_size_changed(int size) {
  std::lock_guard<std::mutex> lg(model.lock_data_for_GUI);
  representation.SetParticleViewSize(size);
  renderWindow->Render();
}
