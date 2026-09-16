#include "preparer_mainwindow.h"

#include <QApplication>
#include <QCommandLineParser>
#include <QTimer>
#include <QFileInfo>
#include <QDir>
#include <QVTKOpenGLNativeWidget.h>
#include <QSurfaceFormat>
#include <omp.h>


int main(int argc, char *argv[])
{
    std::cout << "num_threads " << omp_get_max_threads() << std::endl;
    std::cout << "testing threads" << std::endl;
    int nthreads, tid;
#pragma omp parallel
    { std::cout << omp_get_thread_num(); }
    std::cout << std::endl;

    QApplication a(argc, argv);
    QSurfaceFormat::setDefaultFormat(QVTKOpenGLNativeWidget::defaultFormat());
    QApplication::setAttribute(Qt::AA_ShareOpenGLContexts);

    QCommandLineParser parser;
    parser.setApplicationDescription("Prepating the input files for MPM simulation");
    parser.addPositionalArgument("parameters", QCoreApplication::translate("main", "JSON parameter file"));

    parser.process(a);

    const QStringList args = parser.positionalArguments();

    MainWindow w;

    if(args.size() >= 1)
    {
        QString parametersFile = args[0];

        // Check if the provided path is a directory
        QFileInfo fileInfo(parametersFile);
        if (fileInfo.isDir()) {
            // If it's a directory, append default filename "simulation.json"
            // (single consolidated config, read by both ParameterParser and
            // SimParams -- see cli_preparer_main.cpp for the CLI equivalent)
            parametersFile = QDir(parametersFile).filePath("simulation.json");
        }

        w.LoadParameterFile(parametersFile);
    }

    w.resize(1400, 900);
    w.show();
    return a.exec();

    return 0;
}
