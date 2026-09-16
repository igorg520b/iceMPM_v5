cmake_minimum_required(VERSION 3.20.2)

project(iceMPM_v5 LANGUAGES CXX C CUDA)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcudafe --diag_suppress=esa_on_defaulted_function_ignored -use_fast_maths --expt-relaxed-constexpr")
set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)

option(BUILD_CLI_VERSION "Build CLI version" ON)
option(BUILD_GUI_VERSION "Build Qt version" OFF)
option(BUILD_VISUALIZER "Build the tool for visualization" OFF)
option(BUILD_PREPARER "Build the tool for preparing the data" OFF)
option(BUILD_PREPARER_CLI "Build the CLI tool for preparing the data" OFF)

option(ENABLE_NAN_CHECKS "Perform NaN checks during the simulation" OFF)
option(ENABLE_SCALING_FACTOR "Account for the projection distortion" OFF)

if(BUILD_GUI_VERSION OR BUILD_VISUALIZER OR BUILD_PREPARER)
    set(CMAKE_AUTOUIC ON)
    set(CMAKE_AUTOMOC ON)
    set(CMAKE_AUTORCC ON)
endif()

include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

find_package(Eigen3 REQUIRED NO_MODULE)
find_package(HDF5 REQUIRED COMPONENTS CXX C)
find_package(spdlog REQUIRED)
find_package(fmt REQUIRED)
find_package(PNG REQUIRED)
find_package(PkgConfig REQUIRED)

# Try CMake config first (works on Wisteria)
find_package(netCDF QUIET)
find_package(netCDFCxx QUIET)

if(TARGET netCDF::netcdf AND TARGET netCDF::netcdf-cxx4)
    message(STATUS "Using netCDF via CMake config")

    set(NETCDF_LIBRARIES
        netCDF::netcdf
        netCDF::netcdf-cxx4
    )

else()
    message(STATUS "Falling back to pkg-config")

    find_package(PkgConfig REQUIRED)
    pkg_check_modules(NETCDF REQUIRED IMPORTED_TARGET netcdf-cxx4)

    set(NETCDF_LIBRARIES PkgConfig::NETCDF)
endif()


find_package(Qt6 COMPONENTS Widgets QUIET)

if(Qt6_FOUND)
    set(QT_LIB Qt6)
else()
    find_package(Qt5 COMPONENTS Widgets REQUIRED)
    set(QT_LIB Qt5)
endif()


if(BUILD_GUI_VERSION)
# Qt GUI executable


      find_package(VTK 9.1 REQUIRED COMPONENTS
      CommonColor
      CommonCore
      FiltersSources
      FiltersGeometry
      InteractionStyle
      RenderingContextOpenGL2
      RenderingCore
      RenderingFreeType
      RenderingGL2PSOpenGL2
      RenderingOpenGL2
      IOGeometry
      RenderingAnnotation
      RenderingLabel
      IOImage
      ViewsQt
      FiltersFlowPaths
    )
    add_compile_definitions(QT_NO_KEYWORDS)

    add_executable(gplate
        gui/main.cpp

        gui/windows/mainwindow.cpp
        gui/windows/mainwindow.h
        gui/windows/mainwindow.ui
        icons.qrc

        gui/vtk/visual_representation.cpp
        gui/vtk/visual_representation.h
        gui/vtk/visualization_options.h
        gui/vtk/visualization_options.cpp
        gui/vtk/vtkHedgeHogCustom.cpp
        gui/vtk/vtkHedgeHogCustom.h

        gui/backgroundworker.cpp
        gui/backgroundworker.h

        # simulation
        simulation/parameters_sim.h
        simulation/parameters_sim.cpp
        simulation/config_loader.h
        simulation/config_loader.cpp
        simulation/projection.cpp
        simulation/model.cpp
        simulation/model.h

        simulation/givens.cuh
        simulation/helper_math.cuh
        simulation/gpu_implementation5.h
        simulation/gpu_implementation5.cpp

        simulation/hssoa/host_side_soa.h
        simulation/hssoa/host_side_soa.cpp
        simulation/hssoa/proxypoint2d.h
        simulation/hssoa/proxypoint2d.cpp

        simulation/gpu_partition.h
        simulation/partition_params.h
        simulation/gpu_partition.cpp
        simulation/kernel_declarations.cuh
        simulation/kernels.cu

        simulation/data_manager/host_side_data.h
        simulation/data_manager/host_side_data.cpp
        simulation/data_manager/windinterpolator.cpp
        simulation/data_manager/directory_manager.cpp
        simulation/data_manager/directory_manager.h
        simulation/data_manager/windinterpolator.h
        simulation/data_manager/currentinterpolator.cpp
        simulation/data_manager/currentinterpolator.h

        simulation/data_manager/poisson_disk_sampling.h

        gui/colormap.h
        gui/colormap.cpp
        simulation/data_manager/intinterval.h
    )
    set_source_files_properties(simulation/gpu_partition.cpp simulation/kernels.cu simulation/kernel_declarations.cuh PROPERTIES LANGUAGE CUDA)

    set_target_properties(gplate PROPERTIES CUDA_ARCHITECTURES "80;89")
    set_target_properties(gplate PROPERTIES CUDA_SEPARABLE_COMPILATION ON CUDA_RESOLVE_DEVICE_SYMBOLS ON)

    target_include_directories(gplate PRIVATE gui gui/windows gui/vtk simulation simulation/hssoa simulation/data_manager External External/rapidjson)
    target_link_libraries(gplate PRIVATE ${QT_LIB}::Widgets)

    # Eigen
    find_package(Eigen3 REQUIRED NO_MODULE)
    target_link_libraries(gplate PRIVATE Eigen3::Eigen)

    # VTK
    target_link_libraries(gplate PRIVATE ${VTK_LIBRARIES})

    # HDF5
    target_link_libraries(gplate PRIVATE HDF5::HDF5)
    target_include_directories(gplate PRIVATE ${HDF5_INCLUDE_DIRS})

    # spdlog
    target_link_libraries(gplate PRIVATE spdlog::spdlog)

    # OpenMP
    target_compile_options(gplate PRIVATE -fopenmp -Wno-stringop-overflow -Wno-invalid-offsetof)
    target_link_libraries(gplate PRIVATE gomp)

    target_link_libraries(gplate PRIVATE ${NETCDF_LIBRARIES})

    vtk_module_autoinit(
    TARGETS gplate
    MODULES ${VTK_LIBRARIES}
    )

    if(ENABLE_NAN_CHECKS)
        target_compile_definitions(gplate PRIVATE ENABLE_NAN_CHECKS)
    endif()

    if(ENABLE_SCALING_FACTOR)
        target_compile_definitions(gplate PRIVATE ENABLE_SCALING_FACTOR)
    endif()


endif()




if(BUILD_CLI_VERSION)

# CLI version
    add_executable(cplate
        cli/main.cpp

        # simulation
        simulation/parameters_sim.h
        simulation/parameters_sim.cpp
        simulation/config_loader.h
        simulation/config_loader.cpp
        simulation/projection.cpp
        simulation/model.cpp
        simulation/model.h

        simulation/givens.cuh
        simulation/helper_math.cuh
        simulation/gpu_implementation5.h
        simulation/gpu_implementation5.cpp

        simulation/hssoa/host_side_soa.h
        simulation/hssoa/host_side_soa.cpp
        simulation/hssoa/proxypoint2d.h
        simulation/hssoa/proxypoint2d.cpp

        simulation/gpu_partition.h
        simulation/partition_params.h
        simulation/gpu_partition.cpp
        simulation/kernel_declarations.cuh
        simulation/kernels.cu

        simulation/data_manager/host_side_data.h
        simulation/data_manager/host_side_data.cpp
        simulation/data_manager/windinterpolator.cpp
        simulation/data_manager/directory_manager.cpp
        simulation/data_manager/directory_manager.h
        simulation/data_manager/windinterpolator.h
        simulation/data_manager/currentinterpolator.cpp
        simulation/data_manager/currentinterpolator.h

        simulation/data_manager/poisson_disk_sampling.h

        gui/colormap.h
        gui/colormap.cpp

        gui/vtk/visualization_options.h
        gui/vtk/visualization_options.cpp

        simulation/data_manager/intinterval.h
    )
    set_source_files_properties(simulation/gpu_partition.cpp simulation/kernels.cu simulation/kernel_declarations.cuh PROPERTIES LANGUAGE CUDA)

    set_target_properties(cplate PROPERTIES CUDA_ARCHITECTURES "80;89")
    set_target_properties(cplate PROPERTIES CUDA_SEPARABLE_COMPILATION ON CUDA_RESOLVE_DEVICE_SYMBOLS ON)

    target_link_libraries(cplate PRIVATE Eigen3::Eigen spdlog::spdlog)
    target_link_libraries(cplate PRIVATE fmt::fmt)

    target_include_directories(cplate PRIVATE simulation gui/vtk simulation/hssoa simulation/data_manager External External/rapidjson)

    target_link_libraries(cplate PRIVATE ${HDF5_LIBRARIES} ${HDF5_CXX_LIBRARIES})

    target_include_directories(cplate PRIVATE ${HDF5_INCLUDE_DIRS})

    target_compile_options(cplate PRIVATE -fopenmp -Wno-stringop-overflow -Wno-invalid-offsetof)
    target_link_libraries(cplate PRIVATE gomp)

    target_link_libraries(cplate PRIVATE ${NETCDF_LIBRARIES})

    if(ENABLE_NAN_CHECKS)
        target_compile_definitions(cplate PRIVATE ENABLE_NAN_CHECKS)
    endif()

    if(ENABLE_SCALING_FACTOR)
        target_compile_definitions(cplate PRIVATE ENABLE_SCALING_FACTOR)
    endif()

endif()




if(BUILD_VISUALIZER)
    find_package(VTK 9.1 REQUIRED COMPONENTS
      CommonColor
      CommonCore
      FiltersSources
      FiltersGeometry
      InteractionStyle
      RenderingContextOpenGL2
      RenderingCore
      RenderingFreeType
      RenderingGL2PSOpenGL2
      RenderingOpenGL2
      IOGeometry
      RenderingAnnotation
      RenderingLabel
      IOImage
      ViewsQt
      FiltersFlowPaths
    )
    add_compile_definitions(QT_NO_KEYWORDS)

    add_executable(visualizer
        visualizer/pp_main.cpp
        visualizer/window/visualizer_mainwindow.cpp
        visualizer/window/visualizer_mainwindow.h
        visualizer/window/render_selector_dialog.cpp
        visualizer/window/render_selector_dialog.h
        icons.qrc

        gui/vtk/visual_representation.cpp
        gui/vtk/visual_representation.h
        gui/vtk/visualization_options.h
        gui/vtk/visualization_options.cpp
        gui/vtk/vtkHedgeHogCustom.cpp
        gui/vtk/vtkHedgeHogCustom.h
        gui/vtk/wind_rose_widget.cpp
        gui/vtk/wind_rose_widget.h


        simulation/parameters_sim.h
        simulation/parameters_sim.cpp
        simulation/config_loader.h
        simulation/config_loader.cpp
        simulation/projection.cpp

        gui/colormap.h
        gui/colormap.cpp

        simulation/hssoa/host_side_soa.h
        simulation/hssoa/host_side_soa.cpp
        simulation/hssoa/proxypoint2d.h
        simulation/hssoa/proxypoint2d.cpp

        simulation/data_manager/windinterpolator.cpp
        simulation/data_manager/directory_manager.cpp
        simulation/data_manager/directory_manager.h
        simulation/data_manager/windinterpolator.h
        simulation/data_manager/currentinterpolator.cpp
        simulation/data_manager/currentinterpolator.h


        simulation/data_manager/intinterval.h

        visualizer/frame_utils.h
        visualizer/frame_utils.cpp

        simulation/data_manager/host_side_data.h
        simulation/data_manager/host_side_data.cpp
    )

    target_include_directories(visualizer PRIVATE visualizer visualizer/window gui External gui/vtk)
    target_include_directories(visualizer PRIVATE simulation simulation/hssoa simulation/data_manager)
    target_link_libraries(visualizer PRIVATE ${QT_LIB}::Widgets)

    # Eigen
    find_package(Eigen3 REQUIRED NO_MODULE)
    target_link_libraries(visualizer PRIVATE Eigen3::Eigen)

    # VTK
    target_link_libraries(visualizer PRIVATE ${VTK_LIBRARIES})

    # HDF5
    target_link_libraries(visualizer PRIVATE HDF5::HDF5)
    target_include_directories(visualizer PRIVATE ${HDF5_INCLUDE_DIRS})

    # spdlog
    target_link_libraries(visualizer PRIVATE spdlog::spdlog)

    target_link_libraries(visualizer PRIVATE ${NETCDF_LIBRARIES})

    vtk_module_autoinit(
    TARGETS visualizer
    MODULES ${VTK_LIBRARIES}
    )

    target_compile_options(visualizer PRIVATE -fopenmp -Wno-stringop-overflow -Wno-invalid-offsetof)
    target_link_libraries(visualizer PRIVATE gomp)

endif()



if(BUILD_PREPARER)

    find_package(VTK 9.1 REQUIRED COMPONENTS
    CommonCore
    IOGeometry
    FiltersSources
    FiltersGeometry
    IOImage
    InteractionStyle
    RenderingContextOpenGL2
    RenderingCore
    RenderingFreeType
    RenderingGL2PSOpenGL2
    RenderingOpenGL2
    RenderingAnnotation
    RenderingLabel
    ViewsQt
    FiltersFlowPaths
    )

    add_executable(preparer
        preparer/preparer_main.cpp
        preparer/preparer_mainwindow.cpp
        preparer/preparer_mainwindow.h

        preparer/parameterparser.h
        preparer/parameterparser.cpp

        preparer/data_preparer.h
        preparer/data_preparer.cpp

        simulation/data_manager/intinterval.h

        gui/vtk/visual_representation.cpp
        gui/vtk/visual_representation.h
        gui/vtk/visualization_options.h
        gui/vtk/visualization_options.cpp
        gui/vtk/vtkHedgeHogCustom.cpp
        gui/vtk/vtkHedgeHogCustom.h

        gui/colormap.h
        gui/colormap.cpp
        simulation/parameters_sim.h
        simulation/parameters_sim.cpp
        simulation/config_loader.h
        simulation/config_loader.cpp
        simulation/projection.cpp

        simulation/hssoa/host_side_soa.h
        simulation/hssoa/host_side_soa.cpp
        simulation/hssoa/proxypoint2d.h
        simulation/hssoa/proxypoint2d.cpp

        simulation/data_manager/host_side_data.h
        simulation/data_manager/host_side_data.cpp
        simulation/data_manager/windinterpolator.cpp
        simulation/data_manager/directory_manager.cpp
        simulation/data_manager/directory_manager.h
        simulation/data_manager/windinterpolator.h
        simulation/data_manager/currentinterpolator.cpp
        simulation/data_manager/currentinterpolator.h

        simulation/data_manager/poisson_disk_sampling.h
    )

    # Note: preparer no longer uses CUDA for Kelvin wake computation
    # Kelvin wake is computed on CPU using OpenMP

    target_link_libraries(preparer PRIVATE spdlog::spdlog Eigen3::Eigen HDF5::HDF5 ${VTK_LIBRARIES})
    target_link_libraries(preparer PRIVATE ${QT_LIB}::Widgets)

    target_include_directories(preparer PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${HDF5_INCLUDE_DIRS})
    target_include_directories(preparer PRIVATE External preparer/vtk preparer/svg preparer/processing preparer gui/vtk gui simulation simulation/hssoa simulation/data_manager)

    target_compile_options(preparer PRIVATE -fopenmp -Wno-stringop-overflow -Wno-invalid-offsetof)
    target_link_libraries(preparer PRIVATE gomp)

    target_link_libraries(preparer PRIVATE PNG::PNG)

    target_link_libraries(preparer PRIVATE ${NETCDF_LIBRARIES})

    vtk_module_autoinit(
    TARGETS preparer
    MODULES ${VTK_LIBRARIES}
    )

endif()



if(BUILD_PREPARER_CLI)
    add_executable(preparer_cli
        preparer/cli/cli_preparer_main.cpp
        preparer/data_preparer.cpp
        simulation/data_manager/host_side_data.cpp
        simulation/parameters_sim.cpp
        simulation/config_loader.h
        simulation/config_loader.cpp
        simulation/projection.cpp
        preparer/parameterparser.cpp
        simulation/hssoa/host_side_soa.cpp
        gui/colormap.cpp
        simulation/data_manager/windinterpolator.cpp
        simulation/data_manager/directory_manager.cpp
        simulation/data_manager/directory_manager.h
        simulation/data_manager/currentinterpolator.cpp
        simulation/hssoa/proxypoint2d.cpp
    )

    target_link_libraries(preparer_cli PRIVATE
        spdlog::spdlog Eigen3::Eigen HDF5::HDF5 PNG::PNG)

    target_include_directories(preparer_cli PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${HDF5_INCLUDE_DIRS})
    target_include_directories(preparer_cli PRIVATE External preparer/vtk preparer/svg preparer/processing preparer gui/vtk gui simulation simulation/hssoa simulation/data_manager)

    target_compile_options(preparer_cli PRIVATE -fopenmp -Wno-stringop-overflow -Wno-invalid-offsetof)
    target_link_libraries(preparer_cli PRIVATE gomp)

    target_link_libraries(preparer_cli PRIVATE ${NETCDF_LIBRARIES})
endif()

