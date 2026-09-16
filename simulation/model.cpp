#include "model.h"
#include <spdlog/spdlog.h>
#include <algorithm>
#include <thread>
#include "parameters_sim.h"



bool Model::Step()
{
    std::cout << '\n';
    LOGR("step {} ({}) started; date {}; host pts {}; cap {}",
                 sim_data.prms.SimulationStep, sim_data.prms.AnimationFrameNumber(),
         SimParams::FormatUnixTimeToISO(sim_data.prms.SimulationStartTime + (long long)sim_data.prms.SimulationTime),
         sim_data.hssoa.size, sim_data.hssoa.capacity);
    spdlog::default_logger()->flush();

    gpu.reset_timings();
    // gpu.clear_force_accumulator(); // currently don't record forces
    double simulation_time;
    int count_unupdated_steps = 0;

    do
    {
        const int step = sim_data.prms.SimulationStep + count_unupdated_steps;
        simulation_time = sim_data.prms.InitialTimeStep * step;

        if(step % sim_data.prms.IceStrengthUpdateStepInterval == 0)
        {
            const double dt_th = sim_data.prms.IceStrengthUpdateStepInterval * sim_data.prms.InitialTimeStep;
            gpu.compute_ice_strength(sim_data.windInterp.current_wind_alpha, dt_th);
        }

        gpu.reset_grid();
        gpu.p2g();

        bool frames_changed = false;
        {
            std::lock_guard<std::mutex> lg(lock_data_for_GUI);
            bool ocean_changed = sim_data.currentInterp.SetTime(simulation_time + sim_data.prms.SimulationStartTime);
            bool wind_changed = sim_data.windInterp.SetTime(simulation_time + sim_data.prms.SimulationStartTime);
            if(ocean_changed) gpu.update_ocean_current_field(sim_data.currentInterp);
            if(wind_changed) gpu.update_wind_field(sim_data.windInterp);
        }

        gpu.update_nodes(simulation_time);
        gpu.contact();
        const bool isCycleEnd = (step + 1) % sim_data.prms.UpdateEveryNthStep == 0;
        gpu.g2p(isCycleEnd, step);

        bool attempt_point_transfer = (step) % sim_data.prms.PointTransferPeriod == 0;
        if(attempt_point_transfer) gpu.point_transfer();
        gpu.record_timings();

        count_unupdated_steps++;
        if(intentionalSlowdown) // for GUI to unfreeze
        {
            gpu.synchronize();
            std::this_thread::sleep_for(std::chrono::milliseconds(intentionalSlowdown));
        }
    } while((sim_data.prms.SimulationStep+count_unupdated_steps) % sim_data.prms.UpdateEveryNthStep != 0);

    sim_data.prms.SimulationTime = simulation_time;
    sim_data.prms.SimulationStep += count_unupdated_steps;

    if(m_save_future.valid()) {
        auto t_start = std::chrono::high_resolution_clock::now();
        m_save_future.get();
        auto t_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = t_end - t_start;
        if(diff.count() > 0.5) {
            LOGR("Step(): time waiting for frame save {:.3f} s", diff.count());
        }
    }
    gpu.render_visualized_data();

    bool external_termination = false;

    {
        std::lock_guard<std::mutex> lg(lock_data_for_GUI);
        gpu.transfer_from_device();

        // Normalize timings and check for squeeze conditions
        bool squeeze_required = false;

        // Check external instructions
        if(CheckExternalInstructions(squeeze_required))
        {
            pause_requested = true;
        }

        if(pause_requested)
        {
            external_termination = true;
            pause_requested = false;
        }
        if(scheduled_termination_frame != -1 && sim_data.prms.AnimationFrameNumber() >= scheduled_termination_frame)
        {
            external_termination = true;
            LOGR("Scheduled termination at frame {} reached", scheduled_termination_frame);
        }

        for(GPU_Partition &p : gpu.partitions)
        {
            p.normalize_timings(count_unupdated_steps);
            const unsigned pts_free_slots = p.pparams.pitch_pts-p.pparams.count_pts;

            const float disabled_proportion = (float)p.get_disabled_pts()/p.pparams.count_pts;
            if(disabled_proportion > SimParams::disabled_pts_proportion_threshold) squeeze_required = true;
            const float free_space_proportion = (float)pts_free_slots/p.pparams.pitch_pts;
            if(gpu.partitions.size() > 1 && free_space_proportion < SimParams::free_space_threshold) squeeze_required = true;
        }
        PrintTimingTable();

        if(squeeze_required)
        {
            sim_data.hssoa.RemoveDisabledAndSort(sim_data.prms.GridYTotal);
            gpu.split_hssoa_into_partitions();
            gpu.transfer_to_device();
            sim_data.currentInterp.SetTime(prms.SimulationTime + prms.SimulationStartTime);
            sim_data.windInterp.SetTime(prms.SimulationTime + prms.SimulationStartTime);
            // Rebalancing shifts gridX_offset per partition, which changes
            // which part of the (unmoved) device grid_forcing buffer
            // corresponds to which global grid cell -- so even the static
            // scale-factor plane needs re-transferring here, same as
            // wind/current, even though its own values never change.
            gpu.update_forcing_fields(sim_data.currentInterp, sim_data.windInterp);
            SyncTopologyRequired = true;
            LOGR("Model::Step() squeezing and sorting HSSOA done\n");
        }
    }

    if(transfer_completion_callback) transfer_completion_callback();    // signal GUI to udpate

    // snapshot is synchronous, frame save is async
    bool saveSnapshot = ((sim_data.prms.SimulationStep / sim_data.prms.UpdateEveryNthStep) % sim_data.prms.SnapshotPeriod == 0) ||
                        (sim_data.prms.SimulationTime >= sim_data.prms.SimulationEndTime) || 
                        external_termination;

    if(saveSnapshot) sim_data.SaveSnapshot(sim_data.prms.SimulationStep, sim_data.prms.SimulationTime, false, sim_data.snapshot_directory);  // synchronous

    m_save_future = std::async(std::launch::async, &HostSideData::SaveFrame, &sim_data,
                              sim_data.prms.SimulationStep, sim_data.prms.SimulationTime, true, 3);

    if (external_termination) {
        if (render_on_termination_requested && sim_data.prms.SaveAnimationAsJPG) {
            if (m_save_future.valid()) m_save_future.get();
            RenderAnimation();
        }
        return false;
    }
    return (sim_data.prms.SimulationTime < sim_data.prms.SimulationEndTime && !gpu.error_code && sim_data.hssoa.size);
}

Model::Model() : gpu(sim_data), prms(sim_data.prms)
{
    SyncTopologyRequired = true;
    LOGR("Model constructor done");
}

Model::~Model()
{
    if (m_save_future.valid()) m_save_future.get();
    LOGR("Model destructor done");
}


void Model::Prepare()
{
    LOGR("Model::Prepare()");
    gpu.update_constants();
    sim_data.currentInterp.SetTime(prms.SimulationTime + prms.SimulationStartTime);
    sim_data.windInterp.SetTime(prms.SimulationTime + prms.SimulationStartTime);
    gpu.update_forcing_fields(sim_data.currentInterp, sim_data.windInterp);
}


void Model::PrintTimingTable()
{
    LOGR("finished {:>8.1f} of {:>8.1f} ({}); host pts {}; cap {}; err {:#x}",
         sim_data.prms.SimulationTime, sim_data.prms.SimulationEndTime,
         sim_data.prms.AnimationFrameNumber(), sim_data.hssoa.size, sim_data.hssoa.capacity, gpu.error_code);

    // print out timings
    LOGR("{0:^3s} {1:^9s} {2:^7s} {3:^7s} | {4:^8s} {5:^8s} {6:^8s} | {7:^8s}",
         "P-D",  "pts", "free",  "dis",    "p2g",  "u",  "g2p",   "tot");

    for(GPU_Partition &p : gpu.partitions)
    {
        const unsigned pts_free_slots = p.pparams.pitch_pts-p.pparams.count_pts;

        LOGR("{0:>1}-{1:>1} {2:>9} {3:>7} {4:>7} | {5:>8.1f} {6:>8.1f} {7:>8.1f} | {8:>8.1f}",
             p.pparams.PartitionID, // 0  P-D
             p.Device,              // 1
             p.pparams.count_pts,   // 2 pts
             pts_free_slots, // 3 free space
             p.get_disabled_pts(),   // 4 disabled
             p.timing_10_P2GAndHalo,    // 5 p2g
             p.timing_30_updateGrid,    // 6 grid update
             p.timing_40_G2P,           // 7 g2p
             p.timing_stepTotal);       // 8 total
    }
    LOGR("\n");
}


void Model::LoadParameterFile(std::string fileName)
{
    LOGR("Model::LoadParameterFile {}", fileName);

    // Get JSON directory for resolving relative paths
    std::filesystem::path jsonFileDir = std::filesystem::path(fileName).parent_path();
    if (jsonFileDir.empty()) {
        jsonFileDir = ".";
    }

    // Parse configuration from JSON
    std::vector<std::string> carra1Files;
    std::map<std::string, std::string> parseResult = sim_data.prms.ParseFile(fileName, &carra1Files);
    sim_data.SimulationTitle = parseResult["SimulationTitle"];
    sim_data.dirs.Configure(parseResult["RuntimeDirectory"], parseResult["ProjectName"]);

    // Everything this run reads/writes for the project itself
    // (grid.h5/output/snapshots/logs) is anchored to RuntimeDirectory when
    // configured, so it never lands next to simulation.json inside the
    // repo -- see DirectoryManager. Only CARRA1Data/GLO12Data/etc. (the raw
    // forcing data) stay resolved against jsonFileDir.
    std::filesystem::path projectDir = sim_data.dirs.ProjectDirectory();
    std::filesystem::path outputDir = sim_data.dirs.OutputDirectory();
    std::filesystem::path logDir = sim_data.dirs.LogsDirectory();
    std::filesystem::create_directories(logDir);
    std::filesystem::path fullLogPath = logDir / "multisink.txt";

    auto file_sink = std::make_shared<spdlog::sinks::basic_file_sink_mt>(fullLogPath.string(), true);
    auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
    auto lg = std::make_shared<spdlog::logger>("multi_sink", spdlog::sinks_init_list({console_sink, file_sink}));
    spdlog::set_default_logger(lg);
//    spdlog::set_pattern("%v");

    // Load grid data (pre-created by plate_preparer)
    std::filesystem::path gridPath = projectDir / parseResult["GridData"];
    sim_data.LoadGridDataFromFile(gridPath.string());

    // Cleaned up directory management
    sim_data.data_directory = projectDir.string();
    sim_data.output_directory = outputDir.string();
    sim_data.snapshot_directory = sim_data.dirs.SnapshotsDirectory();

    // Create directories
    std::filesystem::create_directories(sim_data.output_directory);
    std::filesystem::create_directories(sim_data.snapshot_directory);

    std::filesystem::path framesDir = sim_data.dirs.FramesDirectory();
    std::filesystem::create_directories(framesDir);

    // Load points from snapshot - STRICT CHECKING
    std::filesystem::path snapshotPath;
    if (parseResult.count("Snapshot")) {
        snapshotPath = parseResult["Snapshot"];
        if (snapshotPath.is_relative()) {
            snapshotPath = sim_data.dirs.SnapshotsDirectory() / snapshotPath;
        }
    } else {
        throw std::runtime_error("Starting simulation requires 'Snapshot' parameter in .json file");
    }

    if (!std::filesystem::exists(snapshotPath)) {
        throw std::runtime_error(fmt::format("Snapshot file not found: {}", snapshotPath.string()));
    }
    LOGR("Loading snapshot from: {}", snapshotPath.string());
    sim_data.ReadPointsFromSnapshot(snapshotPath.string());
    // sim_data.VerifyPoints();

    // Allocate point arrays and transfer to GPU partitions
    gpu.SplitIntoPartitionsAndTransferToDevice();
    
    for (const std::string& rawPath : carra1Files) {
        std::filesystem::path carra1Path(rawPath);
        if (carra1Path.is_relative()) {
            carra1Path = jsonFileDir / carra1Path;
        }
        sim_data.windInterp.SetCARRA1Path(carra1Path.string());
    }

    // Load GLO12 if present
    if (parseResult.count("GLO12Data")) {
        std::filesystem::path glo12Path = jsonFileDir / parseResult["GLO12Data"];
        sim_data.currentInterp.SetGLO12Path(glo12Path.string());
    }

    if (parseResult.count("GLO12Tides")) {
        std::filesystem::path glo12TidesPath = jsonFileDir / parseResult["GLO12Tides"];
        sim_data.currentInterp.SetGLO12TidesPath(glo12TidesPath.string());
    }

    // sithick (ice thickness) lives in its own GLO12 file in the real Kane
    // Basin deployment (same lat/lon/time grid as currents, different .nc).
    if (parseResult.count("GLO12ThicknessData")) {
        std::filesystem::path glo12ThicknessPath = jsonFileDir / parseResult["GLO12ThicknessData"];
        sim_data.currentInterp.SetGLO12ThicknessPath(glo12ThicknessPath.string());
    }

    LOGR("Model::LoadParameterFile() about to invoke current/wind SetTime");
    sim_data.currentInterp.SetTime(sim_data.prms.SimulationTime + sim_data.prms.SimulationStartTime);
    sim_data.windInterp.SetTime(sim_data.prms.SimulationTime + sim_data.prms.SimulationStartTime);
    gpu.update_ocean_current_field(sim_data.currentInterp);
    gpu.update_wind_field(sim_data.windInterp);

    // Print memory allocation summary
    LOGR("");
    LOGR("Memory allocation summary:");
    LOGR("  Grid data:    {:.3f} GB", sim_data.allocated_bytes[0] / 1e9);
    LOGR("  Particle data: {:.3f} GB", sim_data.allocated_bytes[1] / 1e9);
    LOGR("  Total:        {:.3f} GB", (sim_data.allocated_bytes[0] + sim_data.allocated_bytes[1]) / 1e9);
    LOGR("");
    spdlog::default_logger()->flush();

    // Final GPU preparation and rendering
    Prepare();
    LOGR("LoadParameterFile - invoking gpu.render_visualized_data()");
    spdlog::default_logger()->flush();
    gpu.render_visualized_data();
    LOGR("LoadParameterFile - invoking gpu.transfer_from_device()");
    spdlog::default_logger()->flush();
    gpu.transfer_from_device();

    LOGR("LoadParameterFile completed successfully");
    spdlog::default_logger()->flush();
}


bool Model::CheckExternalInstructions(bool& squeeze_required)
{
    std::filesystem::path instructPath = std::filesystem::path(sim_data.data_directory) / "instruct.txt";
    if (!std::filesystem::exists(instructPath)) return false;

    // 1. Open and read command
    std::ifstream ifs(instructPath);
    std::string command;
    ifs >> command;
    
    int frame = -1;
    bool has_frame = bool(ifs >> frame);

    // 2. Close and delete file immediately
    ifs.close();
    try {
        std::filesystem::remove(instructPath);
    } catch(const std::filesystem::filesystem_error& e) {
        LOGR("Error deleting instruct.txt: {}", e.what());
    }
    
    bool terminate = false;
    
    // 3. Evaluate instruction logic
    if (command == "terminate") {
        render_on_termination_requested = true;
        if (has_frame) {
            scheduled_termination_frame = frame;
            LOGR("Instruction: terminate at frame {}", frame);
        } else {
            terminate = true;
            LOGR("Instruction: terminate immediately");
        }
    } else if (command == "sort") {
        squeeze_required = true;
        LOGR("Instruction: sort (squeeze) forced");
    } else if (command == "render") {
        LOGR("Instruction: render");
    }
    
    // 4. Execute post-instruction actions
    if (command == "render" && sim_data.prms.SaveAnimationAsJPG) {
        RenderAnimation();
    }
    
    return terminate;
}

void Model::RenderAnimation()
{
    const int num_frames = sim_data.prms.SimulationStep / sim_data.prms.UpdateEveryNthStep;
    std::filesystem::path renderDir = std::filesystem::path(sim_data.output_directory) / "raster" / "grid_fracture_type";
    std::string ffmpeg_cmd = fmt::format(
        "cd \"{}\" && ffmpeg -y -r 30 -f image2 -start_number 1 -i \"%05d.jpg\" -vframes {} -vcodec libx264 -vf \"scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:-1:-1:white\" -crf 21 -pix_fmt yuv420p \"render.mp4\"",
        renderDir.string(), num_frames
    );
    LOGR("Render instruction received. Output directory: {}", renderDir.string());
    LOGR("ffmpeg command prepared: {}", ffmpeg_cmd);
    LOGR("ffmpeg is not available on the remote server, skipping execution.");
    // int ret = std::system(ffmpeg_cmd.c_str());
    // if (ret != 0) {
    //     LOGR("Warning: ffmpeg command returned {}", ret);
    // } else {
    //     try {
    //         std::filesystem::path sourceFile = renderDir / "render.mp4";
    //         std::filesystem::path destFile = std::filesystem::path(sim_data.output_directory) / "render.mp4";
    //         if (std::filesystem::exists(sourceFile)) {
    //             if (std::filesystem::exists(destFile)) std::filesystem::remove(destFile);
    //             std::filesystem::rename(sourceFile, destFile);
    //             LOGR("Moved render.mp4 to {}", destFile.string());
    //         }
    //     } catch (const std::filesystem::filesystem_error& e) {
    //         LOGR("Error moving render.mp4: {}", e.what());
    //     }
    // }
}
