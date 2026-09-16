// frame_utils.cpp

#include "frame_utils.h"
#include <filesystem>
#include <regex>
#include <fmt/format.h>
#include <spdlog/spdlog.h>
#include "parameters_sim.h"

namespace frame_utils {

int ScanFrameDirectory(const std::string& directoryName)
{
    LOGR("frame_utils::ScanFrameDirectory scanning: {}", directoryName);

    std::filesystem::path dirPath(directoryName);

    // One combined f{:05d}.h5 per frame (see HostSideData::SaveFrame) --
    // also matches the older frame_{:05d}.h5 naming.
    const std::regex filePattern(R"(^(frame_|f)\d+\.h5$)");

    int foundCount = 0;

    if (std::filesystem::exists(dirPath)) {
        for (const auto& entry : std::filesystem::directory_iterator(dirPath)) {
            if (entry.is_regular_file()) {
                const std::string filename = entry.path().filename().string();
                if (std::regex_match(filename, filePattern)) {
                    foundCount++;
                }
            }
        }
    }

    LOGR("frame_utils::ScanFrameDirectory found {} matching frame files in directory {}", foundCount, dirPath.string());
    return foundCount;
}


std::string GetFramePath(const std::string& frameDirectory, int frameNumber)
{
    std::filesystem::path dirPath(frameDirectory);
    std::string baseName = fmt::format(fmt::runtime("f{:05d}.h5"), frameNumber);
    return (dirPath / baseName).string();
}

} // namespace frame_utils
