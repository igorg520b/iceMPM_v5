#include "directory_manager.h"

#include <stdexcept>

void DirectoryManager::Configure(const std::string& runtimeDirectory_, const std::string& projectName_)
{
    if (configured) {
        throw std::logic_error("DirectoryManager::Configure called more than once");
    }
    configured = true;
    runtimeDirectory = runtimeDirectory_;
    projectName = projectName_;
}

void DirectoryManager::RequireConfigured() const
{
    if (!configured || runtimeDirectory.empty()) {
        throw std::runtime_error(
            "DirectoryManager: RuntimeDirectory is not set -- add \"RuntimeDirectory\" "
            "to the config before requesting any generated-data path");
    }
}

std::string DirectoryManager::DataCacheRoot() const
{
    RequireConfigured();
    return runtimeDirectory + "/_data";
}

std::string DirectoryManager::ProjectDirectory() const
{
    RequireConfigured();
    return runtimeDirectory + "/input/" + projectName;
}

std::string DirectoryManager::SnapshotsDirectory() const
{
    return ProjectDirectory() + "/snapshots";
}

std::string DirectoryManager::OutputDirectory() const
{
    return ProjectDirectory() + "/output";
}

std::string DirectoryManager::FramesDirectory() const
{
    return OutputDirectory() + "/frames";
}

std::string DirectoryManager::LogsDirectory() const
{
    return OutputDirectory() + "/logs";
}

std::string DirectoryManager::SpinupIceStrengthDirectory() const
{
    return OutputDirectory() + "/spinup_ice_strength";
}

std::string DirectoryManager::SpinupTemperatureDirectory() const
{
    return OutputDirectory() + "/spinup_temperature";
}
