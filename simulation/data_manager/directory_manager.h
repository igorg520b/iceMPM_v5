#ifndef DIRECTORY_MANAGER_H
#define DIRECTORY_MANAGER_H

#include <string>

// Host-only owner of every generated-data path this project writes: the
// shared _data/ cache root and each project's own input/<ProjectName>/
// (grid.h5, snapshots, output/{frames,logs,spinup_*}) directory. Anchored to
// an absolute RuntimeDirectory (set once, from base_config.json) so nothing
// generated ever lands inside the repo, regardless of CWD or where
// simulation.json happens to live.
//
// No fallback: if RuntimeDirectory was never configured (or configured
// empty), every accessor below throws rather than silently degrading to
// some CWD- or config-file-relative path. Never touches SimParams --
// SimParams is copied wholesale into GPU constant memory and cannot hold
// strings.
class DirectoryManager
{
public:
    // Must be called exactly once, right after ParseFile(). Does not itself
    // validate runtimeDirectory -- that check is deferred to the first
    // directory actually requested (see the accessors below), so the error
    // names the specific accessor that failed. Throws std::logic_error on a
    // second call -- a second call almost certainly means two different
    // projects' settings are about to get mixed together, which should
    // crash loudly rather than silently overwrite.
    void Configure(const std::string& runtimeDirectory, const std::string& projectName);

    // Every accessor below throws std::runtime_error if Configure() was
    // never called, or was called with an empty runtimeDirectory.
    std::string DataCacheRoot() const;       // "<runtimeDirectory>/_data"
    std::string ProjectDirectory() const;    // "<runtimeDirectory>/input/<projectName>"
    std::string SnapshotsDirectory() const;  // ProjectDirectory/snapshots
    std::string OutputDirectory() const;     // ProjectDirectory/output
    std::string FramesDirectory() const;     // OutputDirectory/frames
    std::string LogsDirectory() const;       // OutputDirectory/logs

    // Thermal spin-up diagnostic frame dumps (preparer's "Render Spin-Up
    // Frames" option, see preparer_mainwindow.cpp's thermal_spinup_triggered).
    std::string SpinupIceStrengthDirectory() const;  // OutputDirectory/spinup_ice_strength
    std::string SpinupTemperatureDirectory() const;  // OutputDirectory/spinup_temperature

private:
    // Throws std::runtime_error if Configure() wasn't called with a
    // non-empty runtimeDirectory.
    void RequireConfigured() const;

    bool configured = false;
    std::string runtimeDirectory;
    std::string projectName;
};

#endif // DIRECTORY_MANAGER_H
