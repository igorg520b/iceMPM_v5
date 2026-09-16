#ifndef CONFIG_LOADER_H
#define CONFIG_LOADER_H

#include <string>
#include "rapidjson/document.h"

// Reads fileName and, if it has a "ParentConfig" field (a path resolved
// relative to fileName's own directory), recursively loads and merges the
// parent chain first, then overlays fileName's own top-level fields on top
// (child wins). A file with no "ParentConfig" just returns its own parsed
// document. Throws std::runtime_error on a missing file, invalid JSON, or a
// ParentConfig cycle.
rapidjson::Document LoadMergedConfigDocument(const std::string& fileName);

#endif // CONFIG_LOADER_H
