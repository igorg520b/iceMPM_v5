#include "config_loader.h"

#include <fstream>
#include <filesystem>
#include <vector>
#include <stdexcept>
#include <fmt/format.h>

namespace {

constexpr int kMaxParentChainDepth = 20;

rapidjson::Document ParseOwnDocument(const std::string& fileName)
{
    if(!std::filesystem::exists(fileName))
        throw std::runtime_error(fmt::format("configuration file not found: {}", fileName));

    std::ifstream fileStream(fileName);
    std::string strConfigFile;
    strConfigFile.resize(std::filesystem::file_size(fileName));
    fileStream.read(strConfigFile.data(), strConfigFile.length());
    fileStream.close();

    rapidjson::Document doc;
    doc.Parse(strConfigFile.data());
    if(!doc.IsObject())
        throw std::runtime_error(fmt::format("configuration file is not JSON: {}", fileName));

    return doc;
}

// Deep-merges own_doc's top-level members on top of "result" (which already
// reflects the full parent chain), using result's own allocator.
void MergeOnTop(rapidjson::Document& result, rapidjson::Document& own_doc)
{
    for(auto it = own_doc.MemberBegin(); it != own_doc.MemberEnd(); ++it)
    {
        const char* keyStr = it->name.GetString();
        if(result.HasMember(keyStr)) result.RemoveMember(keyStr);
        rapidjson::Value key(it->name, result.GetAllocator());
        rapidjson::Value value(it->value, result.GetAllocator());
        result.AddMember(key, value, result.GetAllocator());
    }
}

rapidjson::Document LoadMergedConfigDocumentImpl(const std::string& fileName, std::vector<std::string>& visited)
{
    if((int)visited.size() > kMaxParentChainDepth)
        throw std::runtime_error(fmt::format("ParentConfig chain exceeds max depth ({}): {}", kMaxParentChainDepth, fileName));

    const std::string canonicalPath = std::filesystem::canonical(fileName).string();
    for(const std::string& v : visited)
        if(v == canonicalPath)
            throw std::runtime_error(fmt::format("ParentConfig cycle detected involving: {}", canonicalPath));
    visited.push_back(canonicalPath);

    rapidjson::Document own_doc = ParseOwnDocument(fileName);

    if(!own_doc.HasMember("ParentConfig")) return own_doc;

    std::filesystem::path parentPath = std::filesystem::path(fileName).parent_path() / own_doc["ParentConfig"].GetString();
    rapidjson::Document result = LoadMergedConfigDocumentImpl(parentPath.string(), visited);
    MergeOnTop(result, own_doc);
    return result;
}

} // namespace

rapidjson::Document LoadMergedConfigDocument(const std::string& fileName)
{
    std::vector<std::string> visited;
    return LoadMergedConfigDocumentImpl(fileName, visited);
}
