#include "parameterparser.h"
#include "config_loader.h"

#include <spdlog/spdlog.h>
#include <filesystem>
#include <iostream>


void ParameterParser::LoadParamsFile(std::string fileName)
{
    if(!std::filesystem::exists(fileName))
    {
        spdlog::info("params.json does not exist");
        return;
    }

    // Extract directory path from JSON filename
    std::filesystem::path configPath(fileName);
    ConfigFileDirectory = configPath.parent_path().string();
    if(ConfigFileDirectory.empty()) ConfigFileDirectory = ".";

    rapidjson::Document doc = LoadMergedConfigDocument(fileName);

    if(doc.HasMember("ImageIceMask")) ImageIceMask = doc["ImageIceMask"].GetString();

    if(doc.HasMember("ImageColor")) ImageColor = doc["ImageColor"].GetString();
    if(doc.HasMember("ImageCrushedMask")) ImageCrushedMask = doc["ImageCrushedMask"].GetString();
    if(doc.HasMember("ImageCrackedMask")) ImageCrackedMask = doc["ImageCrackedMask"].GetString();
    if(doc.HasMember("ImageLandMask")) ImageLandMask = doc["ImageLandMask"].GetString();
    if(doc.HasMember("ImageThicknessMask")) ImageThicknessMask = doc["ImageThicknessMask"].GetString();
    if(doc.HasMember("ImageFootprintMask")) ImageFootprintMask = doc["ImageFootprintMask"].GetString();
    if(doc.HasMember("AnalysisRegionMask")) AnalysisRegionMask = doc["AnalysisRegionMask"].GetString();

    if(doc.HasMember("PointsPerCell")) PointsPerCell = doc["PointsPerCell"].GetInt();
    if(doc.HasMember("GeneratePoints")) GeneratePoints = doc["GeneratePoints"].GetBool();
    if(doc.HasMember("LoadSnapshotForSpinUp")) LoadSnapshotForSpinUp = doc["LoadSnapshotForSpinUp"].GetString();

    // Ice thickness scaling parameters
    if(doc.HasMember("ThicknessFrom")) ThicknessFrom = doc["ThicknessFrom"].GetDouble();
    if(doc.HasMember("ThicknessTo")) ThicknessTo = doc["ThicknessTo"].GetDouble();

    if(doc.HasMember("ProportionOfCrackedPoints")) ProportionOfCrackedPoints = doc["ProportionOfCrackedPoints"].GetDouble();
    if(doc.HasMember("StdDevOfThickness")) StdDevOfThickness = doc["StdDevOfThickness"].GetDouble();

    spdlog::info("parameter file loaded\n");
}
