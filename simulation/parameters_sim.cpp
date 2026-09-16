#include "parameters_sim.h"
#include "config_loader.h"
#include <spdlog/spdlog.h>

#include <string>
#include <filesystem>
#include <cstdio>
#include <ctime>
#include <algorithm>
#include <cmath>

long long SimParams::ParseISODateToUnixTime(const std::string &dateStr)
{
    std::tm tm{};
    int year, month, day, hour = 0, minute = 0, second = 0;
    int n = std::sscanf(dateStr.c_str(), "%d-%d-%dT%d:%d:%d", &year, &month, &day, &hour, &minute, &second);
    if (n < 3) {
        throw std::runtime_error("Could not parse date '" + dateStr + "' (expected YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS)");
    }
    tm.tm_year = year - 1900;
    tm.tm_mon = month - 1;
    tm.tm_mday = day;
    tm.tm_hour = hour;
    tm.tm_min = minute;
    tm.tm_sec = second;
#if defined(_WIN32)
    return static_cast<long long>(_mkgmtime(&tm));
#else
    return static_cast<long long>(timegm(&tm));
#endif
}

std::string SimParams::FormatUnixTimeToISO(long long epoch)
{
    std::time_t tt = static_cast<std::time_t>(epoch);
    std::tm tm_utc{};
    gmtime_r(&tt, &tm_utc);
    char buf[32];
    std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%S", &tm_utc);
    return std::string(buf);
}

void SimParams::Reset()
{
    SimulationStep = 0;
    SimulationTime = 0;
    GridXTotal = GridYTotal = 0;
    ModeledRegionOffsetX = ModeledRegionOffsetY = 0;

    proj.LAT_0 = 0.0;
    proj.LON_0 = 0.0;
    proj.RESIZE_FACTOR = 1.0;
    proj.TRANSFORM_COEFFS[0] = 1.0; proj.TRANSFORM_COEFFS[1] = 0.0; proj.TRANSFORM_COEFFS[2] = 0.0;
    proj.TRANSFORM_COEFFS[3] = 0.0; proj.TRANSFORM_COEFFS[4] = 1.0; proj.TRANSFORM_COEFFS[5] = 0.0;
    InitializationImageSizeX = InitializationImageSizeY = 0;

    nPartitions = 1;
    GridHaloSize = 10;
    HaloDiffusionThreshold = 5;
    PointTransferPeriod = 20;

    SaveSnapshots = false;
    SnapshotPeriod = 25;

    SaveAnimationAsJPG = false;
    JPG_OffsetX = JPG_OffsetY = JPG_Crop_Width = JPG_Crop_Height =0;

    nPtsInitial = 0;

    waterDragNormalized = 5.0e-3;
    windDragNormalized = 1.4e-3;
    TestWindSpeed = 0.0;
    ForcingSpinUpDuration = 3600.0;
    SimulationStartTime = 0;
    PreSimulationStartTime = 0;
    SimulationEndDateTime = 0;

    InitialTimeStep = 3.e-5;
    YoungsModulus = 5.e8;

    SimulationEndTime = 20000;
    AnimationFramePeriod = 200;

    PoissonsRatio = 0.3;
    IceDensity = 916;

    IceTensileStrength = 3.46e5;

    IceCompressiveThreshold = 100e6;

    IceShearStrength = 3.8e5;
    UseLiveIceStrength = true;
    IceStrengthFactorTensile = 1.96;
    AllowFracture = true;
    UseDoubleGrid = true;
    IceFrictionCoeff = 0.3;

    // Placeholder defaults, pending calibration -- see comments in parameters_sim.h.
    IceTensileFailureStrength = 5e4;
    IceFractureAngle = 5.0;
    Cf = 0.4;
    Cf_residual = 0.5;
    GammaStar = 0.1;
    IceYieldFrictionAngle = 20.0;
    TauInv = 2.0;   // tau ~ 0.5s

    DP_phi = 55;
    DP_threshold_p = -280;

    IceSalinity = 6.0;
    IceStrengthFactor = 2.15;
    IceStrengthFactorCompressive = 1.2;
    IceStrengthFactorCompressiveYield = 1.2;
    IceStrengthUpdatePeriod = 60.0;

    tpb_P2G = 256;
    tpb_Upd = 512;
    tpb_G2P = 128;

    InitializationImageSizeX = InitializationImageSizeY = 0;

    UseWindData = false;
    UseGLO12Data = false;
    UseGLO12Tides = false;

    extra_space_pts = 0.15;     // extra allocation for storage of points (to allow transfer between GPU partitions)

    ComputeLame();
    ComputeHelperVariables();
    spdlog::info("SimParams reset");
}



std::map<std::string,std::string> SimParams::ParseFile(std::string fileName, std::vector<std::string>* carra1FilesOut)
{
    LOGR("SimParams ParseFile {}",fileName);
    rapidjson::Document doc = LoadMergedConfigDocument(fileName);

    // parse strings and save into "result"
    std::map<std::string,std::string> result;
    if(doc.HasMember("GridData")) result["GridData"] = doc["GridData"].GetString();
    if(doc.HasMember("Snapshot")) result["Snapshot"] = doc["Snapshot"].GetString();

    // ProjectName/RuntimeDirectory are consumed by DirectoryManager (host-only)
    // to decide where generated data gets written -- never stored on SimParams
    // itself, since SimParams is copied wholesale into GPU constant memory.
    if(doc.HasMember("ProjectName")) result["ProjectName"] = doc["ProjectName"].GetString();
    else result["ProjectName"] = "default_project";
    if(doc.HasMember("RuntimeDirectory")) result["RuntimeDirectory"] = doc["RuntimeDirectory"].GetString();

    if(doc.HasMember("CARRA1Data")) {
        const auto& cd = doc["CARRA1Data"];
        std::vector<std::string> parsedFiles;
        if (cd.IsArray()) {
            for (const auto& v : cd.GetArray()) parsedFiles.push_back(v.GetString());
        } else if (cd.IsString()) {
            parsedFiles.push_back(cd.GetString());
        }
        if (!parsedFiles.empty()) {
            UseWindData = true; // Auto-enable if path(s) provided
            if (carra1FilesOut) *carra1FilesOut = std::move(parsedFiles);
        }
    }

    if(doc.HasMember("GLO12Data")) {
        result["GLO12Data"] = doc["GLO12Data"].GetString();
        UseGLO12Data = true;
    }
    
    // Guarded on non-empty, unlike GLO12Data/GLO12ThicknessData below: a
    // project inheriting from a ParentConfig that sets a real GLO12Tides
    // path needs a way to opt back out (e.g. no tidal-current product
    // exists for the requested date range) by overriding it to "" --
    // without this check that would still flow through to model.cpp as a
    // present-but-empty path and fail loading a directory as an HDF5 file.
    if(doc.HasMember("GLO12Tides") && doc["GLO12Tides"].GetStringLength() > 0) {
        result["GLO12Tides"] = doc["GLO12Tides"].GetString();
        UseGLO12Tides = true;
    }

    if(doc.HasMember("GLO12ThicknessData")) {
        result["GLO12ThicknessData"] = doc["GLO12ThicknessData"].GetString();
        UseGLO12Data = true; // thickness alone (no currents) should still work
    }

    // Projection params (single shared Projection class -- see projection.h)
    proj.ParseJson(doc);

    if(doc.HasMember("SimulationTitle")) result["SimulationTitle"] = doc["SimulationTitle"].GetString();
    else result["SimulationTitle"] = "default_simulation";

    if(doc.HasMember("SaveSnapshots")) SaveSnapshots = doc["SaveSnapshots"].GetBool();

    if(doc.HasMember("SnapshotPeriod")) SnapshotPeriod = doc["SnapshotPeriod"].GetInt();
    if(doc.HasMember("SimulationEndTime")) SimulationEndTime = doc["SimulationEndTime"].GetDouble();

    if(doc.HasMember("InitialTimeStep")) InitialTimeStep = doc["InitialTimeStep"].GetDouble();
    if(doc.HasMember("AnimationFramePeriod")) AnimationFramePeriod = doc["AnimationFramePeriod"].GetDouble();

    if(doc.HasMember("YoungsModulus")) YoungsModulus = doc["YoungsModulus"].GetDouble();
    if(doc.HasMember("PoissonsRatio")) PoissonsRatio = doc["PoissonsRatio"].GetDouble();
    if(doc.HasMember("IceDensity")) IceDensity = doc["IceDensity"].GetDouble();

    if(doc.HasMember("IceTensileStrength")) IceTensileStrength = doc["IceTensileStrength"].GetDouble();
    if(doc.HasMember("IceShearStrength")) IceShearStrength = doc["IceShearStrength"].GetDouble();
    if(doc.HasMember("IceTensileFailureStrength")) IceTensileFailureStrength = doc["IceTensileFailureStrength"].GetDouble();
    if(doc.HasMember("IceFractureAngle")) IceFractureAngle = doc["IceFractureAngle"].GetDouble();
    if(doc.HasMember("Cf")) Cf = doc["Cf"].GetDouble();
    if(doc.HasMember("Cf_residual")) Cf_residual = doc["Cf_residual"].GetDouble();
    if(doc.HasMember("GammaStar")) GammaStar = doc["GammaStar"].GetDouble();
    if(doc.HasMember("IceYieldFrictionAngle")) IceYieldFrictionAngle = doc["IceYieldFrictionAngle"].GetDouble();
    if(doc.HasMember("TauInv")) TauInv = doc["TauInv"].GetDouble();
    if(doc.HasMember("UseLiveIceStrength")) UseLiveIceStrength = doc["UseLiveIceStrength"].GetBool();
    if(doc.HasMember("IceStrengthFactorTensile")) IceStrengthFactorTensile = doc["IceStrengthFactorTensile"].GetDouble();
    if(doc.HasMember("AllowFracture")) AllowFracture = doc["AllowFracture"].GetBool();
    if(doc.HasMember("UseDoubleGrid")) UseDoubleGrid = doc["UseDoubleGrid"].GetBool();

    if(doc.HasMember("IceCompressiveThreshold")) IceCompressiveThreshold = doc["IceCompressiveThreshold"].GetDouble();
    if(doc.HasMember("DP_phi")) DP_phi = doc["DP_phi"].GetDouble();
    if(doc.HasMember("DP_threshold_p")) DP_threshold_p = doc["DP_threshold_p"].GetDouble();
    if(doc.HasMember("IceFrictionCoeff")) IceFrictionCoeff = doc["IceFrictionCoeff"].GetDouble();

    if(doc.HasMember("JPG_OffsetX")) JPG_OffsetX = doc["JPG_OffsetX"].GetInt();
    if(doc.HasMember("JPG_OffsetY")) JPG_OffsetY = doc["JPG_OffsetY"].GetInt();
    if(doc.HasMember("JPG_Crop_Width")) JPG_Crop_Width = doc["JPG_Crop_Width"].GetInt();
    if(doc.HasMember("JPG_Crop_Height")) JPG_Crop_Height = doc["JPG_Crop_Height"].GetInt();
    if(doc.HasMember("SaveAnimationAsJPG")) SaveAnimationAsJPG = doc["SaveAnimationAsJPG"].GetBool();

    if(doc.HasMember("waterDragNormalized")) waterDragNormalized = doc["waterDragNormalized"].GetDouble();
    if(doc.HasMember("windDragNormalized")) windDragNormalized = doc["windDragNormalized"].GetDouble();
    if(doc.HasMember("TestWindSpeed")) TestWindSpeed = doc["TestWindSpeed"].GetDouble();
    if(doc.HasMember("ForcingSpinUpDuration")) ForcingSpinUpDuration = doc["ForcingSpinUpDuration"].GetDouble();
    if(doc.HasMember("SimulationStartDate")) SimulationStartTime = ParseISODateToUnixTime(doc["SimulationStartDate"].GetString());
    if(doc.HasMember("PreSimulationStartDate")) PreSimulationStartTime = ParseISODateToUnixTime(doc["PreSimulationStartDate"].GetString());
    if(doc.HasMember("SimulationEndDate")) SimulationEndDateTime = ParseISODateToUnixTime(doc["SimulationEndDate"].GetString());

    if(doc.HasMember("IceSalinity")) IceSalinity = doc["IceSalinity"].GetDouble();
    if(doc.HasMember("IceStrengthFactor")) IceStrengthFactor = doc["IceStrengthFactor"].GetDouble();
    if(doc.HasMember("IceStrengthFactorCompressive")) IceStrengthFactorCompressive = doc["IceStrengthFactorCompressive"].GetDouble();
    if(doc.HasMember("IceStrengthFactorCompressiveYield")) IceStrengthFactorCompressiveYield = doc["IceStrengthFactorCompressiveYield"].GetDouble();
    if(doc.HasMember("IceStrengthUpdatePeriod")) IceStrengthUpdatePeriod = doc["IceStrengthUpdatePeriod"].GetDouble();

    if(doc.HasMember("tpb_P2G")) tpb_P2G = doc["tpb_P2G"].GetInt();
    if(doc.HasMember("tpb_Upd")) tpb_Upd = doc["tpb_Upd"].GetInt();
    if(doc.HasMember("tpb_G2P")) tpb_G2P = doc["tpb_G2P"].GetInt();

    if(doc.HasMember("nPartitions")) nPartitions = doc["nPartitions"].GetUint();
    if(nPartitions == 1) extra_space_pts = 0;

    if(doc.HasMember("GridHaloSize")) GridHaloSize = doc["GridHaloSize"].GetUint();
    if(doc.HasMember("HaloDiffusionThreshold")) HaloDiffusionThreshold = doc["HaloDiffusionThreshold"].GetUint();

    spdlog::info("SimParams::ParseFile done");
    return result;
}

void SimParams::ComputeLame()
{
    lambda = YoungsModulus*PoissonsRatio/((1+PoissonsRatio)*(1-2*PoissonsRatio));
    mu = YoungsModulus/(2*(1+PoissonsRatio));
    kappa = mu*2./3. + lambda;
}

void SimParams::ComputeHelperVariables()
{
    ParticleMass = ParticleArea * IceDensity;

    UpdateEveryNthStep = (int)(AnimationFramePeriod / InitialTimeStep);
    cellsize_inv = 1./cellsize; // cellsize itself is set when loading .h5 file
    Dp_inv = 4./(cellsize*cellsize);
    dt_area_Dpinv = InitialTimeStep*ParticleArea*Dp_inv;
    vmax = 0.25*cellsize/InitialTimeStep;

    IceStrengthUpdateStepInterval = std::max(1, (int)std::lround(IceStrengthUpdatePeriod / InitialTimeStep));

    DP_tan_phi = std::tan(DP_phi*pi/180.);
    IceFractureAngle_tan = std::tan(IceFractureAngle*pi/180.);
    IceYieldFrictionAngle_tan = std::tan(IceYieldFrictionAngle*pi/180.);

    // TauInv<=0 is a sentinel for instant projection (no viscosity), not the
    // literal math limit -- exp(-dt*0)=1 would mean q_new=q_tr, i.e. zero
    // relaxation ever (the opposite of instant).
    RelaxationAlpha = (TauInv <= 0.0) ? 0.0 : std::exp(-InitialTimeStep*TauInv);

    // compute suggested time step
//    double suggested_dt = 0.8 * cellsize * sqrt(IceDensity*ThicknessFrom/YoungsModulus);
//    LOGR("SimParams::ComputeHelperVariables(): suggested_dt: {}", suggested_dt);

    ComputeLame();
}



void SimParams::Printout()
{
    LOGR("");
    LOGR("Simulation Parameters:");
    LOGR("nPartitions: {}", nPartitions);
    LOGR("extra_space_pts: {:.6g}", extra_space_pts);
    LOGR("InitialTimeStep: {:.6g}, SimulationEndTime: {:.6g}", InitialTimeStep, SimulationEndTime);
    LOGR("AnimationFramePeriod: {:.6g}", AnimationFramePeriod);
    LOGR("GridXTotal: {}, GridYTotal: {}", GridXTotal, GridYTotal);
    LOGR("ModeledRegionOffsetX: {}, ModeledRegionOffsetY: {}", ModeledRegionOffsetX, ModeledRegionOffsetY);
    LOGR("InitializationImageSizeX: {}, InitializationImageSizeY: {}", InitializationImageSizeX, InitializationImageSizeY);
    LOGR("Projection: LAT_0={:.4g}, LON_0={:.4g}, MetersPerPixel={:.6g}", proj.LAT_0, proj.LON_0, proj.MetersPerPixel());
    LOGR("SimulationStartTime (epoch s): {}", SimulationStartTime);
    LOGR("PreSimulationStartTime (epoch s): {}, SimulationEndDateTime (epoch s): {}", PreSimulationStartTime, SimulationEndDateTime);
    LOGR("SimulationStep: {}", SimulationStep);
    LOGR("SimulationTime: {:.6g}", SimulationTime);
    LOGR("UpdateEveryNthStep: {}", UpdateEveryNthStep);

    // parameters
    LOGR("");
    LOGR("Parameters:");
    LOGR("dt_area_Dpinv: {:.6g}, vmax: {:.6g}", dt_area_Dpinv, vmax);

    LOGR("lambda: {:.6g}, mu: {:.6g}, kappa: {:.6g}", lambda, mu, kappa);
    LOGR("ParticleArea: {:.6g}", ParticleArea);
    LOGR("ParticleMass: {:.6g}", ParticleMass);
    LOGR("DP_phi: {:.6g}, DP_threshold_p: {:.6g}", DP_phi, DP_threshold_p);
    LOGR("PoissonsRatio: {:.6g}, YoungsModulus: {:.6g}",
                             PoissonsRatio, YoungsModulus);
    LOGR("IceTensileStrength: {:.6g}, IceShearStrength: {:.6g}, IceTensileFailureStrength: {:.6g}",
                             IceTensileStrength, IceShearStrength, IceTensileFailureStrength);
    LOGR("IceFractureAngle: {:.6g}, Cf: {:.6g}, Cf_residual: {:.6g}, GammaStar: {:.6g}, IceYieldFrictionAngle: {:.6g}",
                             IceFractureAngle, Cf, Cf_residual, GammaStar, IceYieldFrictionAngle);
    LOGR("TauInv: {:.6g}, RelaxationAlpha: {:.6g}", TauInv, RelaxationAlpha);
    LOGR("UseLiveIceStrength: {}, IceStrengthFactorTensile: {:.6g}", UseLiveIceStrength, IceStrengthFactorTensile);
    LOGR("IceCompressiveThreshold: {:.6g}, IceStrengthFactorCompressive: {:.6g}, IceStrengthFactorCompressiveYield: {:.6g}", IceCompressiveThreshold, IceStrengthFactorCompressive, IceStrengthFactorCompressiveYield);
    LOGR("AllowFracture: {}", AllowFracture);
    LOGR("UseDoubleGrid: {}", UseDoubleGrid);
    LOGR("IceFrictionCoeff: {:.6g}", IceFrictionCoeff);

    LOGR("waterDragNormalized: {:.6g}", waterDragNormalized);
    LOGR("windDragNormalized: {:.6g}", windDragNormalized);
    LOGR("TestWindSpeed: {:.6g}", TestWindSpeed);
    LOGR("ForcingSpinUpDuration: {:.6g}", ForcingSpinUpDuration);

    // points
    LOGR("");
    LOGR("Points:");
    LOGR("nPtsInitial: {}", nPtsInitial);

    // grid
    LOGR("");
    LOGR("Grid:");
    LOGR("cellsize: {:.6g}; cellsize*InitializationImageSizeX: {:.6g}",
                             cellsize, cellsize * InitializationImageSizeX);
    LOGR("Sim grid: {} x {}", GridXTotal, GridYTotal);
    LOGR("Original image: {} x {}", InitializationImageSizeX, InitializationImageSizeY);
    LOGR("Offset of the modelled region: [{}, {}]",
                             ModeledRegionOffsetX, ModeledRegionOffsetY);

    LOGR("END PARAMETER PRINTOUT\n");
}

bool SimParams::IsPersistentGridArray(int idx)
{
    // All GPU arrays are cleared before visualization rendering
    // Forces (fx, fy) are summarized in a separate phase before render_visualized_data()
    // So no arrays need to be marked as persistent
    return false;
}
