#include "projection.h"

#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>

#include <spdlog/spdlog.h>
#include <fmt/format.h>

#include "rapidjson/document.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

bool Projection::ParseJson(const rapidjson::Value& doc)
{
    bool found_center = false;

    if (doc.HasMember("PROJ_LAT_0")) { LAT_0 = doc["PROJ_LAT_0"].GetDouble(); found_center = true; }
    if (doc.HasMember("PROJ_LON_0")) { LON_0 = doc["PROJ_LON_0"].GetDouble(); found_center = true; }
    if (doc.HasMember("PROJ_RESIZE_FACTOR")) RESIZE_FACTOR = doc["PROJ_RESIZE_FACTOR"].GetDouble();
    if (doc.HasMember("PROJ_TRANSFORM_COEFFS")) {
        const rapidjson::Value& a = doc["PROJ_TRANSFORM_COEFFS"];
        if (a.IsArray() && a.Size() == 6) {
            for (int i = 0; i < 6; i++) TRANSFORM_COEFFS[i] = a[i].GetDouble();
        } else {
            throw std::runtime_error("PROJ_TRANSFORM_COEFFS must be an array of 6 numbers");
        }
    }

    return found_center;
}

bool Projection::ParseFile(const std::string& fileName)
{
    std::ifstream ifs(fileName);
    if (!ifs.is_open()) {
        throw std::runtime_error(fmt::format("Projection::ParseFile: could not open {}", fileName));
    }
    std::stringstream ss;
    ss << ifs.rdbuf();
    std::string content = ss.str();

    rapidjson::Document doc;
    doc.Parse(content.c_str());
    if (doc.HasParseError()) {
        throw std::runtime_error(fmt::format("Projection::ParseFile: JSON parse error in {}", fileName));
    }

    bool ok = ParseJson(doc);
    spdlog::info(fmt::format("Projection loaded from {}: LAT_0={}, LON_0={}, RESIZE_FACTOR={}, MetersPerPixel={}",
                              fileName, LAT_0, LON_0, RESIZE_FACTOR, MetersPerPixel()));
    return ok;
}

Projection::LatLon Projection::ProjectPixel(double x, double y) const
{
    // 1. Pixel -> projected-plane meters (affine transform)
    double px = x * RESIZE_FACTOR;
    double py = y * RESIZE_FACTOR;

    double x_geo = TRANSFORM_COEFFS[0] * px + TRANSFORM_COEFFS[1] * py + TRANSFORM_COEFFS[2];
    double y_geo = TRANSFORM_COEFFS[3] * px + TRANSFORM_COEFFS[4] * py + TRANSFORM_COEFFS[5];

    // 2. Inverse general/oblique stereographic (Snyder, spherical form, k0=1)
    double rho = std::sqrt(x_geo * x_geo + y_geo * y_geo);

    double phi0 = LAT_0 * (M_PI / 180.0);
    double lam0 = LON_0 * (M_PI / 180.0);

    double phi, lam;
    if (rho < 1e-9) {
        phi = phi0;
        lam = lam0;
    } else {
        double c = 2.0 * std::atan2(rho, 2.0 * R);
        double cos_c = std::cos(c);
        double sin_c = std::sin(c);

        phi = std::asin(cos_c * std::sin(phi0) + (y_geo * sin_c * std::cos(phi0)) / rho);

        double num = x_geo * sin_c;
        double den = rho * std::cos(phi0) * cos_c - y_geo * std::sin(phi0) * sin_c;
        lam = lam0 + std::atan2(num, den);
    }

    return {phi * (180.0 / M_PI), lam * (180.0 / M_PI), true};
}

std::pair<double, double> Projection::InverseProjectPixel(double lat_deg, double lon_deg) const
{
    double phi = lat_deg * (M_PI / 180.0);
    double lam = lon_deg * (M_PI / 180.0);
    double phi0 = LAT_0 * (M_PI / 180.0);
    double lam0 = LON_0 * (M_PI / 180.0);

    // Forward general/oblique stereographic (Snyder, spherical form, k0=1)
    double cos_c = std::sin(phi0) * std::sin(phi) + std::cos(phi0) * std::cos(phi) * std::cos(lam - lam0);
    double k = 2.0 / (1.0 + cos_c);

    double x_geo = R * k * std::cos(phi) * std::sin(lam - lam0);
    double y_geo = R * k * (std::cos(phi0) * std::sin(phi) - std::sin(phi0) * std::cos(phi) * std::cos(lam - lam0));

    // Invert the affine transform: x_geo = a*px + b*py + c ; y_geo = d*px + e*py + f
    double a = TRANSFORM_COEFFS[0], b = TRANSFORM_COEFFS[1], c = TRANSFORM_COEFFS[2];
    double d = TRANSFORM_COEFFS[3], e = TRANSFORM_COEFFS[4], f = TRANSFORM_COEFFS[5];

    x_geo -= c;
    y_geo -= f;

    double det = a * e - b * d;
    if (std::abs(det) < 1e-9) return {-1e9, -1e9};

    double px = (e * x_geo - b * y_geo) / det;
    double py = (-d * x_geo + a * y_geo) / det;

    return {px / RESIZE_FACTOR, py / RESIZE_FACTOR};
}

Projection::RotMat Projection::ComputeRotation(double phi, double lam) const
{
    // Local North/East direction vectors, via numerical forward
    // differentiation of the forward projection (moving a small step
    // North/East on the sphere -> how much x_geo/y_geo changes). Gives the
    // rotation from geographic (u,v) to projected-plane (x,y) components.
    double phi0 = LAT_0 * (M_PI / 180.0);
    double lam0 = LON_0 * (M_PI / 180.0);

    auto Fwd = [&](double p, double l) -> std::pair<double, double> {
        double cos_c = std::sin(phi0) * std::sin(p) + std::cos(phi0) * std::cos(p) * std::cos(l - lam0);
        double k = 2.0 / (1.0 + cos_c);
        double x = R * k * std::cos(p) * std::sin(l - lam0);
        double y = R * k * (std::cos(phi0) * std::sin(p) - std::sin(phi0) * std::cos(p) * std::cos(l - lam0));
        return {x, y};
    };

    const double delta = 1e-5; // small radian step

    std::pair<double, double> P = Fwd(phi, lam);

    std::pair<double, double> Pn = Fwd(phi + delta, lam);
    double dx_n = Pn.first - P.first;
    double dy_n = Pn.second - P.second;
    double len_n = std::sqrt(dx_n * dx_n + dy_n * dy_n);
    if (len_n < 1e-9) len_n = 1.0;

    std::pair<double, double> Pe = Fwd(phi, lam + delta);
    double dx_e = Pe.first - P.first;
    double dy_e = Pe.second - P.second;
    double len_e = std::sqrt(dx_e * dx_e + dy_e * dy_e);
    if (len_e < 1e-9) len_e = 1.0;

    return {dx_e / len_e, dy_e / len_e, dx_n / len_n, dy_n / len_n};
}

double Projection::ScaleFactor(double lat_rad, double lon_rad) const
{
    // Snyder's point scale factor for the stereographic projection (k0=1):
    // k = 2 / (1 + sin(phi0)*sin(phi) + cos(phi0)*cos(phi)*cos(lam-lam0))
    double phi0 = LAT_0 * (M_PI / 180.0);
    double lam0 = LON_0 * (M_PI / 180.0);
    double cos_c = std::sin(phi0) * std::sin(lat_rad) + std::cos(phi0) * std::cos(lat_rad) * std::cos(lon_rad - lam0);
    return 2.0 / (1.0 + cos_c);
}

double Projection::AreaScaleFactor(double lat_rad, double lon_rad) const
{
    double k = ScaleFactor(lat_rad, lon_rad);
    return k * k;
}

double Projection::MetersPerPixel() const
{
    return std::abs(TRANSFORM_COEFFS[0]) * RESIZE_FACTOR;
}
