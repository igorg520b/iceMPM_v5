#ifndef PROJECTION_H
#define PROJECTION_H

#include <string>
#include <utility>

#include "rapidjson/document.h"

// Single shared implementation of the map projection used to relate
// reanalysis lat/lon data (CARRA1, GLO12) to pixel/grid coordinates.
//
// This is the general/oblique STEREOGRAPHIC projection tangent at a
// configurable center point (Kane Basin: lat0=80, lon0=-69), matching
// _python_code/image_download/kane_basin_projection.py's
// "+proj=stere +lat_0=.. +lon_0=.. +R=.. +k=1" -- NOT the specialized polar
// stereographic variant (which fixes lat_0 at +/-90 and uses a
// true-scale-latitude parameter instead of a tangent-point scale factor).
//
// Previously this math (and its parameters) was hand-copy-pasted between
// WindInterpolator::ProjectPixel/ComputeRotation and
// CurrentInterpolator::ProjectPixel/ComputeRotation (byte-for-byte
// identical in the case of ProjectPixel). This class is the single place
// both classes -- and anything else that needs to relate pixels to
// lat/lon (e.g. the GLO12 ice-thickness renderer) -- should call into.
struct Projection
{
    // NOTE: deliberately no in-class default member initializers here.
    // SimParams (which owns a Projection member) is copied wholesale into
    // GPU __constant__ memory (cudaMemcpyToSymbol), which requires a
    // trivially-constructible type; NSDMIs would give Projection (and
    // therefore SimParams) a non-trivial default constructor and break
    // that. Left zero-initialized until ParseJson/ParseFile populates them,
    // matching how the original PROJ_* fields on SimParams worked.
    double LAT_0;   // projection center latitude, degrees
    double LON_0;   // projection center longitude, degrees
    static constexpr double R = 6371000.0;  // sphere radius, meters (matches the Python pipeline; not the WGS84 ellipsoid)

    // Affine transform from (possibly resized) pixel coordinates to
    // projected-plane meters: x' = a*px + b*py + c ; y' = d*px + e*py + f
    double TRANSFORM_COEFFS[6];

    // Pixel coordinates are multiplied by this factor before the affine
    // transform is applied -- accounts for the working image being a
    // downsampled version of the raster the affine transform was computed for.
    double RESIZE_FACTOR;

    struct LatLon { double lat_deg, lon_deg; bool valid; };
    struct RotMat { double ex, ey, nx, ny; };  // local east/north unit basis vectors, in projected-plane (x,y) coordinates

    // Parse PROJ_* fields from an already-open rapidjson document/value.
    // Returns true if at least the projection center (LAT_0/LON_0) was found.
    bool ParseJson(const rapidjson::Value& doc);

    // Load PROJ_* fields from a standalone JSON file.
    bool ParseFile(const std::string& fileName);

    // pixel (x,y), in the same raw pixel space TRANSFORM_COEFFS/RESIZE_FACTOR
    // were computed for -> lat/lon (degrees). LatLon::valid is false if the
    // point falls outside the projection's valid domain.
    LatLon ProjectPixel(double x, double y) const;

    // lat/lon (degrees) -> pixel (x,y) in that same raw pixel space (the
    // inverse of ProjectPixel). Does NOT know about any simulation grid
    // sub-region offset/flip -- callers that need grid-local (i,j) apply
    // that themselves on top of this.
    std::pair<double, double> InverseProjectPixel(double lat_deg, double lon_deg) const;

    // Local east/north unit basis vectors (in projected-plane coordinates)
    // at a given lat/lon (radians) -- used to rotate geographic (u,v)
    // vectors (e.g. wind, current) into the grid's local frame.
    RotMat ComputeRotation(double lat_rad, double lon_rad) const;

    // Point scale factor k (linear/length distortion relative to the
    // tangent point), exact closed form (Snyder's stereographic formula),
    // not numerically differentiated.
    double ScaleFactor(double lat_rad, double lon_rad) const;

    // Area distortion factor = k^2 (stereographic is conformal: the same
    // scale factor applies in every direction, so area scale is simply the
    // square of the linear scale factor). This is what the GPU kernels
    // should multiply cell-area-derived quantities by to correct for
    // projection distortion away from the tangent point.
    double AreaScaleFactor(double lat_rad, double lon_rad) const;

    // Meters per pixel in the raw pixel space (i.e. the simulation's
    // cellsize), derived directly from the affine transform instead of a
    // separately specified physical dimension.
    double MetersPerPixel() const;
};

#endif // PROJECTION_H
