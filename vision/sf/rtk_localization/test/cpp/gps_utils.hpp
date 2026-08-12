// gps_utils.hpp
#pragma once

#include <GeographicLib/Geodesic.hpp>
#include <GeographicLib/UTMUPS.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <iomanip>
#include <cmath>

using namespace GeographicLib;

class Location {
public:
    double lat, lon;
    double altitude = 0.0;

    Location(double latitude, double longitude, double alt = 0.0)
        : lat(latitude), lon(longitude), altitude(alt) {}

    // Distance to another location in meters (exact WGS84)
    double distance_to(const Location& other) const {
        const Geodesic& geod = Geodesic::WGS84();
        double dist;
        geod.Inverse(lat, lon, other.lat, other.lon, dist);
        return dist;  // meters
    }

    // Bearing from this → other in radians [0, 2π], 0 = North, clockwise
    double bearing_to(const Location& other) const {
        const Geodesic& geod = Geodesic::WGS84();
        double azi;
        double dist;
        geod.Inverse(lat, lon, other.lat, other.lon, dist, azi);
        double bearing_deg = std::fmod(azi + 360.0, 360.0);
        return bearing_deg * M_PI / 180.0;  // to radians
    }

    // Convert to UTM
    struct UTM {
        double easting, northing;
        int zone;
        char band;
        std::string zone_str() const { return std::to_string(zone) + band; }

        double x() const { return easting; }
        double y() const { return northing; }

        double distance_to(const UTM& other) const {
            if (zone != other.zone || band != other.band)
                throw std::runtime_error("UTM zones differ!");
            double dx = easting - other.easting;
            double dy = northing - other.northing;
            return std::hypot(dx, dy);
        }

        friend std::ostream& operator<<(std::ostream& os, const UTM& u) {
            os << "UTM(easting=" << std::fixed << std::setprecision(3) << u.easting
               << ", northing=" << u.northing
               << ", zone=" << u.zone << u.band << ")";
            return os;
        }
    };

    UTM to_utm() const {
        int zone;
        bool northp;
        double easting, northing;
        UTMUPS::Forward(lat, lon, zone, northp, easting, northing);
        return {easting, northing, zone, northp ? 'N' : 'S'};
    }

    friend std::ostream& operator<<(std::ostream& os, const Location& loc) {
        os << "Location(lat=" << std::fixed << std::setprecision(9) << loc.lat
           << ", lon=" << loc.lon;
        if (loc.altitude != 0.0)
            os << ", alt=" << loc.altitude;
        os << ")";
        return os;
    }
};

// Parse your string exactly like Python version
std::vector<Location> parse_locations(const std::string& text) {
    std::vector<Location> locations;
    std::stringstream ss(text);
    double lat, lon;

    while (ss >> lat >> lon) {
        locations.emplace_back(lat, lon);
    }
    return locations;
}