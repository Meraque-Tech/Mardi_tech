#include "gps_utils.hpp"

int main() {
    const std::string loc_str = R"(
        49.899600976838286 8.899341648789813
        49.8998894578428   8.898535985040668
        49.90096619250414  8.899532919301544
        49.90066376911161  8.900237165860988
    )";

    auto locations = parse_locations(loc_str);

    std::cout << "=== Lat/Lon ===\n";
    for (const auto& loc : locations)
        std::cout << loc << "\n";

    std::cout << "\n=== UTM ===\n";
    std::vector<Location::UTM> utms;
    for (const auto& loc : locations) {
        auto u = loc.to_utm();
        utms.push_back(u);
        std::cout << u << "\n";
    }

    std::cout << "\n=== Distances & Bearings ===\n";
    for (size_t i = 1; i < locations.size(); ++i) {
        double dist = locations[0].distance_to(locations[i]);
        double bearing_deg = locations[0].bearing_to(locations[i]) * 180.0 / M_PI;
        std::cout << "Point " << i << " → "
                  << "dist: " << dist << " m, "
                  << "bearing: " << bearing_deg << "°\n";
    }

    return 0;
}