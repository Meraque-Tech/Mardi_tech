#include "gps_utils/gps_utils.hpp"

#include <cmath>

namespace gps_utils
{

GpsConverter::GpsConverter()
: geod_(GeographicLib::Geodesic::WGS84())
{
}

// Distance to another location in meters (exact WGS84)
double GpsConverter::distance_lat_lon(double lat1, double lon1,
                                      double lat2, double lon2) const
{
  double dist;
  double azi1, azi2;  // unused but required by API if you choose that overload

  geod_.Inverse(lat1, lon1, lat2, lon2, dist, azi1, azi2);
  return dist;
}

double GpsConverter::bearing_lat_lon(double lat1, double lon1,
                                     double lat2, double lon2) const
{
  double dist;
  double azi1, azi2;

  // azi1 = forward azimuth (deg) from point 1 to point 2
  geod_.Inverse(lat1, lon1, lat2, lon2, dist, azi1, azi2);

  double azi_norm = std::fmod(azi1 + 360.0, 360.0);  // [0, 360)
  return azi_norm * M_PI / 180.0;                    // radians
}

UTMPoint GpsConverter::latlon_to_utm(double lat, double lon) const
{
  int zone;
  bool northp;
  double x, y;

  GeographicLib::UTMUPS::Forward(lat, lon, zone, northp, x, y);
  return UTMPoint{ x, y, zone, northp ? 'N' : 'S' };
}

geometry_msgs::msg::PoseStamped GpsConverter::make_pose(
  double lat,
  double lon,
  const std::string & frame_id) const
{
  geometry_msgs::msg::PoseStamped pose;

  pose.header.frame_id = frame_id;
  pose.header.stamp = rclcpp::Clock().now();

  auto utm = latlon_to_utm(lat, lon);

  pose.pose.position.x = utm.easting;
  pose.pose.position.y = utm.northing;
  pose.pose.position.z = 0.0;

  pose.pose.orientation.x = 0.0;
  pose.pose.orientation.y = 0.0;
  pose.pose.orientation.z = 0.0;
  pose.pose.orientation.w = 1.0;

  return pose;
}

}  // namespace gps_utils
