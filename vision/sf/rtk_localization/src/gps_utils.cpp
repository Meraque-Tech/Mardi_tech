#include "gps_utils/gps_utils.hpp"

#include <cmath>

namespace gps_utils
{

GpsConverter::GpsConverter()
: geod_(GeographicLib::Geodesic::WGS84())
{
}

/* ===== ENU origin handling ===== */
void GpsConverter::set_enu_origin(double lat, double lon, double alt)
{
  local_cart_.Reset(lat, lon, alt);
  origin_set_ = true;
}

/* ===== Geodetic helpers ===== */
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



/* ===== Geodetic ↔ UTM ===== */
UTMPoint GpsConverter::latlon_to_utm(double lat, double lon) const
{
  int zone;
  bool northp;
  double x, y;

  GeographicLib::UTMUPS::Forward(lat, lon, zone, northp, x, y);
  return UTMPoint{ x, y, zone, northp ? 'N' : 'S' };
}

LatLon GpsConverter::utm_to_latlon(double easting,
                                  double northing,
                                  int zone,
                                  char hemisphere) const
{
  double lat, lon;

  bool northp = (hemisphere == 'N');

  GeographicLib::UTMUPS::Reverse(
    zone,
    northp,
    easting,
    northing,
    lat,
    lon);

  return LatLon{lat, lon};
}



/* ===== Geodetic ↔ ENU ===== */
ENU GpsConverter::latlon_to_enu(double lat, double lon, double alt) const
{
  if (!origin_set_) throw std::runtime_error("ENU origin not set");

  double e, n, u;
  local_cart_.Forward(lat, lon, alt, e, n, u);
  return {e, n, u};
}

LatLonAlt GpsConverter::enu_to_latlon(double e, double n, double u) const
{
  if (!origin_set_) throw std::runtime_error("ENU origin not set");

  double lat, lon, alt;
  local_cart_.Reverse(e, n, u, lat, lon, alt);
  return {lat, lon, alt};
}



/* ===== Geodetic ↔ ECEF ===== */
ECEF GpsConverter::latlon_to_ecef(double lat, double lon, double alt) const
{
  double x, y, z;
  GeographicLib::Geocentric::WGS84().Forward(lat, lon, alt, x, y, z);
  return {x, y, z};
}


LatLonAlt GpsConverter::ecef_to_latlon(double x, double y, double z) const
{
  double lat, lon, alt;
  GeographicLib::Geocentric::WGS84().Reverse(x, y, z, lat, lon, alt);
  return {lat, lon, alt};
}


/* ===== ECEF ↔ ENU ===== */
ENU GpsConverter::ecef_to_enu(const ECEF &ecef) const
{
  auto lla = ecef_to_latlon(ecef.x, ecef.y, ecef.z);
  return latlon_to_enu(lla.lat, lla.lon, lla.alt);
}

ECEF GpsConverter::enu_to_ecef(const ENU &enu) const
{
  auto lla = enu_to_latlon(enu.e, enu.n, enu.u);
  return latlon_to_ecef(lla.lat, lla.lon, lla.alt);
}


 // /* ===== ENU ↔ AER ===== */
AER GpsConverter::enu_to_aer(const ENU &enu) const
{
  AER aer;
  aer.range = std::sqrt(enu.e*enu.e + enu.n*enu.n + enu.u*enu.u);
  aer.azimuth = std::atan2(enu.e, enu.n);  // ENU convention
  aer.elevation = std::atan2(enu.u,
                     std::sqrt(enu.e*enu.e + enu.n*enu.n));
  return aer;
}

ENU GpsConverter::aer_to_enu(const AER &aer) const
{
  ENU enu;
  enu.e = aer.range * std::cos(aer.elevation) * std::sin(aer.azimuth);
  enu.n = aer.range * std::cos(aer.elevation) * std::cos(aer.azimuth);
  enu.u = aer.range * std::sin(aer.elevation);
  return enu;
}

/* ===== ENU ↔ NED ===== */
NED GpsConverter::enu_to_ned(const ENU &enu) const
{
  return {enu.n, enu.e, -enu.u};
}

ENU GpsConverter::ned_to_enu(const NED &ned) const
{
  return {ned.e, ned.n, -ned.d};
}

/* ===== NED ↔ AER ===== */
AER GpsConverter::ned_to_aer(const NED &ned) const
{
  double r = std::sqrt(ned.n*ned.n + ned.e*ned.e + ned.d*ned.d);
  return {
    std::atan2(ned.e, ned.n),
    std::atan2(-ned.d, std::sqrt(ned.n*ned.n + ned.e*ned.e)),
    r
  };
}

NED GpsConverter::aer_to_ned(const AER &aer) const
{
  return {
    aer.range * std::cos(aer.elevation) * std::cos(aer.azimuth),
    aer.range * std::cos(aer.elevation) * std::sin(aer.azimuth),
   -aer.range * std::sin(aer.elevation)
  };
}



double GpsConverter::computeYawFromECEF(
        const ECEF& ecef_left, 
        const ECEF& ecef_right, double offset) const
{
    // Convert both antennas to ENU
    ENU left_enu  = ecef_to_enu(ecef_left);
    ENU right_enu = ecef_to_enu(ecef_right);

    // Baseline vector in ENU
    double dE = right_enu.e - left_enu.e;
    double dN = right_enu.n - left_enu.n;

    
    // std::cout << std::fixed << std::setprecision(6)
    //       << "RTK baseline ENU -> dE: " << dE
    //       << " m, dN: " << dN << " m" << std::endl;

    

    // ROS yaw: X-forward, Y-left, Z-up
    // Heading is measured from North, hence +pi/2 shift
    // double yaw = std::atan2(dN, dE) + M_PI / 2.0;
    double yaw = std::atan2(dN, dE) + offset;

    // ROS yaw = atan2(ROS_Y, ROS_X) = atan2(-dE, dN)
    // double yaw = std::atan2(-dE, dN);

    // yaw = yaw - M_PI_2;

    // Normalize to [-pi, pi]
    yaw = std::atan2(std::sin(yaw), std::cos(yaw));

    return yaw;
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

geometry_msgs::msg::Pose GpsConverter::enuToRosPose(
    const ENU& enu,
    double yaw_enu
) const
{
    geometry_msgs::msg::Pose pose;

    // Position (ENU → ROS)
    pose.position.x = enu.n;   // X = North
    pose.position.y = -enu.e;  // Y = -East
    pose.position.z = enu.u;   // Z = Up

    // Orientation
    tf2::Quaternion q;
    q.setRPY(0.0, 0.0, yaw_enu);

    pose.orientation.x = q.x();
    pose.orientation.y = q.y();
    pose.orientation.z = q.z();
    pose.orientation.w = q.w();

    return pose;
}


ENU GpsConverter::rosToEnuPose(
    const geometry_msgs::msg::Pose& pose,
    double& yaw_enu
) const
{
    ENU enu;

    // Position (ROS → ENU)
    enu.e = -pose.position.y;
    enu.n =  pose.position.x;
    enu.u =  pose.position.z;

    // Orientation
    tf2::Quaternion q(
        pose.orientation.x,
        pose.orientation.y,
        pose.orientation.z,
        pose.orientation.w
    );

    double roll, pitch, yaw;
    tf2::Matrix3x3(q).getRPY(roll, pitch, yaw);
    yaw_enu = yaw;

    return enu;
}

double GpsConverter::enuDistance2D(const ENU& a, const ENU& b)
{
    const double dE = a.e - b.e;
    const double dN = a.n - b.n;
    return std::sqrt(dE * dE + dN * dN);
}



}  // namespace gps_utils
