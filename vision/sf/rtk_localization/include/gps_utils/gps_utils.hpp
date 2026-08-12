#pragma once

#include <GeographicLib/Geodesic.hpp>
#include <GeographicLib/UTMUPS.hpp>
#include <GeographicLib/LocalCartesian.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/quaternion.hpp>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <rclcpp/rclcpp.hpp>


#include <string>

namespace gps_utils {

struct Waypoint {
  double lat;
  double lon;
  std::string name = "";
};

struct UTMPoint {
  double easting;
  double northing;
  int zone;
  char band;
};

struct LatLon
{
  double lat;
  double lon;
};

struct LatLonAlt {
  double lat;
  double lon;
  double alt;
};


struct ECEF {
  double x, y, z;
};

struct ENU {
  double e, n, u;
};

struct NED {
  double n, e, d;
};

struct AER {
  double azimuth;   // radians
  double elevation; // radians
  double range;     // meters
};




class GpsConverter
{
public:
  GpsConverter();

  /* ===== ENU origin handling ===== */
  void set_enu_origin(double lat, double lon, double alt);
  bool is_origin_set() const { return origin_set_; }

  /* ===== Geodetic helpers ===== */
  // Distance to another location in meters (exact WGS84)
  double distance_lat_lon(double lat1, double lon1,
                          double lat2, double lon2) const;

  // Bearing from (lat1, lon1) to (lat2, lon2), in radians [0, 2π)
  double bearing_lat_lon(double lat1, double lon1,
                         double lat2, double lon2) const;
  
  /* ===== Geodetic ↔ UTM ===== */
  UTMPoint latlon_to_utm(double lat, double lon) const;

  LatLon utm_to_latlon(double easting,
                       double northing,
                       int zone,
                       char hemisphere) const;
  
  /* ===== Geodetic ↔ ECEF ===== */
  ECEF latlon_to_ecef(double lat, double lon, double alt) const;
  LatLonAlt ecef_to_latlon(double x, double y, double z) const;

  /* ===== ECEF ↔ ENU ===== */
  ENU ecef_to_enu(const ECEF &ecef) const;
  ECEF enu_to_ecef(const ENU &enu) const;
  
   /* ===== Geodetic ↔ ENU ===== */
  ENU latlon_to_enu(double lat, double lon, double alt) const;
  LatLonAlt enu_to_latlon(double e, double n, double u) const;

  




  // /* ===== ENU ↔ AER ===== */
  AER enu_to_aer(const ENU &enu) const;
  ENU aer_to_enu(const AER &aer) const;

  /* ===== ENU ↔ NED ===== */
  NED enu_to_ned(const ENU &enu) const;
  ENU ned_to_enu(const NED &ned) const;

  /* ===== NED ↔ AER ===== */
  AER ned_to_aer(const NED &ned) const;
  NED aer_to_ned(const AER &aer) const;

  

  /* ===== ROS helpers ===== */
  geometry_msgs::msg::PoseStamped make_pose(
    double lat,
    double lon,
    const std::string & frame_id) const;
  
  double computeYawFromECEF(
        const ECEF& ecef_left, 
        const ECEF& ecef_right, double offset) const;

  
  geometry_msgs::msg::Pose enuToRosPose(
        const ENU& enu,
        double yaw_enu   // radians, ENU convention
    ) const;
  
  ENU rosToEnuPose(
        const geometry_msgs::msg::Pose& pose,
        double& yaw_enu   // output (radians)
    ) const;

  double enuDistance2D(const ENU& a, const ENU& b);
    


private:
  GeographicLib::Geodesic geod_;  // WGS84 ellipsoid
  GeographicLib::LocalCartesian local_cart_;
  bool origin_set_ = false;
};

}  // namespace gps_utils
