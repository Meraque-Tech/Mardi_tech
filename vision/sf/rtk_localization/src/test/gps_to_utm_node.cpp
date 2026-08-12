#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/nav_sat_fix.hpp>
#include <gps_utils/gps_utils.hpp>
#include <angle_utils/angle_utils.hpp>


class GnssFixListener : public rclcpp::Node
{
public:
  GnssFixListener() : Node("gnss_fix_listener")
  {
    subscription_ = this->create_subscription<sensor_msgs::msg::NavSatFix>(
      "/gps/fix",
      10,
      std::bind(&GnssFixListener::on_fix, this, std::placeholders::_1));

    RCLCPP_INFO(this->get_logger(), "GNSS Fix Listener started. Subscribed to /gps/fix");
  }

private:
  void on_fix(const sensor_msgs::msg::NavSatFix::SharedPtr msg)
  {
    // RCLCPP_INFO(
    //   this->get_logger(),
    //   "GNSS Fix -> Latitude: %.8f, Longitude: %.8f, Altitude: %.2f",
    //   msg->latitude,
    //   msg->longitude,
    //   msg->altitude);
    const double lat = msg->latitude;
    const double lon = msg->longitude;

    auto utm = converter_.latlon_to_utm(lat, lon);

    double ref_lat = 49.90996619;
    double ref_lon = 8.89953292;

    double dist = converter_.distance_lat_lon(ref_lat, ref_lon, lat, lon);

     RCLCPP_INFO(
      this->get_logger(),
      "GNSS Fix -> Lat: %.8f, Lon: %.8f, Alt: %.2f | "
      "UTM -> E: %.3f, N: %.3f, Zone: %d%c | "
      "Distance to reference: %.2f m",
      lat,
      lon,
      msg->altitude,
      utm.easting,
      utm.northing,
      utm.zone,
      utm.band,
      dist);

    double bearing_rad = converter_.bearing_lat_lon(ref_lat, ref_lon, lat, lon);
    double bearing_deg = bearing_rad * 180.0 / M_PI;

    RCLCPP_INFO(rclcpp::get_logger("bearing_test"),
            "Bearing = %.3f deg (%.6f rad)", bearing_deg, bearing_rad);

  }

  rclcpp::Subscription<sensor_msgs::msg::NavSatFix>::SharedPtr subscription_;
  gps_utils::GpsConverter converter_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<GnssFixListener>());
  rclcpp::shutdown();
  return 0;
}
