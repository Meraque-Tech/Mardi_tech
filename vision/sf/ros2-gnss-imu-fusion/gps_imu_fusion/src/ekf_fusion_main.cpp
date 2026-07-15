#include "gps_imu_fusion/ekf_fusion_node.hpp"
#include <rclcpp/rclcpp.hpp>

int main(int argc, char* argv[]) {
  rclcpp::init(argc, argv);
  rclcpp::NodeOptions options;
  options.automatically_declare_parameters_from_overrides(true);
  auto node = std::make_shared<gps_imu_fusion::EkfFusionNode>(options);
  rclcpp::spin(node);
  rclcpp::shutdown();
  
  return 0;
}