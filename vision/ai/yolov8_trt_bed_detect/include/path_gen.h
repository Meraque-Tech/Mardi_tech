#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <sl/Camera.hpp>

#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/float32.hpp>
#include <std_msgs/msg/float32_multi_array.hpp>

#include <tf2_ros/static_transform_broadcaster.h>
#include "tf2/LinearMath/Quaternion.h"
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <tf2/exceptions.h>
#include <geometry_msgs/msg/pose_stamped.hpp>

#include <visualization_msgs/msg/marker_array.hpp>
#include <visualization_msgs/msg/marker.hpp>

#include <algorithm>
#include <tuple>
#include <numeric>
#include <nav_msgs/msg/path.hpp>
#include "std_msgs/msg/u_int8.hpp"


using static_tf2 = std::shared_ptr<tf2_ros::StaticTransformBroadcaster>;
using viz_msg= visualization_msgs::msg::MarkerArray;
using viz_msg_f = rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr;

double Cosin(double a, double b, double c);
float euclidean_distance_onearg(const sl::float3& point1);
float euclidean_distance_twoarg(const sl::float3& point1, const sl::float3& point2);
double gamma_sign_correction(float gamma, float x);
std::tuple<float, float> find_global_PQR(const sl::float3& img_frame, const sl::float3& bed_PQR);
void send_static_target_tf( static_tf2 static_target_tf,
                            rclcpp::Time now, 
                            std::string parent_frame, 
                            std::string child_frame, 
                            double x, 
                            double y, 
                            double z, 
                            double th_x, 
                            double th_y, 
                            double th_z);


std::tuple<float, float> find_HV_slop(float x1, float y1, float x2, float y2);
std::tuple<float, float> WP_creation(float gth_m, float gpx, float gpy, int type_wp, float k);
std::tuple<sl::float3, sl::float3, sl::float3, float, float, float, float> distance_cluster_lrf(cv::Mat &img, sl::Mat &point_cloud, sl::float4 &plane_eq, auto &it);
geometry_msgs::msg::PoseStamped tf2_pose(rclcpp::Time now, geometry_msgs::msg::TransformStamped &tf_msg);
void updateWpNavigationMarkers(viz_msg_f wp_navigation_markers_pub_,std::vector<geometry_msgs::msg::PoseStamped> &poses_);
void resetUniqueId();
int getUniqueId();
geometry_msgs::msg::TransformStamped tf_sub(std::string parent_frame, std::string child_frame);

std::string camera_frame ="camera_link";
std::string bed_frameP = "bed_linkP";
std::string bed_framePP = "bed_linkPP";
std::string bed_frameM = "bed_linkM";
std::string bed_frameR = "bed_linkR";
std::string bed_frameRR = "bed_linkRR";
std::string bed_framePP_v = "bed_linkPP_v";
std::string bed_frameRR_v = "bed_linkRR_v";
std::string bed_framePP_vb = "bed_linkPP_vb";
std::string bed_framePP_vf = "bed_linkPP_vf";
std::string bed_frameRR_vb = "bed_linkRR_vb";
std::string bed_frameRR_vf = "bed_linkRR_vf";
std::string bed_frameRR_Mb = "bed_linkRR_Mb";
std::string map_frame = "map";

float k_h = 0.5;
float k_v = 0.5;

std::tuple<std::vector<float>, std::vector<float>> path_generator(float xi, float yi, float xf, float yf);
void startNavToPose(geometry_msgs::msg::PoseStamped pose);
// rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr LH_pose_stamped_pub;
// rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr RH_pose_stamped_pub;
// rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr FH_pose_stamped_pub;

rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr left_pose_array_pub;
rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr right_pose_array_pub;
rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr font_pose_array_pub;
rclcpp::Publisher<std_msgs::msg::UInt8>::SharedPtr bed_detection_nav2_call_pub;
bool start_bed_detection_ = 0;
bool bed_detection_fb_ = 1;
bool start_bed_detection_plane_algo_ = 0;

// std::shared_ptr<lytbot2_msg_srv::srv::BedDetection::Response> bed_detection_response;