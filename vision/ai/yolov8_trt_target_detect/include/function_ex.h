#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <sl/Camera.hpp>

#include <rclcpp/rclcpp.hpp>
#include <rclcpp/time.hpp>
#include <rclcpp_action/rclcpp_action.hpp>
#include <std_msgs/msg/float32.hpp>
#include <std_msgs/msg/float32_multi_array.hpp>
#include "std_msgs/msg/u_int8.hpp"
#include <algorithm>
#include <tuple>
#include <numeric>

#include <tf2_ros/static_transform_broadcaster.h>
#include "tf2/LinearMath/Quaternion.h"
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <tf2/exceptions.h>

#include <geometry_msgs/msg/pose_stamped.hpp>

#include <visualization_msgs/msg/marker_array.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <tf2/exceptions.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>

#include <nav2_msgs/action/navigate_to_pose.hpp>
#include <nav_msgs/msg/path.hpp>

// #include <nav2_util/geometry_utils.hpp>

// using nav2_util::geometry_utils::orientationAroundZAxis;
// #include <pcl/point_cloud.h>
// #include <pcl/point_types.h>
// #include <pcl/kdtree/kdtree.h>
// #include <pcl/kdtree/kdtree_flann.h>
// #include <pcl/segmentation/extract_clusters.h>

using float32_multi_array = std_msgs::msg::Float32MultiArray;
using float32_multi_array_f = rclcpp::Publisher<float32_multi_array>::SharedPtr;
using static_tf2 = std::shared_ptr<tf2_ros::StaticTransformBroadcaster>;
using viz_msg = visualization_msgs::msg::MarkerArray;
using viz_msg_f = rclcpp::Publisher<viz_msg>::SharedPtr;

// using tf_buffer_ = std::unique_ptr<tf2_ros::Buffer>;
// std::shared_ptr<tf2_ros::TransformListener> transform_listener_{nullptr};

struct Point3D {
    double x, y, z;
    Point3D(double _x, double _y, double _z) : x(_x), y(_y), z(_z) {}
};

std::tuple<float, size_t, float, size_t> minmax_idx(const std::vector<float>& v);
Point3D crossProduct(const Point3D& a, const Point3D& b);
void bed_cluster_find(cv::Mat &img, sl::Mat &point_cloud, sl::float4 &plane_eq, cv::Rect &r);
double calculatePolygonArea3D(const std::vector<Point3D>& vertices);
void drawPolygon(cv::Mat& image, const std::vector<cv::Point>& points, const cv::Scalar& color);
sl::float3 pix_pcl(int pix_x, int pix_y, sl::Mat &point_cloud, cv::Mat &image);
void space_finder(cv::Mat &img, sl::Mat &point_cloud, sl::float4 &plane_eq, cv::Rect &r, cv::Point &topLeft, cv::Point &bottomRight, cv::Scalar &top_clr, cv::Scalar &floor_clr);
cv::Point new_boundary(cv::Mat &img, int X, int Y);
bool boundary_chek(cv::Mat &img, int X, int Y);
void plane_detectio_algo(std::vector<cv::Mat> &img_batch, 
                std::vector<std::vector<Detection>> &res_batch, 
                sl::Mat &point_cloud, sl::float4 &plane_eq, 
                sl::CameraParameters& zedCameraParams,
                float32_multi_array_f front_pose_pub,
                float32_multi_array_f left_pose_pub,
                float32_multi_array_f right_pose_pub,
                float32_multi_array_f img_pose_pub,
                static_tf2 static_target_tf,
                viz_msg_f viz_f,
                rclcpp::Time now);

void sofa_plane_detectio_algo(std::vector<cv::Mat> &img_batch, 
                std::vector<std::vector<Detection>> &res_batch, 
                sl::Mat &point_cloud, sl::float4 &plane_eq, 
                sl::CameraParameters& zedCameraParams,
                float32_multi_array_f front_pose_pub,
                float32_multi_array_f left_pose_pub,
                float32_multi_array_f right_pose_pub,
                float32_multi_array_f img_pose_pub,
                static_tf2 static_target_tf,
                viz_msg_f viz_f,
                rclcpp::Time now);
                
int getOCVtype(sl::MAT_TYPE type);
cv::Mat slMat2cvMat(sl::Mat& input);
float dis_from_pane(sl::float4 &plane_eq,sl::float3 &pcl);
cv::Point projectToImage(const sl::float3& point_3d, const sl::CameraParameters& zedCameraParams, const cv::Size& imageSize);
void drawBoundingBox(const std::vector<sl::float3>& boundingBoxCorners, const sl::CameraParameters& zedCameraParams, cv::Mat& img);
sl::float3 draw_grid(cv::Mat &img, sl::Mat &point_cloud, sl::float4 &plane_eq, cv::Rect &r, cv::Point &top_left, cv::Point &top_right, cv::Point &bottom_left, cv::Point &bottom_right);
float calculateAverage(const std::vector<float>& points);
cv::Point calculateAverage_pxl(const std::vector<cv::Point>& points);
cv::Point calculateAverageWithStdDev_pxl(const std::vector<cv::Point>& points, double stdDevThreshold);
float euclidean_distance(const sl::float3& point1, const sl::float3& point2);

sl::float4 findPlaneCoefficients(sl::float3 pix_pcl_P,
                                 sl::float3 pix_pcl_Q,
                                 sl::float3 pix_pcl_R);

sl::float3 min_pix(cv::Mat &img, sl::Mat &point_cloud, 
            sl::float4 &plane_eq, 
            cv::Rect &r, 
            cv::Point &topLeft, 
            cv::Point &bottomRight, 
            int I_xx);
                        
sl::float3 find_foot_point_on_plane(const sl::float4 &plane_eq, const sl::float3 &pcl);
sl::float3 pix_pcl_cluster(cv::Mat &img, sl::Mat &point_cloud, sl::float4 &plane_eq, cv::Rect &r, int x, int y);
std::tuple<sl::float3, cv::Point> min_pix_V(cv::Mat &img, sl::Mat &point_cloud, sl::float4 &plane_eq, cv::Rect &r, cv::Point &topLeft, cv::Point &topRight, int I_yy, sl::float3 &riff_cluster);
float min_pix_H(cv::Mat &img, sl::Mat &point_cloud, sl::float4 &plane_eq, cv::Rect &r, cv::Point &topLeft, cv::Point &topRight, sl::float3 &riff_cluster);

// std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
// std::shared_ptr<tf2_ros::TransformListener> transform_listener_{nullptr};
std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
std::shared_ptr<tf2_ros::TransformListener> transform_listener_{nullptr};
geometry_msgs::msg::TransformStamped tf_sub(std::string parent_frame, std::string child_frame);

nav2_msgs::action::NavigateToPose::Goal navigation_goal_;
using NavigationGoalHandle = rclcpp_action::ClientGoalHandle<nav2_msgs::action::NavigateToPose>;
NavigationGoalHandle::SharedPtr navigation_goal_handle_;

rclcpp_action::Client<nav2_msgs::action::NavigateToPose>::SharedPtr navigation_action_client_;
// navigation_goal_ = nav2_msgs::action::NavigateToPose::Goal();

rclcpp::Node::SharedPtr node;
std::chrono::milliseconds server_timeout_;
// std::shared_ptr<rclcpp::Node> node;
double conf_;
// std::string conf_score= "";
float conf_score_value = 0.0; 
float distance_thresh = 1.0;
float font_dis = 2.0;
float dis_img = 1.0;
