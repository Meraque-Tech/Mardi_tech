#include "ros_utils.h"


std::tuple<float, float> AlgoUtils::WP_creation(float gth_m, float gpx, float gpy, int type_wp, float k)
{
    double gpx_f, gpy_f;

    if(type_wp == 1)
    {
        gpx_f = gpx + k*cos(gth_m);
        gpy_f = gpy + k*sin(gth_m);
    }
    if(type_wp == 2)
    {
        gpx_f = gpx + k*cos(gth_m);
        gpy_f = gpy - k*sin(gth_m);
    }
    if(type_wp == 3)
    {
        gpx_f = gpx - k*cos(gth_m);
        gpy_f = gpy + k*sin(gth_m);
    }
    if(type_wp == 4)
    {
        gpx_f = gpx - k*cos(gth_m);
        gpy_f = gpy - k*sin(gth_m);
    }
    return std::make_tuple(gpx_f, gpy_f);

}

double AlgoUtils::Cosin(double a, double b, double c)
{
  return acos(((a*a)+(b*b)-(c*c))/(2*a*b));
}

double AlgoUtils::gamma_sign_correction(float gamma, float x)
{
    if (x < 0)
    {
        gamma = -abs(gamma);
    }

    if (x > 0)
    {
        gamma = abs(gamma);
    }

    return gamma;
}


std::tuple<cv::Point, size_t, cv::Point, size_t> AlgoUtils::minmax_idx_pix(const std::vector<cv::Point>& v) {
    if (v.empty()) {
        return {}; // Return an empty tuple for an empty input
    }

    auto minmax_pair = std::minmax_element(v.begin(), v.end(),
        [](const cv::Point& a, const cv::Point& b) {
            return (a.x + a.y) < (b.x + b.y);
        }
    );

    cv::Point min_val = *minmax_pair.first;
    size_t min_index = static_cast<size_t>(std::distance(v.begin(), minmax_pair.first));

    cv::Point max_val = *minmax_pair.second;
    size_t max_index = static_cast<size_t>(std::distance(v.begin(), minmax_pair.second));

    return std::make_tuple(min_val, min_index, max_val, max_index);
}

sl::float3 AlgoUtils::tf_sub_to_eular_angles(std::string parent_frame, std::string child_frame)
{
    sl::float3 rpy;
    geometry_msgs::msg::TransformStamped tf_sub_l = tf_sub(parent_frame, child_frame);
    tf2::Quaternion q(tf_sub_l.transform.rotation.x, tf_sub_l.transform.rotation.y, tf_sub_l.transform.rotation.z, tf_sub_l.transform.rotation.w);
    tf2::Matrix3x3 m(q);
    double roll, pitch, yaw;
    m.getRPY(roll, pitch, yaw);


    rpy.x = roll;
    rpy.y = pitch;
    rpy.z = yaw;
    return rpy;
}


std::tuple<float, float> AlgoUtils::find_HV_slop(float x1, float y1, float x2, float y2)
{
    float gth_m_h = atan2((y2 - y1), (x2 - x1));
    float gth_m_v = atan2(-(x2 - x1), (y2 - y1));
    return std::make_tuple(gth_m_h, gth_m_v);
}

float AlgoUtils::euclidean_distance_onearg(const sl::float3& point1) 
{
    float dx = point1.x;
    float dy = point1.y;
    float dz = point1.z;
    return std::sqrt(dx*dx + dy*dy + dz*dz);
}

float AlgoUtils::euclidean_distance_twoarg(const sl::float3& point1, const sl::float3& point2) 
{
    float dx = point2.x - point1.x;
    float dy = point2.y - point1.y;
    float dz = point2.z - point1.z;
    return std::sqrt(dx*dx + dy*dy + dz*dz);
}

int AlgoUtils::euclidean_distance_twoarg_pix(const cv::Point &point1, const cv::Point &point2) 
{
    float dx = point2.x - point1.x;
    float dy = point2.y - point1.y;
    return std::sqrt(dx * dx + dy * dy);
}

sl::float3 AlgoUtils::midpoint_of_rectangle(const std::vector<sl::float3>& vertices) {
    sl::float3 midpoint;
    int numVertices = vertices.size();

    // Initialize the midpoint to (0, 0, 0)
    midpoint.x = 0.0f;
    midpoint.y = 0.0f;
    midpoint.z = 0.0f;

    // Sum up the coordinates of all vertices
    for (int i = 0; i < numVertices; i++) {
        midpoint.x += vertices[i].x;
        midpoint.y += vertices[i].y;
        midpoint.z += vertices[i].z;
    }

    // Calculate the average by dividing by the number of vertices
    midpoint.x /= numVertices;
    midpoint.y /= numVertices;
    midpoint.z /= numVertices;

    return midpoint;
}

float AlgoUtils::dis_from_pane(sl::float4 &plane_eq,sl::float3 &pcl)
{
    // Extract plane parameters
    float distance;

    float a = plane_eq.x;
    float b = plane_eq.y;
    float c = plane_eq.z;
    float d = plane_eq.w;
    float X = pcl.x;
    float Y = pcl.y;
    float Z = pcl.z;
    if (a==0 && b==0 && c== 0)
    {
        distance = 0;
        return 0;
    }
    else
    {
        distance = fabs(a * X + b * Y + c * Z + d) / sqrt(a * a + b * b + c * c); // distance notmal vector
        return distance;
    }
    
}

sl::float4 AlgoUtils::findPlaneCoefficients(sl::float3 pix_pcl_P, sl::float3 pix_pcl_Q, sl::float3 pix_pcl_R)
{
    sl::float4 coff;
    float x1 = pix_pcl_P.x;
    float y1 = pix_pcl_P.y;
    float z1 = pix_pcl_P.z;
    float x2 = pix_pcl_Q.x;
    float y2 = pix_pcl_Q.y;
    float z2 = pix_pcl_Q.z;
    float x3 = pix_pcl_R.x;
    float y3 = pix_pcl_R.y;
    float z3 = pix_pcl_R.z;

    float a1 = x2 - x1;
    float b1 = y2 - y1;
    float c1 = z2 - z1;
    float a2 = x3 - x1;
    float b2 = y3 - y1;
    float c2 = z3 - z1;
    float a = b1 * c2 - b2 * c1;
    float b = a2 * c1 - a1 * c2;
    float c = a1 * b2 - b1 * a2;
    float d = (-a * x1 - b * y1 - c * z1);

    // std::cout << std::fixed;
    // std::cout << std::setprecision(2);
    // std::cout << "equation of plane is " << a << " x + " << b
    //           << " y + " << c << " z + " << d << std::endl;

    coff.x = a;
    coff.y = b;
    coff.z = c;
    coff.w = d;
    return coff;
}

sl::float3 AlgoUtils::find_foot_point_on_plane(const sl::float4 &plane_eq, const sl::float3 &pcl)
{
    sl::float3 point;
    
    // Extract plane parameters
    float a = plane_eq.x;
    float b = plane_eq.y;
    float c = plane_eq.z;
    float d = plane_eq.w;
    
    float X = pcl.x;
    float Y = pcl.y;
    float Z = pcl.z;

    float k = (-a * X - b * Y - c * Z - d) / (float)(a * a + b * b + c * c);
    float X0 = a * k + X;
    float Y0 = b * k + Y;
    float Z0 = c * k + Z;

    point.x = X0;
    point.y = Y0;
    point.z = Z0;

    return point;
}

std::tuple<bool, sl::float4> AlgoUtils::find_plane_eq(sl::Plane &plane, sl::Pose &pose, sl::Camera &zed)
{  
    sl::POSITIONAL_TRACKING_STATE tracking_state = zed.getPosition(pose);
    if (tracking_state == sl::POSITIONAL_TRACKING_STATE::OK) 
    {
        sl::Transform resetTrackingFloorFrame;
        sl::ERROR_CODE plane_err = zed.findFloorPlane(plane, resetTrackingFloorFrame);

        if (plane_err != sl::ERROR_CODE::SUCCESS)
        {
            std::cout << "No plane found" << std::endl;
            sl::float4 k;
            k.x = 0.0;
            k.y = 0.0;
            k.z = 0.0;
            k.w = 0.0;
            return std::make_tuple(0, k);
        }

        if(plane_err == sl::ERROR_CODE::SUCCESS)
        {
            sl::float4 plane_eq = plane.getPlaneEquation();
            return std::make_tuple(1, plane_eq);
        }
    }
}

std::vector<sl::float3> AlgoUtils::inner_2d_points_gen(sl::float3 &init_coordinates, sl::float3 &final_coordinates, int NOP)
{
  sl::float3 init_coordinates_ = init_coordinates;
  sl::float3 final_coordinates_ = final_coordinates;
  std::vector<sl::float3> inner_2d_points;
  inner_2d_points.push_back(init_coordinates);

  for( int j = 0; j < NOP; j+=1)
  {
    sl::float3 inner_2d_point;
    inner_2d_point.x = (j*final_coordinates_.x + (NOP-j)*init_coordinates_.x)/(NOP);
    inner_2d_point.y = (j*final_coordinates_.y + (NOP-j)*init_coordinates_.y)/(NOP);
    inner_2d_point.z = 0.0;
    inner_2d_points.push_back(inner_2d_point); 
  }
  inner_2d_points.push_back(final_coordinates);
  
  return inner_2d_points;
}