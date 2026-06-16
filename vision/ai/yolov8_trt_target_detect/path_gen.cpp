#include "path_gen.h"
#include "function_ex.h"

double unique_id;

float euclidean_distance_onearg(const sl::float3& point1) 
{
    float dx = point1.x;
    float dy = point1.y;
    float dz = point1.z;
    return std::sqrt(dx*dx + dy*dy + dz*dz);
}

float euclidean_distance_twoarg(const sl::float3& point1, const sl::float3& point2) 
{
    float dx = point2.x - point1.x;
    float dy = point2.y - point1.y;
    float dz = point2.z - point1.z;
    return std::sqrt(dx*dx + dy*dy + dz*dz);
}

double Cosin(double a, double b, double c)
{
  return acos(((a*a)+(b*b)-(c*c))/(2*a*b));
}

double gamma_sign_correction(float gamma, float x)
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

std::tuple<float, float> find_global_PQR(const sl::float3& img_frame, const sl::float3& bed_PQR)
{
    float d_img_frame = euclidean_distance_onearg(img_frame);
    float d_bed_PQR = euclidean_distance_onearg(bed_PQR);
    float d_img_frame_bed_PQR = euclidean_distance_twoarg(img_frame, bed_PQR);
    float gamma_PQR_C_im = Cosin(d_img_frame, d_bed_PQR, d_img_frame_bed_PQR);
    float gammma_corr = gamma_sign_correction(gamma_PQR_C_im, bed_PQR.x);
    float gpx = d_bed_PQR*sin(gammma_corr);
    float gpz = d_bed_PQR*cos(gammma_corr);
    return {-gpx, gpz};
}

void send_static_target_tf( static_tf2 static_target_tf,
                            rclcpp::Time now, 
                            std::string parent_frame, 
                            std::string child_frame, 
                            double x, 
                            double y, 
                            double z, 
                            double th_x, 
                            double th_y, 
                            double th_z)
{
  geometry_msgs::msg::TransformStamped tf_msg;
  // rclcpp::Time now = this->get_clock()->now();
  tf_msg.header.stamp = now;
  tf_msg.header.frame_id = parent_frame;
  tf_msg.child_frame_id = child_frame;

  tf_msg.transform.translation.x = x;
  tf_msg.transform.translation.y = y;
  tf_msg.transform.translation.z = z;

  tf2::Quaternion q;
  q.setRPY(th_x, th_y, th_z);
  tf_msg.transform.rotation.x = q.x();
  tf_msg.transform.rotation.y = q.y();
  tf_msg.transform.rotation.z = q.z();
  tf_msg.transform.rotation.w = q.w();

  static_target_tf->sendTransform(tf_msg);
}

std::tuple<float, float> find_HV_slop(float x1, float y1, float x2, float y2)
{
    float gth_m_h = atan2((y2 - y1), (x2 - x1));
    float gth_m_v = atan2(-(x2 - x1), (y2 - y1));
    return {gth_m_h, gth_m_v};
}

std::tuple<float, float> WP_creation(float gth_m, float gpx, float gpy, int type_wp, float k)
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
    return {gpx_f, gpy_f};

}

int getOCVtype(sl::MAT_TYPE type)
{
    int cv_type = -1;
    switch (type) {
        case sl::MAT_TYPE::F32_C1: cv_type = CV_32FC1; break;
        case sl::MAT_TYPE::F32_C2: cv_type = CV_32FC2; break;
        case sl::MAT_TYPE::F32_C3: cv_type = CV_32FC3; break;
        case sl::MAT_TYPE::F32_C4: cv_type = CV_32FC4; break;
        case sl::MAT_TYPE::U8_C1: cv_type = CV_8UC1; break;
        case sl::MAT_TYPE::U8_C2: cv_type = CV_8UC2; break;
        case sl::MAT_TYPE::U8_C3: cv_type = CV_8UC3; break;
        case sl::MAT_TYPE::U8_C4: cv_type = CV_8UC4; break;
        default: break;
    }
    return cv_type;
}

std::vector<sl::uint2> cvt(const cv::Rect &bbox_in){
    std::vector<sl::uint2> bbox_out(4);
    bbox_out[0] = sl::uint2(bbox_in.x, bbox_in.y);
    bbox_out[1] = sl::uint2(bbox_in.x + bbox_in.width, bbox_in.y);
    bbox_out[2] = sl::uint2(bbox_in.x + bbox_in.width, bbox_in.y + bbox_in.height);
    bbox_out[3] = sl::uint2(bbox_in.x, bbox_in.y + bbox_in.height);
    return bbox_out;
}


cv::Mat slMat2cvMat(sl::Mat& input) {
    // Since cv::Mat data requires a uchar* pointer, we get the uchar1 pointer from sl::Mat (getPtr<T>())
    // cv::Mat and sl::Mat will share a single memory structure
    return cv::Mat(input.getHeight(), input.getWidth(), getOCVtype(input.getDataType()), input.getPtr<sl::uchar1>(sl::MEM::CPU), input.getStepBytes(sl::MEM::CPU));
}



std::tuple<float, size_t, float, size_t> minmax_idx(const std::vector<float>& v) {
    if (v.empty()) {
        return {}; // Return an empty tuple for an empty input
    }

    auto minmax_pair = std::minmax_element(v.begin(), v.end());

    float min_val = *minmax_pair.first;
    size_t min_index = static_cast<size_t>(std::distance(v.begin(), minmax_pair.first));

    float max_val = *minmax_pair.second;
    size_t max_index = static_cast<size_t>(std::distance(v.begin(), minmax_pair.second));

    return std::make_tuple(min_val, min_index, max_val, max_index);
}

sl::float3 pix_pcl(int pix_x, int pix_y, sl::Mat &point_cloud, cv::Mat &image)
{
    sl::float4 point_cloud_value;
    point_cloud.getValue(pix_x, pix_y, &point_cloud_value);
    if (std::isfinite(point_cloud_value.x) && std::isfinite(point_cloud_value.y) && std::isfinite(point_cloud_value.z))
    {
        sl::float3 pcl;
        pcl.x = point_cloud_value.x;
        pcl.y = point_cloud_value.y;
        pcl.z = point_cloud_value.z;
        return pcl;
    }
    else
    {
        sl::float3 pcl;
        pcl.x = 0.0;
        pcl.y = 0.0;
        pcl.z = 0.0;
        return pcl;
    }
}


float dis_from_pane(sl::float4 &plane_eq,sl::float3 &pcl)
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
        distance = (a * X + b * Y + c * Z + d) / sqrt(a * a + b * b + c * c); // distance notmal vector
        return distance;
    }
    
    
    
}

float euclidean_distance(const sl::float3& point1, const sl::float3& point2) {
    float dx = point2.x - point1.x;
    float dy = point2.y - point1.y;
    float dz = point2.z - point1.z;
    return std::sqrt(dx*dx + dy*dy + dz*dz);
}

sl::float4 findPlaneCoefficients(sl::float3 pix_pcl_P,
                                 sl::float3 pix_pcl_Q,
                                 sl::float3 pix_pcl_R)
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

sl::float3 find_foot_point_on_plane(const sl::float4 &plane_eq, const sl::float3 &pcl)
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

std::tuple<sl::float3, cv::Point> min_pix_V(cv::Mat &img, sl::Mat &point_cloud, sl::float4 &plane_eq, cv::Rect &r, cv::Point &topLeft, cv::Point &topRight, int I_yy, sl::float3 &riff_cluster)
{
    float diff = r.x/200;
    std::vector<cv::Point> plane_points_top;
    std::vector<float> cluster_points_pcl_top;
    std::vector<sl::float3> pix_pcl_top;
    for(float i = topLeft.x; i <= topRight.x; i+=diff) 
    {
        sl::float3 pix_pcl_ = pix_pcl(i, I_yy, point_cloud, img);
        float dis_from_pane_ = dis_from_pane(plane_eq, pix_pcl_);

        if (dis_from_pane_> 0.1f && pix_pcl_.x != 0 && pix_pcl_.y != 0 && pix_pcl_.z != 0)
        {
            cv::circle(img, cv::Point(i, I_yy), 2, (0, 255, 0), -1, 1, 0);
            plane_points_top.push_back(cv::Point(i, I_yy));
            cluster_points_pcl_top.push_back(euclidean_distance(pix_pcl_, riff_cluster));
            pix_pcl_top.push_back(pix_pcl_);
        }
    }
    if (!cluster_points_pcl_top.empty())
    {
        auto [min_val, min_index, max_val, max_index] = minmax_idx(cluster_points_pcl_top);
        
        int dx = plane_points_top.at(min_index).x;
        int dy = plane_points_top.at(min_index).y;
        sl::float3 pix_pcl_top_ = pix_pcl_top.at(min_index);

        cv::circle(img, cv::Point(dx, dy), 4, cv::Scalar(255, 0, 255), -1, 1, 0);

        return {pix_pcl_top_, cv::Point(dx, dy)};
    }

}

float min_pix_H(cv::Mat &img, sl::Mat &point_cloud, sl::float4 &plane_eq, cv::Rect &r, cv::Point &topLeft, cv::Point &topRight, sl::float3 &riff_cluster)
{
    float diff = r.x/30;
    std::vector<cv::Point> plane_points_top;
    std::vector<float> cluster_points_pcl_top;
    std::vector<sl::float3> pix_pcl_top;
    for(float i = topLeft.x; i <= topRight.x; i+=diff)
        for(float j = topLeft.y; j <= topRight.y; j+=diff)
        {
            sl::float3 pix_pcl_ = pix_pcl(i, j, point_cloud, img);
            float dis_from_pane_ = dis_from_pane(plane_eq, pix_pcl_);

            if (dis_from_pane_< 0.01f && pix_pcl_.x != 0 && pix_pcl_.y != 0 && pix_pcl_.z != 0)
            {
                cv::circle(img, cv::Point(i, j), 2, (0, 255, 0), -1, 1, 0);
                plane_points_top.push_back(cv::Point(i, j));
                cluster_points_pcl_top.push_back(euclidean_distance(pix_pcl_, riff_cluster));
                pix_pcl_top.push_back(pix_pcl_);
            }
        }
    if (!cluster_points_pcl_top.empty())
    {
        auto [min_val, min_index, max_val, max_index] = minmax_idx(cluster_points_pcl_top);
        
        int dx = plane_points_top.at(max_index).x;
        int dy = plane_points_top.at(max_index).y;
        sl::float3 pix_pcl_top_ = pix_pcl_top.at(max_index);

        cv::circle(img, cv::Point(dx, dy), 4, cv::Scalar(255, 0, 255), -1, 1, 0);

        return max_val;
    }

}

sl::float3 pix_pcl_cluster(cv::Mat &img, sl::Mat &point_cloud, sl::float4 &plane_eq, cv::Rect &r, int x, int y)
{

    // float diff = r.x/30;
    std::vector<cv::Point> plane_points_top;
    std::vector<float> cluster_points_pcl_top;
    std::vector<sl::float3> pix_pcl_top;
    int range = r.width/20;
    sl::float3 pix_pcl_;
    float dis_from_pane_;
    
    for(int i = 0; i < range; i++)
    {
        pix_pcl_ = pix_pcl(x+i, y+i, point_cloud, img);
        dis_from_pane_ = dis_from_pane(plane_eq, pix_pcl_);

        if (dis_from_pane_> 0.1f && pix_pcl_.x != 0 && pix_pcl_.y != 0 && pix_pcl_.z != 0)
        {
            cv::circle(img, cv::Point(x+i, y+i), 2, (0, 255, 0), -1, 1, 0);
            plane_points_top.push_back(cv::Point(x+i, y+i));
            cluster_points_pcl_top.push_back(euclidean_distance_onearg(pix_pcl_));
            pix_pcl_top.push_back(pix_pcl_);
        }

        pix_pcl_ = pix_pcl(x-i, y+i, point_cloud, img);
        dis_from_pane_ = dis_from_pane(plane_eq, pix_pcl_);

        if (dis_from_pane_> 0.1f && pix_pcl_.x != 0 && pix_pcl_.y != 0 && pix_pcl_.z != 0)
        {
            cv::circle(img, cv::Point(x-i, y+i), 2, (0, 255, 0), -1, 1, 0);
            plane_points_top.push_back(cv::Point(x-i, y+i));
            cluster_points_pcl_top.push_back(euclidean_distance_onearg(pix_pcl_));
            pix_pcl_top.push_back(pix_pcl_);
        }

        pix_pcl_ = pix_pcl(x+i, y-i, point_cloud, img);
        dis_from_pane_ = dis_from_pane(plane_eq, pix_pcl_);

        if (dis_from_pane_> 0.1f && pix_pcl_.x != 0 && pix_pcl_.y != 0 && pix_pcl_.z != 0)
        {
            cv::circle(img, cv::Point(x+i, y-i), 2, (0, 255, 0), -1, 1, 0);
            plane_points_top.push_back(cv::Point(x+i, y-i));
            cluster_points_pcl_top.push_back(euclidean_distance_onearg(pix_pcl_));
            pix_pcl_top.push_back(pix_pcl_);
        }

        pix_pcl_ = pix_pcl(x-i, y-i, point_cloud, img);
        dis_from_pane_ = dis_from_pane(plane_eq, pix_pcl_);

        if (dis_from_pane_> 0.1f && pix_pcl_.x != 0 && pix_pcl_.y != 0 && pix_pcl_.z != 0)
        {
            cv::circle(img, cv::Point(x-i, y-i), 2, (0, 255, 0), -1, 1, 0);
            plane_points_top.push_back(cv::Point(x-i, y-i));
            cluster_points_pcl_top.push_back(euclidean_distance_onearg(pix_pcl_));
            pix_pcl_top.push_back(pix_pcl_);
        }
        
    }

    if (!cluster_points_pcl_top.empty())
    {
        auto [min_val, min_index, max_val, max_index] = minmax_idx(cluster_points_pcl_top);
        
        int dx = plane_points_top.at(min_index).x;
        int dy = plane_points_top.at(min_index).y;
        sl::float3 pix_pcl_top_ = pix_pcl_top.at(min_index);

        cv::circle(img, cv::Point(dx, dy), 4, cv::Scalar(0, 0, 255), -1, 1, 0);

        return pix_pcl_top_;
    }
 
}

std::tuple<sl::float3, sl::float3, sl::float3, float, float, float, float> distance_cluster_lrf(cv::Mat &img, sl::Mat &point_cloud, sl::float4 &plane_eq, auto &it)
{
    cv::Rect r = get_rect(img, it.bbox);
    int origin = 10;

    int I_xx = (2*r.x+r.width)/2;
    int I_yy =  (2*r.y+r.height)/2;

    int I_xxL = (r.x + origin)/2;
    int I_yyL =  I_yy;

    int I_xxR = (r.x+r.width+img.cols-origin)/2;
    int I_yyR =  I_yy;

    int im_x = img.cols/2;
    int im_y = img.rows/2;

    // cv::circle(img, cv::Point(im_x, im_y), 4, cv::Scalar(0, 255, 255), -1, 1, 0);

    cv::Point topLeft_Img = cv::Point(origin, r.y);
    cv::Point bottomRight_Img = cv::Point(origin-10, r.y+r.height);

    cv::Point topLeft = cv::Point(r.x, r.y);
    cv::Point topLeft_mid = cv::Point(I_xxL+(r.x - origin)/3, r.y); 
    cv::Point topright = cv::Point(r.x+r.width, r.y);
    cv::Point topright_mid = cv::Point(I_xxR-(img.cols-origin-r.x-r.width)/3, r.y);
    cv::Point bottomLeft = cv::Point(r.x, r.y+r.height);
    cv::Point bottomRight = cv::Point(r.x+r.width, r.y+r.height);
    cv::Point img_topLeft = cv::Point(r.x/25,r.y);
    cv::Point img_topRight = cv::Point(img.cols-r.x/25,r.y);



    cv::rectangle(img, r, cv::Scalar(0x27, 0xC1, 0x36), 2);
    cv::rectangle(img, cv::Point(origin,r.y), bottomLeft, cv::Scalar(0x27, 0, 0), 2);
    cv::rectangle(img, topright, cv::Point(img.cols-origin, r.y+r.height), cv::Scalar(0x27, 0, 0), 2);
    cv::rectangle(img, cv::Point(origin, r.y+r.height), cv::Point(img.cols-origin, img.rows-origin), cv::Scalar(0x27, 0, 0), 2);

    cv::putText(img, std::to_string((int) it.class_id), cv::Point(r.x, r.y - 1), cv::FONT_HERSHEY_PLAIN,
        1.2, cv::Scalar(0xFF, 0xFF, 0xFF), 2);
    
    sl::float3 center_im = pix_pcl_cluster(img, point_cloud, plane_eq, r, im_x, im_y); // detected object(bed) min Img point from mid cluster
    sl::float3 mid_L_bed = pix_pcl_cluster(img, point_cloud, plane_eq, r, r.x, I_yy); // detected object(bed) min Left point from mid cluster
    sl::float3 mid_P_bed = pix_pcl_cluster(img, point_cloud, plane_eq, r, I_xx-r.width/3, I_yy); // detected object(bed) min P point from mid cluster
    sl::float3 mid_bed = pix_pcl_cluster(img, point_cloud, plane_eq, r, I_xx, I_yy); // detected object(bed) min mid point from mid cluster
    sl::float3 mid_Q_bed = pix_pcl_cluster(img, point_cloud, plane_eq, r, I_xx+r.width/3, I_yy); // detected object(bed) min Q point from mid cluster
    sl::float3 mid_R_bed = pix_pcl_cluster(img, point_cloud, plane_eq, r, r.x+r.width, I_yy); // detected object(bed) min Right point from mid cluster

    // find left and right side Horizontal distance for robot to enter
    auto [LL_object_vec, LL_object_vec_cv] = min_pix_V(img, point_cloud, plane_eq, r, img_topLeft, topLeft_mid, I_yy, mid_L_bed); // detected object(bed) min LL_object_vector point from mid cluster
    auto [RR_object_vec, RR_object_vec_cv] = min_pix_V(img, point_cloud, plane_eq, r, topright_mid, img_topRight, I_yy, mid_R_bed); // detected object(bed) min RR_object_vector point from mid cluster



    // find bed plain equation
    sl::float4 findPlaneCoefficients_PQR =  findPlaneCoefficients(mid_bed, mid_P_bed, mid_Q_bed); // find findPlaneCoefficients_PQR
    
    // // find vertical distance from bed plane to LL_object_vec and RR_object_vec
    cv::Point RR_in_plane = cv::Point(RR_object_vec_cv.x, r.y+r.height);
    float LV_dis = min_pix_H(img, point_cloud, plane_eq, r, LL_object_vec_cv, bottomLeft, mid_L_bed);
    float RV_dis = min_pix_H(img, point_cloud, plane_eq, r, topright, RR_in_plane, mid_R_bed);

    cv::Point topLeft_Img_ = cv::Point(r.x, r.y+r.height);
    cv::Point bottomRight_Img_ = cv::Point(r.x+r.width, img.rows-origin);
    // cv::Point mid_object = cv::Point(mid_bed.x, r.y+r.height);
    float front_H_dis = min_pix_H(img, point_cloud, plane_eq, r, topLeft_Img_, bottomRight_Img_, mid_bed);

    // // find foot coordinates of LL_object_vec, RR_object_vec, mid_R_bed, mid_L_bed in bed plane.
    sl::float3 LL_object_vec_foot_point = find_foot_point_on_plane(findPlaneCoefficients_PQR, LL_object_vec);
    sl::float3 mid_L_bed_foot_point = find_foot_point_on_plane(findPlaneCoefficients_PQR, mid_L_bed);
    sl::float3 RR_object_vec_foot_point = find_foot_point_on_plane(findPlaneCoefficients_PQR, RR_object_vec);
    sl::float3 mid_R_bed_foot_point = find_foot_point_on_plane(findPlaneCoefficients_PQR, mid_R_bed);
    

    // find Left horizontal euclidean distanc from left most cluster of the bed to left most min clustr of the envirmoments.
    float LH_dis = euclidean_distance(mid_L_bed_foot_point, LL_object_vec_foot_point);
    // find Left horizontal euclidean distanc from right most min cluster of the bed to right most min clustr of the envirmoments.
    float RH_dis = euclidean_distance(mid_R_bed_foot_point, RR_object_vec_foot_point);

    return {center_im, mid_P_bed, mid_Q_bed, LV_dis, RV_dis, LH_dis, RH_dis};
}

geometry_msgs::msg::PoseStamped tf2_pose(rclcpp::Time now, geometry_msgs::msg::TransformStamped &tf_msg)
{   
    geometry_msgs::msg::PoseStamped goal_pose;
    goal_pose.header.stamp = now;
    goal_pose.header.frame_id = "map";
    goal_pose.pose.position.x = tf_msg.transform.translation.x;
    goal_pose.pose.position.y = tf_msg.transform.translation.y;
    goal_pose.pose.position.z = 0.0;
    goal_pose.pose.orientation.x = 0.0;
    goal_pose.pose.orientation.y = 0.0;
    goal_pose.pose.orientation.z = tf_msg.transform.rotation.z;
    goal_pose.pose.orientation.w = tf_msg.transform.rotation.w;
    return goal_pose;
}


void updateWpNavigationMarkers(viz_msg_f wp_navigation_markers_pub_,std::vector<geometry_msgs::msg::PoseStamped> &poses_)
{
    resetUniqueId();
    auto marker_array = std::make_unique<visualization_msgs::msg::MarkerArray>();
    for (size_t i = 0; i < poses_.size(); i++)
    {
        visualization_msgs::msg::Marker arrow_marker;
        arrow_marker.header = poses_[i].header;
        arrow_marker.id = getUniqueId();
        arrow_marker.type = visualization_msgs::msg::Marker::ARROW;
        arrow_marker.action = visualization_msgs::msg::Marker::ADD;
        arrow_marker.pose = poses_[i].pose;
        arrow_marker.scale.x = 0.3;
        arrow_marker.scale.y = 0.05;
        arrow_marker.scale.z = 0.02;
        arrow_marker.color.r = 0;
        arrow_marker.color.g = 255;
        arrow_marker.color.b = 0;
        arrow_marker.color.a = 1.0f;
        // arrow_marker.lifetime = rclcpp::Duration(0s);
        arrow_marker.frame_locked = false;
        marker_array->markers.push_back(arrow_marker);

        // Draw a red circle at the waypoint pose
        visualization_msgs::msg::Marker circle_marker;
        circle_marker.header = poses_[i].header;
        circle_marker.id = getUniqueId();
        circle_marker.type = visualization_msgs::msg::Marker::SPHERE;
        circle_marker.action = visualization_msgs::msg::Marker::ADD;
        circle_marker.pose = poses_[i].pose;
        circle_marker.scale.x = 0.05;
        circle_marker.scale.y = 0.05;
        circle_marker.scale.z = 0.05;
        circle_marker.color.r = 255;
        circle_marker.color.g = 0;
        circle_marker.color.b = 0;
        circle_marker.color.a = 1.0f;
        // circle_marker.lifetime = rclcpp::Duration(0s);
        circle_marker.frame_locked = false;
        marker_array->markers.push_back(circle_marker);

        // Draw the waypoint number
        visualization_msgs::msg::Marker marker_text;
        marker_text.header = poses_[i].header;
        marker_text.id = getUniqueId();
        marker_text.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
        marker_text.action = visualization_msgs::msg::Marker::ADD;
        marker_text.pose = poses_[i].pose;
        marker_text.pose.position.z += 0.2;  // draw it on top of the waypoint
        marker_text.scale.x = 0.07;
        marker_text.scale.y = 0.07;
        marker_text.scale.z = 0.07;
        marker_text.color.r = 0;
        marker_text.color.g = 255;
        marker_text.color.b = 0;
        marker_text.color.a = 1.0f;
        // marker_text.lifetime = rclcpp::Duration(0s);
        marker_text.frame_locked = false;
        marker_text.text = "wp_" + std::to_string(i + 1);
        marker_array->markers.push_back(marker_text);
    }

    if (marker_array->markers.empty()) 
    {
    visualization_msgs::msg::Marker clear_all_marker;
    clear_all_marker.action = visualization_msgs::msg::Marker::DELETEALL;
    marker_array->markers.push_back(clear_all_marker);
    }

    wp_navigation_markers_pub_->publish(std::move(marker_array));
    poses_.clear();

}

void resetUniqueId()
{
  unique_id = 0;
}

int getUniqueId()
{
  int temp_id = unique_id;
  unique_id += 1;
  return temp_id;
}

geometry_msgs::msg::TransformStamped tf_sub(std::string parent_frame, std::string child_frame)
{
  geometry_msgs::msg::TransformStamped transformStamped;
  transformStamped = tf_buffer_->lookupTransform(
    parent_frame.c_str(), child_frame.c_str(),
    tf2::TimePointZero);

    return transformStamped;
}

void startNavToPose(geometry_msgs::msg::PoseStamped pose)
{
  auto is_action_server_ready = navigation_action_client_->wait_for_action_server(std::chrono::seconds(5));
  if (!is_action_server_ready) {
    std::cout << "navigate_to_pose action server is not available \n"<< std::endl;
    return;
  }

  // Send the goal pose
  navigation_goal_.pose = pose;

  auto send_goal_options = rclcpp_action::Client<nav2_msgs::action::NavigateToPose>::SendGoalOptions();

  send_goal_options.result_callback = [node](auto) {
      navigation_goal_handle_.reset();
    };

  auto future_goal_handle = navigation_action_client_->async_send_goal(navigation_goal_, send_goal_options);
  
  if (rclcpp::spin_until_future_complete(node, future_goal_handle, server_timeout_) != rclcpp::FutureReturnCode::SUCCESS)
  {
    std::cout << "Send goal call failed \n"<< std::endl;
    return;
  }

//   // Get the goal handle and save so that we can check on completion in the timer callback
  navigation_goal_handle_ = future_goal_handle.get();
  if (!navigation_goal_handle_) {
    std::cout << "Goal was rejected by server \n"<< std::endl;
    // RCLCPP_ERROR(client_node_->get_logger(), "Goal was rejected by server");
    return;
  }
}


// std::tuple<std::vector<float>, std::vector<float>> path_generator(float xi, float yi, float xf, float yf)
// {
//   vector<float> x;
//   vector<float> y;

// //   for( int j = 0; j < NOP; j+=1)
// //   {
    
// //     float PX = (j*xf + (NOP-j)*xi)/(NOP);
// //     float PY = (j*yf + (NOP-j)*yi)/(NOP);

// //     x.push_back(PX);
// //     y.push_back(PY);
    
// //   }
  
//   return {x, y};
// }