#include "ros_utils.h"

std::vector<sl::float3> AlgoUtils::zed2_3d_bbx_vec(sl::Objects &objects)
{
    std::vector<sl::float3> bbx_3d;
    for (auto data : objects.object_list)
    {
        if (data.bounding_box.size() == 8)
        {
            for(auto it : data.bounding_box)
            {
                bbx_3d.push_back({it.x, it.y, it.z});
            }
        }
    }
    return bbx_3d;
}

void AlgoUtils::bbx_3d_tf_pub(rclcpp::Time now, const std::vector<sl::float3>& bbx_3d)
{
    int icc = 0;
    for(auto it_:bbx_3d)
    {
        if(icc<8)
        {
            rclcpp::Time now = now;
            send_static_target_tf(now, camera_frame, "bbx_3d_" + std::to_string(icc), it_.x, it_.y, it_.z, 0.0, 0.0, 0.0);
            icc +=1;
        }
        
    }
}

std::tuple<sl::float3, sl::float3, sl::float3, sl::float3> AlgoUtils::mid_points_lfr(rclcpp::Time now, std::vector<sl::float3> &bbx_3d)
{
    sl::float3 midpoint = midpoint_of_rectangle(bbx_3d);
    send_static_target_tf(now, camera_frame, "bbx_3d_mid_point", midpoint.x, midpoint.y, midpoint.z, 0.0, 0.0, 0.0);

    std::vector<sl::float3> vertices{bbx_3d.at(0), bbx_3d.at(3), bbx_3d.at(7),bbx_3d.at(4)};
    sl::float3 midpoint_f = midpoint_of_rectangle(vertices);
    send_static_target_tf(now, camera_frame, "bbx_3d_mid_point_f", midpoint_f.x, midpoint_f.y, midpoint_f.z, 0.0, 0.0, 0.0);
    vertices.clear();

    vertices = {bbx_3d.at(0), bbx_3d.at(1), bbx_3d.at(5),bbx_3d.at(4)};
    sl::float3 midpoint_l = midpoint_of_rectangle(vertices);
    send_static_target_tf(now, camera_frame, "bbx_3d_mid_point_l", midpoint_l.x, midpoint_l.y, midpoint_l.z, 0.0, 0.0, 0.0);
    vertices.clear();

    vertices = {bbx_3d.at(2), bbx_3d.at(3), bbx_3d.at(7),bbx_3d.at(6)};
    sl::float3 midpoint_r = midpoint_of_rectangle(vertices);
    send_static_target_tf(now, camera_frame, "bbx_3d_mid_point_r", midpoint_r.x, midpoint_r.y, midpoint_r.z, 0.0, 0.0, 0.0);
    vertices.clear();

    return std::make_tuple(midpoint, midpoint_f, midpoint_l, midpoint_r);
}

sl::float3 AlgoUtils::pix_pcl(int pix_x, int pix_y, sl::Mat &point_cloud, cv::Mat &image)
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

sl::float3 AlgoUtils::mid_point_finder(sl::float3 &lef_bed_pcl, sl::float3 &right_bed_pcl) {
    sl::float3 midpoint;

    midpoint.x = (lef_bed_pcl.x + right_bed_pcl.x) / 2;
    midpoint.y = (lef_bed_pcl.y + right_bed_pcl.y) / 2;
    midpoint.z = (lef_bed_pcl.z + right_bed_pcl.z) / 2;

    return midpoint;
}
