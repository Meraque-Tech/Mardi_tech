#include "ros_utils.h"

std::tuple<sl::float3, sl::float3, sl::float3, double, double> AlgoUtils::gen_wp_all(rclcpp::Time now, sl::float3 main_l, sl::float3 main_r, float bed_dimension, int type_wp)
{
    std::vector<sl::float3> bbx_3d_;
    std::vector<sl::float3> bbx_2d_;
    // 0 left 1 mid 2 right
    auto[gth_m_h, gth_m_v] = find_HV_slop(main_l.x, main_l.y, main_r.x, main_r.y);

    auto [main_l_xf, main_l_yf] = WP_creation(gth_m_v, main_l.x, main_l.y, type_wp, bed_dimension);
    sl::float3 main_lf = {main_l_xf, main_l_yf, main_l.z};
    sl::float3 mid_l_lf = mid_point_finder(main_l, main_lf);

    auto [main_r_xf, main_r_yf] = WP_creation(gth_m_v, main_r.x, main_r.y, type_wp, bed_dimension);
    sl::float3 main_rf = {main_r_xf, main_r_yf, main_r.z};
    sl::float3 mid_r_rf = mid_point_finder(main_r, main_rf);


    bbx_3d_.push_back(main_l); //0
    bbx_3d_.push_back(main_lf); //1
    bbx_3d_.push_back(main_rf); //2
    bbx_3d_.push_back(main_r); //3
    bbx_3d_.push_back(nall_strc); //4
    bbx_3d_.push_back(nall_strc); //5
    bbx_3d_.push_back(nall_strc); //6
    bbx_3d_.push_back(nall_strc); //7
    camera_laser_pub(now, bbx_3d_);

    bbx_2d_.push_back(main_l); //0
    bbx_2d_.push_back(main_lf); //1
    bbx_2d_.push_back(main_rf); //2
    bbx_2d_.push_back(main_r); //3

    // send_static_target_tf(now, map_frame, "eular_orientation", 1.0, 0.0, 0.0, 0.0, 0.0, 0.0);

    if(bed_dimension == bed_L)
    {
        // sl::float3 tf_sub_to_eular_angles_ = tf_sub_to_eular_angles(map_frame, "eular_orientation");
        sl::float3 midpoint_2d = midpoint_of_rectangle(bbx_2d_);
        send_static_target_tf(now, camera_frame, "bbx_2d_mid_point", midpoint_2d.x, midpoint_2d.y, 0.0, 0.0, 0.0, gth_m_h);

        // std::cout<<" left_d_bed --> "<<d_bed<<" type --> "<<type<<std::endl;
        send_static_target_tf(now, camera_frame, "main_l", main_l.x, main_l.y, 0.0, 0.0, 0.0, gth_m_h);
        send_static_target_tf(now, camera_frame, "main_lf", main_l_xf, main_l_yf, 0.0, 0.0, 0.0, gth_m_h);
        send_static_target_tf(now, camera_frame, "mid_l_lf", mid_l_lf.x, mid_l_lf.y, 0.0, 0.0, 0.0, gth_m_h);

        send_static_target_tf(now, camera_frame, "main_r", main_r.x, main_r.y, 0.0, 0.0, 0.0, gth_m_h);
        send_static_target_tf(now, camera_frame, "main_rf", main_r_xf, main_r_yf, 0.0, 0.0, 0.0, gth_m_h);
        send_static_target_tf(now, camera_frame, "mid_r_rf", mid_r_rf.x, mid_r_rf.y, 0.0, 0.0, 0.0, gth_m_h);
        return std::make_tuple(mid_l_lf, mid_r_rf, midpoint_2d, gth_m_h, gth_m_v);
    }

    if(bed_dimension == bed_W)
    {
        // sl::float3 tf_sub_to_eular_angles_ = tf_sub_to_eular_angles(map_frame, "eular_orientation");
        sl::float3 midpoint_2d = midpoint_of_rectangle(bbx_2d_);
        send_static_target_tf(now, camera_frame, "bbx_2d_mid_point", midpoint_2d.x, midpoint_2d.y, 0.0, 0.0, 0.0, M_PI+gth_m_h);

        // std::cout<<" left_d_bed --> "<<d_bed<<" type --> "<<type<<std::endl;
        send_static_target_tf(now, camera_frame, "main_l", main_l.x, main_l.y, 0.0, 0.0, 0.0, M_PI+gth_m_h);
        send_static_target_tf(now, camera_frame, "main_lf", main_l_xf, main_l_yf, 0.0, 0.0, 0.0, M_PI+gth_m_h);
        send_static_target_tf(now, camera_frame, "mid_l_lf", mid_l_lf.x, mid_l_lf.y, 0.0, 0.0, 0.0, M_PI+gth_m_h);

        send_static_target_tf(now, camera_frame, "main_r", main_r.x, main_r.y, 0.0, 0.0, 0.0, M_PI+gth_m_h);
        send_static_target_tf(now, camera_frame, "main_rf", main_r_xf, main_r_yf, 0.0, 0.0, 0.0, M_PI+gth_m_h);
        send_static_target_tf(now, camera_frame, "mid_r_rf", mid_r_rf.x, mid_r_rf.y, 0.0, 0.0, 0.0, M_PI+gth_m_h);
        return std::make_tuple(mid_l_lf, mid_r_rf, midpoint_2d, gth_m_h, gth_m_v);
    }

    

    // }
    // catch (const tf2::TransformException & ex) 
    // {
    //     std::cout<<"Could not transform"<<std::endl;
    //     return std::make_tuple(nall_strc, nall_strc, nall_strc, 0.0, 0.0);
    // }
    

    
}

void AlgoUtils::wp_vec(rclcpp::Time now, std::string parent_frame, std::string child_frame)
{
    try 
    {
        geometry_msgs::msg::TransformStamped tf_sub_l = tf_sub(parent_frame, child_frame);
        geometry_msgs::msg::PoseStamped tf2_pose_l = tf2_pose(now, tf_sub_l);
        tf2_pose_vec_.push_back(tf2_pose_l);

    }
    catch (const tf2::TransformException & ex) 
    {
        std::cout<<"Could not transform"<<std::endl;
    }
}

void AlgoUtils::wp_vec_pub(rclcpp::Time now, std::vector<geometry_msgs::msg::PoseStamped> &tf2_pose_vec_)
{
    if (!tf2_pose_vec_.empty())
    {
        nav_msgs::msg::Path path_msg;
        path_msg.header.stamp = now;
        path_msg.poses = tf2_pose_vec_;
        pose_array_pub->publish(path_msg);
        
        updateWpNavigationMarkers(tf2_pose_vec_);
        tf2_pose_vec_.clear();
        // exit(0);
        
    }

}


void AlgoUtils::gen_wp_logic(rclcpp::Time now, sl::float3 main_l, sl::float3 main_r, int type)
{
    // 0 left 1 mid 2 right
    float d_bed = euclidean_distance_twoarg(main_l, main_r);
    if(type == 0)
    {
        if(d_bed < bed_W)
        {
            std::vector<geometry_msgs::msg::PoseStamped> tf2_pose_vec;
            std::cout<<" <--- left_foot --> "<<cnt<<std::endl;
            auto [mid_l_lf, mid_r_rf, midpoint_2d, gth_m_h, gth_m_v] = gen_wp_all(now, main_l, main_r, bed_L, 4);
            auto [wp_l_xf, wp_l_yf] = WP_creation(gth_m_h, midpoint_2d.x, midpoint_2d.y, 1, k);
            send_static_target_tf(now, camera_frame, "wp_lf", wp_l_xf, wp_l_yf, 0.0, 0.0, 0.0, M_PI+gth_m_h);
            wp_vec(now, map_frame, "wp_lf");
            wp_vec_pub(now, tf2_pose_vec_);
            
        }
        // else if(d_bed > 1.5 && d_bed < bed_L)
        // {
        //     float k_ = k+0.5;
        //     auto [mid_l_lf, mid_r_rf, midpoint_2d, gth_m_h, gth_m_v] = gen_wp_all(now, main_l, main_r, bed_W, 4);
        //     auto [wp_l_xf, wp_l_yf] = WP_creation(gth_m_h, midpoint_2d.x, midpoint_2d.y, 1, k_);
        //     send_static_target_tf(now, camera_frame, "wp_lf", wp_l_xf, wp_l_yf, 0.0, 0.0, 0.0, -gth_m_h);
        //     std::cout<<" <--- left_long --> "<<cnt<<std::endl;
        // }

        
    }
    
    else if(type == 2)
    {
        if(d_bed < bed_W)
        {
            
            std::cout<<" <--- right_foot --> "<<cnt<<std::endl;
            auto [mid_l_lf, mid_r_rf, midpoint_2d, gth_m_h, gth_m_v] = gen_wp_all(now, main_l, main_r, bed_L, 4);
            auto [wp_r_xf, wp_r_yf] = WP_creation(gth_m_h, midpoint_2d.x, midpoint_2d.y, 3, k);
            send_static_target_tf(now, camera_frame, "wp_rf", wp_r_xf, wp_r_yf, 0.0, 0.0, 0.0, M_PI+gth_m_h);
        }
        // else if(d_bed > 1.5 && d_bed < bed_L)
        // {
        //     float k_ = k+0.5;
        //     auto [mid_l_lf, mid_r_rf, midpoint_2d, gth_m_h, gth_m_v] = gen_wp_all(now, main_l, main_r, bed_W);
        //     auto [wp_r_xf, wp_r_yf] = WP_creation(gth_m_h, midpoint_2d.x, midpoint_2d.y, 2, k_);
        //     send_static_target_tf(now, camera_frame, "wp_rf", wp_r_xf, wp_r_yf, 0.0, 0.0, 0.0, gth_m_h);
        //     std::cout<<" <--- right_long --> "<<cnt<<std::endl;
        // }

        
    }

    

}