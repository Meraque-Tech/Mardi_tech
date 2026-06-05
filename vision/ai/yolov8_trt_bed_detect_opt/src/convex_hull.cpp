#include "ros_utils.h"

std::tuple<float, size_t, float, size_t> AlgoUtils::minmax_idx(const std::vector<auto>& v) {
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


std::tuple<sl::float3, int> AlgoUtils::min_cluster_pix_mask_find(cv::Mat &img, std::vector<int>& cluster_points_pix_top ,std::vector<cv::Point>& cluster_points_pix_top_uv, cv::Point& bbx_up, std::vector<sl::float3> &pix_pcl_top, sl::float3 &pix_pcl_img_cm)
{
    auto [min_val, min_index, max_val, max_index] = minmax_idx(cluster_points_pix_top);

    int dx_min = cluster_points_pix_top_uv.at(min_index).x;
    int dy_min = cluster_points_pix_top_uv.at(min_index).y;
    int dx_max = cluster_points_pix_top_uv.at(max_index).x;
    int dy_max = cluster_points_pix_top_uv.at(max_index).y;

    sl::float3 pix_pcl_top_min_ = pix_pcl_top.at(min_index);

    // sl::float3 global_pix_pcl_top_min_;
    // auto[gpx, gpy] = find_global_PQR(pix_pcl_img_cm, pix_pcl_top_min_);
    // global_pix_pcl_top_min_.x = gpx;
    // global_pix_pcl_top_min_.y = gpy;
    // global_pix_pcl_top_min_.z = pix_pcl_top_min_.z;
    // for(int i = bbx_up.y; i <= dy_min; i+=2)
    // {
    //     cv::circle(img, cv::Point(dx_min, i), 2, (255, 255, 0), -1, 1, 0);
    // }

    cv::circle(img, cv::Point(dx_min, dy_min), 8, (255, 255, 0), -1, 1, 0);
    return std::make_tuple(pix_pcl_top_min_, min_index);
    // return std::make_tuple(global_pix_pcl_top_min_, min_index);
}






void AlgoUtils::find_object_mask_convexhull(rclcpp::Time now, cv::Mat &img, std::vector<sl::uint2> object_2Dbbox, sl::float4 &plane_eq, sl::Mat &point_cloud)
{
    cv::Point left_up = cv::Point(object_2Dbbox[0][0],object_2Dbbox[0][1]);
    cv::Point left_down = cv::Point(object_2Dbbox[3][0],object_2Dbbox[3][1]);
    cv::Point right_up = cv::Point(object_2Dbbox[1][0],object_2Dbbox[1][1]);
    cv::Point right_down = cv::Point(object_2Dbbox[2][0],object_2Dbbox[2][1]);

    cv::Point left_mid = cv::Point((left_up.x+left_down.x)/2, (left_up.y+left_down.y)/2);
    cv::Point right_mid = cv::Point((right_up.x+right_down.x)/2, (right_up.y+right_down.y)/2);
    cv::Point mid_img = cv::Point((left_up.x+right_down.x)/2, (left_up.y+right_down.y)/2);

    cv::Point left_left_up = cv::Point(stride, object_2Dbbox[0][1]);
    cv::Point left_left_down = cv::Point(stride, object_2Dbbox[3][1]);

    cv::Point right_right_up = cv::Point(img.size().width-stride,object_2Dbbox[1][1]);
    cv::Point right_right_down = cv::Point(img.size().width-stride,object_2Dbbox[2][1]);
    

    l_up = left_up;
    ll_up = left_left_up;
    l_down = left_down;
    ll_down = left_left_down;

    r_up = right_up;
    rr_up = right_right_up;
    r_down = right_down;
    rr_down = right_right_down;

    img_ = img;
    plane_eq_ = plane_eq; 
    point_cloud_ = point_cloud;

    cv::circle(img, l_up, 10, (255, 0, 255), -1, 1, 0);
    // cv::circle(img, right_mid, 5, (255, 255, 0), -1, 1, 0);
    // cv::circle(img, mid_img, 5, (255, 255, 0), -1, 1, 0);

    // cv::circle(img, left_left_down, 5, (255, 255, 0), -1, 1, 0);
    // cv::circle(img, right_right_up, 5, (255, 255, 0), -1, 1, 0);

    
    float prev_lm=0;
    float prev_rm=0;
    

    float diff = abs(left_up.x - right_up.x)/50;

    if(diff > 0)
    {
        std::vector<cv::Point> cluster_points_pix_top_uv;
        std::vector<sl::float3> pix_pcl_top;

        for(float i = left_up.x; i <= right_down.x; i+=diff)
        {
            for(float j = left_up.y; j <= right_down.y; j+=diff)
            {
                // cv::circle(img, cv::Point(i, j), 2, (0, 255, 0), -1, 1, 0);
                sl::float3 pix_pcl_ = pix_pcl(i,j, point_cloud, img);
                float dis_from_pane_ = dis_from_pane(plane_eq, pix_pcl_);
                // cv::circle(img, cv::Point(i, j), 1, (0, 255, 0), -1, 1, 0);
                if (dis_from_pane_>0.6f && pix_pcl_.x != 0 && pix_pcl_.y != 0 && pix_pcl_.z !=0)
                {
                    cv::circle(img, cv::Point(i, j), 2, (0, 0, 255), -1, 1, 0);
                    cv::Point mask_point = cv::Point(i, j);
                    cluster_points_pix_top_uv.push_back(mask_point);
                    pix_pcl_top.push_back(pix_pcl_);
                }
            }
        }

        if (!cluster_points_pix_top_uv.empty())
        {
            std::vector<int> left_cluster_points_pix_top;
            std::vector<int> right_cluster_points_pix_top;
            std::vector<int> mid_cluster_points_pix_top;
            std::vector<sl::float3> pix_pcl_top_convex_hull;

            std::vector<cv::Point> convex_hull_points_top;
            cv::convexHull(cluster_points_pix_top_uv, convex_hull_points_top, false);
            std::vector<std::vector<cv::Point>> hulls = {convex_hull_points_top};
            cv::drawContours(img, hulls, 0, cv::Scalar(0, 0, 255), 2);

            const cv::Point& point_0 = convex_hull_points_top[0];
            cv::Point left_mid_ = cv::Point(left_mid.x, point_0.y);
            cv::Point right_mid_ = cv::Point(right_mid.x, point_0.y);
            cv::Point mid_img_ = cv::Point(mid_img.x, point_0.y);

            // cv::Point IMG_Left = cv::Point(10, 10);
            // cv::circle(img, IMG_Left, 8, (255, 0, 255), -1, 1, 0);
            // cv::Point IMG_Right = cv::Point(img.cols-20, img.rows-20);
            // cv::circle(img, IMG_Right, 8, (255, 0, 255), -1, 1, 0);

            // int diff_m0 = euclidean_distance_twoarg_pix(mid_img, point_0);
            // std::cout<<" <---diff_m0 --> "<<diff_m0<<std::endl;
            // int diff_mr = euclidean_distance_twoarg_pix(mid_img_, right_mid_);
            
            
            // std::cout<<" <---diff_mr --> "<<diff_mr<<std::endl;

            cv::circle(img, left_mid, 8, (0, 0, 255), -1, 1, 0);
            cv::circle(img, right_mid_, 8, (0, 0, 255), -1, 1, 0);
            cv::circle(img, mid_img, 8, (0, 0, 255), -1, 1, 0);
            cv::circle(img, point_0, 12, (0, 0, 255), -1, 1, 0);

            for (int i = 0; i < convex_hull_points_top.size(); ++i) 
            {
                // Get the current point in the convex hull
                const cv::Point& point = convex_hull_points_top[i];
                cv::circle(img, point, 5, cv::Scalar(0, 255, 0), -1, cv::LINE_AA, 0);

                auto it = std::find(cluster_points_pix_top_uv.begin(), cluster_points_pix_top_uv.end(), point);
                if (it != cluster_points_pix_top_uv.end()) 
                {
                    int index = std::distance(cluster_points_pix_top_uv.begin(), it);
                    // std::cout<<"euclidean_distance_twoarg_pix(l_up, point) --> "<<euclidean_distance_twoarg_pix(l_up, point)<<std::endl;
                    if(euclidean_distance_twoarg_pix(l_up, point) > 0)
                    {
                        left_cluster_points_pix_top.push_back(euclidean_distance_twoarg_pix(l_up, point));
                    }
                    right_cluster_points_pix_top.push_back(euclidean_distance_twoarg_pix(right_mid_, point));
                    mid_cluster_points_pix_top.push_back(euclidean_distance_twoarg_pix(mid_img, point));
                    pix_pcl_top_convex_hull.push_back(pix_pcl_top.at(index));
                }

            }
            if (!left_cluster_points_pix_top.empty() && !right_cluster_points_pix_top.empty() && !mid_cluster_points_pix_top.empty())
            {
                
                int im_x = img.cols/2;
                int im_y = img.rows/2;
                cv::circle(img, cv::Point(im_x, im_y), 2, (0, 0, 255), -1, 1, 0);
                sl::float3 pix_pcl_img_cm = pix_pcl(im_x,im_y, point_cloud, img);
                
                auto [pix_pcl_top_min_left, min_idx_l] = min_cluster_pix_mask_find(img, left_cluster_points_pix_top, convex_hull_points_top, left_up, pix_pcl_top_convex_hull, pix_pcl_img_cm);
                auto [pix_pcl_top_min_right, min_idx_r] = min_cluster_pix_mask_find(img, right_cluster_points_pix_top, convex_hull_points_top, left_up, pix_pcl_top_convex_hull, pix_pcl_img_cm);
                auto [pix_pcl_top_min_mid, min_idx_m] = min_cluster_pix_mask_find(img, mid_cluster_points_pix_top, convex_hull_points_top, left_up, pix_pcl_top_convex_hull, pix_pcl_img_cm);

                float d_lm = euclidean_distance_twoarg(pix_pcl_top_min_left, pix_pcl_top_min_mid);
                auto[lm_m_h, lm_m_v] = find_HV_slop(pix_pcl_top_min_left.x, pix_pcl_top_min_left.y, pix_pcl_top_min_mid.x, pix_pcl_top_min_mid.y);

                float d_rm = euclidean_distance_twoarg(pix_pcl_top_min_right, pix_pcl_top_min_mid);
                std::cout<<" <---diff_rm --> "<<d_rm<<std::endl;
                auto[rm_m_h, rm_m_v] = find_HV_slop(pix_pcl_top_min_right.x, pix_pcl_top_min_right.y, pix_pcl_top_min_mid.x, pix_pcl_top_min_mid.y);

                std::vector<sl::float3> bbx_3d_;
                std::vector<sl::float3> bbx_2d_;
                bbx_2d_.push_back(pix_pcl_top_min_left);
                bbx_2d_.push_back(pix_pcl_top_min_right);
                bbx_2d_.push_back(pix_pcl_top_min_mid);

                // std::cout<<" <---diff_lm --> "<<d_lm<<std::endl;
                

                // if(d_rm > 1.5)
                // {
                //     std::cout<<" bed is in left side of the camera "<<std::endl;
                //     send_static_target_tf(now, camera_frame, "pix_pcl_top_min_left", pix_pcl_top_min_left.x, pix_pcl_top_min_left.y, 0.0, 0.0, 0.0, lm_m_h);
                //     send_static_target_tf(now, camera_frame, "pix_pcl_top_min_right", pix_pcl_top_min_right.x, pix_pcl_top_min_right.y, 0.0, 0.0, 0.0, lm_m_h);
                //     send_static_target_tf(now, camera_frame, "pix_pcl_top_min_mid", pix_pcl_top_min_mid.x, pix_pcl_top_min_mid.y, 0.0, 0.0, 0.0, lm_m_h);

                //     auto[gpx_f, gpy_f] = WP_creation(lm_m_v, pix_pcl_top_min_left.x, pix_pcl_top_min_left.y, 4, d_rm);
                //     send_static_target_tf(now, camera_frame, "pix_pcl_top_min_left_f", gpx_f, gpy_f, 0.0, 0.0, 0.0, lm_m_h);
                //     sl::float3 pix_pcl_top_min_left_f = {gpx_f, gpy_f, 0.0};

                //     bbx_2d_.push_back(pix_pcl_top_min_left_f);
                //     sl::float3 midpoint_2d = midpoint_of_rectangle(bbx_2d_);

                //     bbx_3d_.push_back(pix_pcl_top_min_left); //0
                //     bbx_3d_.push_back(pix_pcl_top_min_left_f); //1
                //     bbx_3d_.push_back(pix_pcl_top_min_right); //2
                //     bbx_3d_.push_back(pix_pcl_top_min_mid); //3
                //     bbx_3d_.push_back(nall_strc); //4
                //     bbx_3d_.push_back(nall_strc); //5
                //     bbx_3d_.push_back(nall_strc); //6
                //     bbx_3d_.push_back(nall_strc); //7
                //     camera_laser_pub(now, bbx_3d_);


                //     send_static_target_tf(now, camera_frame, "bbx_2d_mid_point", midpoint_2d.x, midpoint_2d.y, 0.0, 0.0, 0.0, lm_m_h);
                //     auto [wp_l_xf, wp_l_yf] = WP_creation(lm_m_h, midpoint_2d.x, midpoint_2d.y, 1, k);
                //     send_static_target_tf(now, camera_frame, "wp_lf", wp_l_xf, wp_l_yf, 0.0, 0.0, 0.0, M_PI+lm_m_h);
                //     wp_vec(now, map_frame, "wp_lf");
                //     wp_vec_pub(now, tf2_pose_vec_);


                //     auto gen_wp_feed_msg = std_msgs::msg::UInt8();
                //     gen_wp_feed_msg.data = 1;
                //     path_generation_status_pub->publish(gen_wp_feed_msg);
                // }

                // if(d_rm < 1.1)
                // {
                //     std::cout<<" bed is in right side of the camera "<<std::endl;
                //     send_static_target_tf(now, camera_frame, "pix_pcl_top_min_left", pix_pcl_top_min_left.x, pix_pcl_top_min_left.y, 0.0, 0.0, 0.0, rm_m_h);
                //     send_static_target_tf(now, camera_frame, "pix_pcl_top_min_right", pix_pcl_top_min_right.x, pix_pcl_top_min_right.y, 0.0, 0.0, 0.0, rm_m_h);
                //     send_static_target_tf(now, camera_frame, "pix_pcl_top_min_mid", pix_pcl_top_min_mid.x, pix_pcl_top_min_mid.y, 0.0, 0.0, 0.0, rm_m_h);
                    
                //     auto[gpx_f, gpy_f] = WP_creation(rm_m_v, pix_pcl_top_min_right.x, pix_pcl_top_min_right.y, 1, d_lm);
                //     send_static_target_tf(now, camera_frame, "pix_pcl_top_min_right_f", gpx_f, gpy_f, 0.0, 0.0, 0.0, rm_m_h);
                //     sl::float3 pix_pcl_top_min_right_f = {gpx_f, gpy_f, 0.0};

                //     bbx_2d_.push_back(pix_pcl_top_min_right_f);
                //     sl::float3 midpoint_2d = midpoint_of_rectangle(bbx_2d_);

                //     bbx_3d_.push_back(pix_pcl_top_min_mid); //0
                //     bbx_3d_.push_back(pix_pcl_top_min_left); //1
                //     bbx_3d_.push_back(pix_pcl_top_min_right_f); //2
                //     bbx_3d_.push_back(pix_pcl_top_min_right); //3
                //     bbx_3d_.push_back(nall_strc); //4
                //     bbx_3d_.push_back(nall_strc); //5
                //     bbx_3d_.push_back(nall_strc); //6
                //     bbx_3d_.push_back(nall_strc); //7
                //     camera_laser_pub(now, bbx_3d_);

                //     send_static_target_tf(now, camera_frame, "bbx_2d_mid_point", midpoint_2d.x, midpoint_2d.y, 0.0, 0.0, 0.0, rm_m_h);
                //     auto [wp_r_xf, wp_r_yf] = WP_creation(rm_m_h, midpoint_2d.x, midpoint_2d.y, 1, k);
                //     send_static_target_tf(now, camera_frame, "wp_rf", wp_r_xf, wp_r_yf, 0.0, 0.0, 0.0, M_PI+rm_m_h);
                //     wp_vec(now, map_frame, "wp_rf");
                //     wp_vec_pub(now, tf2_pose_vec_);

                //     auto gen_wp_feed_msg = std_msgs::msg::UInt8();
                //     gen_wp_feed_msg.data = 1;
                //     path_generation_status_pub->publish(gen_wp_feed_msg);

                // }
                

                if(h_active)
                {
                    std::cout<<" <--- long side of the bed --> "<<cnt<<std::endl;
                    auto [mid_l_lf, mid_r_rf, midpoint_2d, gth_m_h, gth_m_v] = gen_wp_all(now, pix_pcl_top_min_left, pix_pcl_top_min_right, bed_W, 4);
                    auto [wp_l_xf, wp_l_yf] = WP_creation(gth_m_v, midpoint_2d.x, midpoint_2d.y, 1, k+0.3);
                    send_static_target_tf(now, camera_frame, "wp_mf", wp_l_xf, wp_l_yf, 0.0, 0.0, 0.0, M_PI+gth_m_v);
                    wp_vec(now, map_frame, "wp_mf");
                    wp_vec_pub(now, tf2_pose_vec_);

                    auto gen_wp_feed_msg = std_msgs::msg::UInt8();
                    gen_wp_feed_msg.data = 1;
                    path_generation_status_pub->publish(gen_wp_feed_msg);

                }

                else if(v_active)
                {
                    std::cout<<" <--- foot side of the bed ---> "<<cnt<<std::endl;
                    auto [mid_l_lf, mid_r_rf, midpoint_2d, gth_m_h, gth_m_v] = gen_wp_all(now, pix_pcl_top_min_left, pix_pcl_top_min_right, bed_L, 4);

                    auto [wp_l_xf, wp_l_yf] = WP_creation(gth_m_h, midpoint_2d.x, midpoint_2d.y, 2, k);
                    send_static_target_tf(now, camera_frame, "wp_l", wp_l_xf, wp_l_yf, 0.0, 0.0, 0.0, gth_m_h);

                    auto [wp_r_xf, wp_r_yf] = WP_creation(gth_m_h, midpoint_2d.x, midpoint_2d.y, 1, k);
                    send_static_target_tf(now, camera_frame, "wp_r", wp_r_xf, wp_r_yf, 0.0, 0.0, 0.0, M_PI+gth_m_h);
                    
                    wp_vec(now, map_frame, "wp_l");
                    wp_vec(now, map_frame, "wp_r");

                    wp_vec_pub(now, tf2_pose_vec_);

                    auto gen_wp_feed_msg = std_msgs::msg::UInt8();
                    gen_wp_feed_msg.data = 1;
                    path_generation_status_pub->publish(gen_wp_feed_msg);

                }

                // std::cout<<"gamma_left ---> "<<gamma_left<<" gamma_mid  ---> "<<gamma_mid<<" gamma_right  ---> "<<gamma_right<<std::endl;

                else if(d_rm > 1.5)
                {
                    std::cout<<" bed is in left side of the camera "<<std::endl;
                    auto [mid_l_lf, mid_r_rf, midpoint_2d, gth_m_h, gth_m_v] = gen_wp_all(now, pix_pcl_top_min_right, pix_pcl_top_min_mid, bed_W, 1);
                    auto [wp_l_xf, wp_l_yf] = WP_creation(gth_m_v, midpoint_2d.x, midpoint_2d.y, 4, k);
                    send_static_target_tf(now, camera_frame, "wp_lf", wp_l_xf, wp_l_yf, 0.0, 0.0, 0.0, gth_m_v);
                    wp_vec(now, map_frame, "wp_lf");
                    wp_vec_pub(now, tf2_pose_vec_);

                    auto gen_wp_feed_msg = std_msgs::msg::UInt8();
                    gen_wp_feed_msg.data = 1;
                    path_generation_status_pub->publish(gen_wp_feed_msg);
                }

                else if(d_rm < 1.1)
                {
                    std::cout<<" bed is in right side of the camera "<<std::endl;
                    auto [mid_l_lf, mid_r_rf, midpoint_2d, gth_m_h, gth_m_v] = gen_wp_all(now, pix_pcl_top_min_left, pix_pcl_top_min_mid, bed_W, 4);
                    auto [wp_r_xf, wp_r_yf] = WP_creation(gth_m_v, midpoint_2d.x, midpoint_2d.y, 1, k);
                    send_static_target_tf(now, camera_frame, "wp_rf", wp_r_xf, wp_r_yf, 0.0, 0.0, 0.0, M_PI+gth_m_v);
                    wp_vec(now, map_frame, "wp_rf");
                    wp_vec_pub(now, tf2_pose_vec_);

                    auto gen_wp_feed_msg = std_msgs::msg::UInt8();
                    gen_wp_feed_msg.data = 1;
                    path_generation_status_pub->publish(gen_wp_feed_msg);
                    
                }

                // else if(type == 1)
                // {
                //     auto [mid_l_lf, mid_r_rf, midpoint_2d, gth_m_h, gth_m_v] = gen_wp_all(now, pix_pcl_top_min_left, pix_pcl_top_min_mid, bed_L, 4);
                //     auto [wp_l_xf, wp_l_yf] = WP_creation(gth_m_h, midpoint_2d.x, midpoint_2d.y, 1, k);
                //     send_static_target_tf(now, camera_frame, "wp_lf", wp_l_xf, wp_l_yf, 0.0, 0.0, 0.0, M_PI+gth_m_h);
                //     wp_vec(now, map_frame, "wp_lf");

                //     auto [mid_l_lf_, mid_r_rf_, midpoint_2d_, gth_m_h_, gth_m_v_] = gen_wp_all(now, pix_pcl_top_min_right, pix_pcl_top_min_mid, bed_L, 1);
                //     auto [wp_r_xf, wp_r_yf] = WP_creation(gth_m_h, midpoint_2d_.x, midpoint_2d_.y, 1, k);
                //     send_static_target_tf(now, camera_frame, "wp_rf", wp_r_xf, wp_r_yf, 0.0, 0.0, 0.0, M_PI+gth_m_h);
                //     wp_vec(now, map_frame, "wp_rf");

                //     wp_vec_pub(now, tf2_pose_vec_);

                //     auto gen_wp_feed_msg = std_msgs::msg::UInt8();
                //     gen_wp_feed_msg.data = 1;
                //     path_generation_status_pub->publish(gen_wp_feed_msg);
                    
                // }

                else
                {
                    auto gen_wp_feed_msg = std_msgs::msg::UInt8();
                    gen_wp_feed_msg.data = 0;
                    path_generation_status_pub->publish(gen_wp_feed_msg);
                }

                cnt+=1;
                    

                    

                // }


                
                

                // send_static_target_tf(now, camera_frame, "pix_pcl_top_min_left", pix_pcl_top_min_left.x, pix_pcl_top_min_left.y, pix_pcl_top_min_left.z, 0.0, 0.0, 0.0);
                // send_static_target_tf(now, camera_frame, "pix_pcl_top_min_right", pix_pcl_top_min_right.x, pix_pcl_top_min_right.y, pix_pcl_top_min_right.z, 0.0, 0.0, 0.0);
                // send_static_target_tf(now, camera_frame, "pix_pcl_top_min_mid", pix_pcl_top_min_mid.x, pix_pcl_top_min_mid.y, pix_pcl_top_min_mid.z, 0.0, 0.0, 0.0);
                // std::cout<<"dis_lr ----> "<<dis_lr
                // <<" dis_lm ----> "<<dis_lm
                // <<" dis_rm ----> "<<dis_rm<<std::endl;
                

            }
            

        }

    }

}
