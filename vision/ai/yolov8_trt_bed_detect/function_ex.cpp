#include "path_gen.cpp"

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
                rclcpp::Time now)
{
    for (size_t i = 0; i < img_batch.size(); i++) 
    {
        auto &res = res_batch[i]; 
        cv::Mat img = img_batch[i];

        if(res.size() > 0)
        {
            for (auto &it : res)
            {
                if(it.conf > conf_score_value)
                {
                    auto [center_im, mid_P_bed, mid_Q_bed, LV_dis, RV_dis, LH_dis, RH_dis] = distance_cluster_lrf(img, point_cloud, plane_eq, it);
                    float d_cm = euclidean_distance_onearg(center_im);
                    auto [gpxP, gpzP] = find_global_PQR(center_im, mid_P_bed);
                    auto [gpxR, gpzR] = find_global_PQR(center_im, mid_Q_bed);
                    double gpxM = (gpxP+gpxR)/2;
                    double gpzM = (gpzP+gpzR)/2;

                    float size_bed = euclidean_distance_twoarg(mid_P_bed, mid_Q_bed);

                    send_static_target_tf(static_target_tf, now, camera_frame, bed_frameP, gpzP, gpxP, 0.0, 0.0, 0.0, 0.0);
                    send_static_target_tf(static_target_tf,now, camera_frame, bed_frameM, gpzM, gpxM, 0.0, 0.0, 0.0, 0.0);
                    send_static_target_tf(static_target_tf, now, camera_frame, bed_frameR, gpzR, gpxR, 0.0, 0.0, 0.0, 0.0);

                    auto [gth_m_h, gth_m_v] = find_HV_slop(gpzR, gpxR, gpzP, gpxP);

                    auto [gpxPP, gpyPP] = WP_creation(gth_m_h, gpzP, gpxP, 1, k_h);
                    auto [gpxRR, gpyRR] = WP_creation(gth_m_h, gpzR, gpxR, 2, k_h);

                    auto [gpxPP_v, gpyPP_v] = WP_creation(gth_m_v, gpxPP, gpyPP, 1, k_v);
                    auto [gpxPP_vf, gpyPP_vf] = WP_creation(gth_m_v, gpxPP, gpyPP, 3, k_v);

                    auto [gpxRR_v, gpyRR_v] = WP_creation(gth_m_v, gpxRR, gpyRR, 2, k_v);
                    auto [gpxRR_vf, gpyRR_vf] = WP_creation(gth_m_v, gpxRR, gpyRR, 4, k_v);

                    auto [gpxPP_vb, gpyPP_vb] = WP_creation(gth_m_v, gpzP, gpxP, 3, k_v);
                    auto [gpxRR_vb, gpyRR_vb] = WP_creation(gth_m_v, gpzR, gpxR, 4, k_v);
                    auto [gpxPP_Mb, gpyPP_Mb] = WP_creation(gth_m_v, gpzM, gpxM, 3, k_v);
                    
                    std::vector<geometry_msgs::msg::PoseStamped> tf2_pose_vec;

                    std::cout << "LH_dis --> "<<LH_dis<<"\n"
                    <<" LV_dis --> "<<LV_dis<<"\n"
                    <<" RH_dis --> "<<RH_dis<<"\n"
                    <<" RV_dis --> "<<RV_dis<<"\n"
                    << "gpxRR_v --> "<<gpxRR_v<<"\n"
                    <<" gpyRR_v --> "<<gpyRR_v<<"\n"
                    <<" d_cm --> "<<d_cm<<"\n"
                    <<" size_bed --> "<<size_bed<<std::endl;



                    if(LH_dis > distance_thresh)
                    {
                        if(LV_dis > distance_thresh)
                        {
                            try
                            {

                                auto bed_msg = std_msgs::msg::UInt8();
                                bed_msg.data = 1;
                                bed_detection_nav2_call_pub->publish(bed_msg);

                                std::vector<geometry_msgs::msg::PoseStamped> LH_tf2_pose_vec;
                                std::cout << "Generate waypoints in LH"<< std::endl;
                                send_static_target_tf(static_target_tf, now, camera_frame, bed_framePP_v, gpxPP_v, gpyPP_v, 0.0, 0.0, 0.0, -gth_m_h);
                                send_static_target_tf(static_target_tf, now, camera_frame, bed_framePP_vf, std::abs(gpxPP_v-gpxPP_vf)/font_dis, gpyPP_vf, 0.0, 0.0, 0.0, 0.0);
    
                                geometry_msgs::msg::TransformStamped tf_sub_b =  tf_sub(map_frame, bed_framePP_v);
                                geometry_msgs::msg::PoseStamped tf2_pose_b = tf2_pose(now, tf_sub_b);

                                geometry_msgs::msg::TransformStamped tf_sub_f =  tf_sub(map_frame, bed_framePP_vf);
                                geometry_msgs::msg::PoseStamped tf2_pose_f = tf2_pose(now, tf_sub_f);

                                tf2_pose_vec.push_back(tf2_pose_f);
                                tf2_pose_vec.push_back(tf2_pose_b);
                                tf2_pose_vec.push_back(tf2_pose_f);
                                // LH_pose_stamped_pub->publish(tf2_pose_b);

                                LH_tf2_pose_vec.push_back(tf2_pose_f);
                                LH_tf2_pose_vec.push_back(tf2_pose_b);
                                LH_tf2_pose_vec.push_back(tf2_pose_f);

                                nav_msgs::msg::Path path_msg;
                                path_msg.header.stamp = now;
                                path_msg.poses = LH_tf2_pose_vec;
                                left_pose_array_pub->publish(path_msg);
                                
                            }
                            catch (...)
                            {

                            }
                        }
                    }

                    if(RH_dis > distance_thresh)
                    {
                        if(RV_dis > distance_thresh)
                        {
                            try
                            {

                                auto bed_msg = std_msgs::msg::UInt8();
                                bed_msg.data = 1;
                                bed_detection_nav2_call_pub->publish(bed_msg);

                                std::vector<geometry_msgs::msg::PoseStamped> RH_tf2_pose_vec;
                                std::cout << "Generate waypoints in RH"<< std::endl;
                                send_static_target_tf(static_target_tf, now, camera_frame, bed_frameRR_v, gpxRR_v, gpyRR_v, 0.0, 0.0, 0.0, gth_m_h);
                                send_static_target_tf(static_target_tf, now, camera_frame, bed_frameRR_vf, std::abs(gpxRR_v - gpxRR_vf)/font_dis, gpyRR_vf, 0.0, 0.0, 0.0, 0.0);

                                geometry_msgs::msg::TransformStamped tf_sub_b =  tf_sub(map_frame, bed_frameRR_v);
                                geometry_msgs::msg::PoseStamped tf2_pose_b = tf2_pose(now, tf_sub_b);

                                geometry_msgs::msg::TransformStamped tf_sub_f =  tf_sub(map_frame, bed_frameRR_vf);
                                geometry_msgs::msg::PoseStamped tf2_pose_f = tf2_pose(now, tf_sub_f);

                                tf2_pose_vec.push_back(tf2_pose_f);
                                tf2_pose_vec.push_back(tf2_pose_b);
                                tf2_pose_vec.push_back(tf2_pose_f);
                                // RH_pose_stamped_pub->publish(tf2_pose_);

                                RH_tf2_pose_vec.push_back(tf2_pose_f);
                                RH_tf2_pose_vec.push_back(tf2_pose_b);
                                RH_tf2_pose_vec.push_back(tf2_pose_f);

                                nav_msgs::msg::Path path_msg;
                                path_msg.header.stamp = now;
                                path_msg.poses = RH_tf2_pose_vec;
                                right_pose_array_pub->publish(path_msg);

                            }
                            catch (...)
                            {
                                
                            }
                            
                            

                        }
                    }

                    if(d_cm > dis_img)
                    {
                        try
                        {
                            auto bed_msg = std_msgs::msg::UInt8();
                            bed_msg.data = 1;
                            bed_detection_nav2_call_pub->publish(bed_msg);

                            std::vector<geometry_msgs::msg::PoseStamped> FH_tf2_pose_vec;
                            std::cout << "Generate waypoints in FH"<< std::endl;
                            send_static_target_tf(static_target_tf, now, camera_frame, bed_frameRR_Mb, gpxPP_Mb, gpyPP_Mb, 0.0, 0.0, 0.0, gth_m_v);
                            geometry_msgs::msg::TransformStamped tf_sub_ =  tf_sub(map_frame, bed_frameRR_Mb);
                            geometry_msgs::msg::PoseStamped tf2_pose_ = tf2_pose(now, tf_sub_);
                            // tf2_pose_vec.push_back(tf2_pose_);
                            // FH_pose_stamped_pub->publish(tf2_pose_);
                            FH_tf2_pose_vec.push_back(tf2_pose_);
                            nav_msgs::msg::Path path_msg;
                            path_msg.header.stamp = now;
                            path_msg.poses = FH_tf2_pose_vec;
                            font_pose_array_pub->publish(path_msg);

                            

                        }
                        catch (...)
                        {
                            
                        }
                            


                    }

                    if (!tf2_pose_vec.empty())
                    {
                        // for (const auto &pose : tf2_pose_vec) 
                        // {
                        //     std::cout << "PoseStamped: " << std::endl;
                        //     std::cout << "Header:" << std::endl;
                        //     std::cout << "  Stamp: " << pose.header.stamp.sec << "s " << pose.header.stamp.nanosec << "ns" << std::endl;
                        //     std::cout << "  Frame ID: " << pose.header.frame_id << std::endl;
                        //     std::cout << "Pose:" << std::endl;
                        //     std::cout << "  Position: (" << pose.pose.position.x << ", "
                        //             << pose.pose.position.y << ", " << pose.pose.position.z << ")" << std::endl;
                        //     std::cout << "  Orientation: (" << pose.pose.orientation.x << ", "
                        //             << pose.pose.orientation.y << ", " << pose.pose.orientation.z << ", "
                        //             << pose.pose.orientation.w << ")" << std::endl;
                        // }

                        // for (const auto& pose_stamped : tf2_pose_vec) 
                        // {
                        //     startNavToPose(pose_stamped);
                        //     exit(0);
                        // }

                        

                        updateWpNavigationMarkers(viz_f, tf2_pose_vec);
                        exit(0);
                        

                        

                    }
                    



                    
                    std::cout << "----------------------------------------------------------------------------"<< std::endl;


                }
                


            }
                
                
                   
        }
        // cv::imshow("detected images", img);
        
    }
        


}
