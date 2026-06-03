#include "ros_utils.h"

std::tuple<float, float> AlgoUtils::find_global_PQR(const sl::float3& img_frame, const sl::float3& bed_PQR)
{
    float d_img_frame = euclidean_distance_onearg(img_frame);
    float d_bed_PQR = euclidean_distance_onearg(bed_PQR);
    float d_img_frame_bed_PQR = euclidean_distance_twoarg(img_frame, bed_PQR);
    float gamma_PQR_C_im = Cosin(d_img_frame, d_bed_PQR, d_img_frame_bed_PQR);
    float gammma_corr = gamma_sign_correction(gamma_PQR_C_im, bed_PQR.x);
    float gpx = d_bed_PQR*cos(gammma_corr);
    float gpy = d_bed_PQR*sin(gammma_corr);
    return std::make_tuple(gpx, gpy);
}

std::tuple<double, double, double> AlgoUtils::find_gamma_lmr(const sl::float3 &pix_pcl_img_cm, const sl::float3 &pix_pcl_top_min_left ,const sl::float3 &pix_pcl_top_min_mid, const sl::float3 &pix_pcl_top_min_right)
{
    float d_img_frame = euclidean_distance_onearg(pix_pcl_img_cm);
    float d_left_frame = euclidean_distance_onearg(pix_pcl_top_min_left);
    float d_mid_frame = euclidean_distance_onearg(pix_pcl_top_min_mid);
    float d_right_frame = euclidean_distance_onearg(pix_pcl_top_min_right);

    float d_img_frame_left = euclidean_distance_twoarg(pix_pcl_img_cm, pix_pcl_top_min_left);
    float d_img_frame_right = euclidean_distance_twoarg(pix_pcl_img_cm, pix_pcl_top_min_right);
    float d_img_frame_mid = euclidean_distance_twoarg(pix_pcl_img_cm, pix_pcl_top_min_mid);

    double gamma_left = gamma_sign_correction(Cosin(d_img_frame, d_left_frame, d_img_frame_left), pix_pcl_top_min_left.y);
    double gamma_mid = gamma_sign_correction(Cosin(d_img_frame, d_mid_frame, d_img_frame_mid), pix_pcl_top_min_mid.y);
    double gamma_right = gamma_sign_correction(Cosin(d_img_frame, d_right_frame, d_img_frame_right), pix_pcl_top_min_right.y);
    return std::make_tuple(gamma_left, gamma_mid, gamma_right);
}

std::tuple<sl::float3, sl::float3, int> AlgoUtils::d_lmr_logic(std::vector<sl::float3> &pix_pcl_top_min_lmr, std::vector<double> &gamma_lmr)
{
    double gamma_l = gamma_lmr.at(0);
    double gamma_m = gamma_lmr.at(1);
    double gamma_r = gamma_lmr.at(2);
    // std::cout<<"gamma_left --->"<<gamma_l<<" gamma_mid ---> "<<gamma_m<<" gamma_right ---> "<<gamma_r<<std::endl;
    // for left cluster

    // for right cluster
    if(gamma_l<0.0 && gamma_m<0.0 && gamma_r<0.0)
    {
        sl::float3 mid_mlr;
        mid_mlr.x = pix_pcl_top_min_lmr.at(2).x;
        mid_mlr.y = pix_pcl_top_min_lmr.at(1).y;
        mid_mlr.z = pix_pcl_top_min_lmr.at(1).z;
        return std::make_tuple(mid_mlr, pix_pcl_top_min_lmr.at(2), 2);
    }
    
    // if(gamma_l<0.0 && gamma_m<0.0 && gamma_r>0.0)
    // {
    //     sl::float3 mid_mlr;
    //     mid_mlr.x = pix_pcl_top_min_lmr.at(2).x;
    //     mid_mlr.y = pix_pcl_top_min_lmr.at(1).y;
    //     mid_mlr.z = pix_pcl_top_min_lmr.at(1).z;
    //     return std::make_tuple(mid_mlr, pix_pcl_top_min_lmr.at(2), 2);
    // }

    if(gamma_l<0.0 && gamma_m>0.0 && gamma_r<0.0)
    {
        sl::float3 mid_mlr;
        mid_mlr.x = pix_pcl_top_min_lmr.at(2).x;
        mid_mlr.y = pix_pcl_top_min_lmr.at(1).y;
        mid_mlr.z = pix_pcl_top_min_lmr.at(1).z;
        return std::make_tuple(mid_mlr, pix_pcl_top_min_lmr.at(2), 1);
    }

    if(gamma_l<0.0 && gamma_m>0.0 && gamma_r>0.0)
    {
        sl::float3 mid_mlr;
        mid_mlr.x = pix_pcl_top_min_lmr.at(2).x;
        mid_mlr.y = pix_pcl_top_min_lmr.at(1).y;
        mid_mlr.z = pix_pcl_top_min_lmr.at(1).z;
        return std::make_tuple(mid_mlr, pix_pcl_top_min_lmr.at(2), 2);
    }

    // if(gamma_l>0.0 && gamma_m<0.0 && gamma_r<0.0)
    // {
    //     sl::float3 mid_mlr;
    //     mid_mlr.x = pix_pcl_top_min_lmr.at(2).x;
    //     mid_mlr.y = pix_pcl_top_min_lmr.at(1).y;
    //     mid_mlr.z = pix_pcl_top_min_lmr.at(1).z;
    //     return std::make_tuple(mid_mlr, pix_pcl_top_min_lmr.at(2), 0);
    // }


    if(gamma_l>0.0 && gamma_m<0.0 && gamma_r>0.0)
    {
        sl::float3 mid_mlr;
        mid_mlr.x = pix_pcl_top_min_lmr.at(2).x;
        mid_mlr.y = pix_pcl_top_min_lmr.at(1).y;
        mid_mlr.z = pix_pcl_top_min_lmr.at(1).z;
        return std::make_tuple(mid_mlr, pix_pcl_top_min_lmr.at(2), 1);
    }

    if(gamma_l>0.0 && gamma_m>0.0 && gamma_r<0.0)
    {
        sl::float3 mid_mlr;
        mid_mlr.x = pix_pcl_top_min_lmr.at(2).x;
        mid_mlr.y = pix_pcl_top_min_lmr.at(1).y;
        mid_mlr.z = pix_pcl_top_min_lmr.at(1).z;
        return std::make_tuple(mid_mlr, pix_pcl_top_min_lmr.at(2), 0);
    }

    if(gamma_l>0.0 && gamma_m>0.0 && gamma_r>0.0)
    {
        sl::float3 mid_mlr;
        mid_mlr.x = pix_pcl_top_min_lmr.at(2).x;
        mid_mlr.y = pix_pcl_top_min_lmr.at(1).y;
        mid_mlr.z = pix_pcl_top_min_lmr.at(1).z;
        return std::make_tuple(mid_mlr, pix_pcl_top_min_lmr.at(2), 0);
    }

    

    else
    {

        return std::make_tuple(nall_strc, nall_strc, 4);
    }

}

