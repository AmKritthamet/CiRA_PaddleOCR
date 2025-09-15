#ifndef NCNN_PPOCRV4_DETECTOR_HPP
#define NCNN_PPOCRV4_DETECTOR_HPP

#include <opencv2/core/core.hpp>
#include <ncnn/net.h>
#include <string>
#include <vector>
#include "db_postprocess.hpp"

class NCNN_PPOCRv4_Detector
{
public:
    // Constructor: pass paths to ncnn param/bin files and detection postprocess config
    NCNN_PPOCRv4_Detector(const std::string &param_path,
                          const std::string &bin_path,
                          double db_thresh = 0.3,
                          double box_thresh = 0.6,
                          int max_candidates = 1000,
                          double unclip_ratio = 1.5,
                          bool use_dilation = false,
                          const std::string &score_mode = "fast");

    // Load and initialize the model
    bool load_model();

    // Main detection pipeline: input image, outputs boxes and scores
    void detect(const cv::Mat &image,
                std::vector<std::vector<cv::Point>> &boxes,
                std::vector<float> &scores);

    // You can add more functions to get/set parameters dynamically if needed

private:
    ncnn::Net net_;
    std::string param_path_;
    std::string bin_path_;

    int input_size_w_ = 960; // You can expose as parameter (PP-OCRv4 default is 960)
    int input_size_h_ = 960;

    // Preprocessing
    float mean_vals_[3] = {0.485f * 255.0f, 0.456f * 255.0f, 0.406f * 255.0f};
    float norm_vals_[3] = {1.0f / 0.229f / 255.0f, 1.0f / 0.224f / 255.0f, 1.0f / 0.225f / 255.0f};

    // Postprocess object
    std::string score_mode_;
    DBPostProcess postprocess_;

    // Thresholds etc. (these match constructor parameters)
    double db_thresh_;
    double box_thresh_;
    int max_candidates_;
    double unclip_ratio_;
    bool use_dilation_;


    // Helper for preprocessing
    void preprocess(const cv::Mat &image, ncnn::Mat &in, float &scale_w, float &scale_h) const;

    void resize_and_pad32(const cv::Mat &src, cv::Mat &dst, float &ratio_h, float &ratio_w) const;
        std::vector<cv::Point> order_points_clockwise(const std::vector<cv::Point>& pts);
        std::vector<cv::Point> clip_det_res(const std::vector<cv::Point>& pts, int img_height, int img_width);

};

#endif // NCNN_PPOCRV4_DETECTOR_HPP
