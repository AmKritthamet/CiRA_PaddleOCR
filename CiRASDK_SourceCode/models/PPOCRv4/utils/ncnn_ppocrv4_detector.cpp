#include "ncnn_ppocrv4_detector.hpp"
#include <opencv2/imgproc/imgproc.hpp>
#include <ncnn/net.h>
#include <ncnn/mat.h>
#include <algorithm>
#include <cmath>
#include <numeric>


// --- Helper for sorting boxes by Y, then X ---
static void sort_boxes_yx(std::vector<std::vector<cv::Point>>& boxes, std::vector<float>& scores) {
    std::vector<size_t> indices(boxes.size());
    std::iota(indices.begin(), indices.end(), 0);

    std::sort(indices.begin(), indices.end(), [&](size_t a, size_t b) {
        float ax = 0, ay = 0, bx = 0, by = 0;
        //for (const auto& pt : boxes[a]) { ax += pt.x; ay += pt.y; }
        for (const auto& pt : boxes[a]) {
            ax += static_cast<float>(pt.x);
            ay += static_cast<float>(pt.y);
        }
        //for (const auto& pt : boxes[b]) { bx += pt.x; by += pt.y; }
        for (const auto& pt : boxes[b]) {
            bx += static_cast<float>(pt.x);
            by += static_cast<float>(pt.y);
        }
        ax /= static_cast<float>(boxes[a].size());
        ay /= static_cast<float>(boxes[a].size());
        bx /= static_cast<float>(boxes[b].size());
        by /= static_cast<float>(boxes[b].size());



        // Y is primary sort, X is secondary
        if (std::abs(ay - by) > 1e-3f) return ay < by;

        return ax < bx;
    });

    // Reorder boxes and scores
    std::vector<std::vector<cv::Point>> boxes_sorted;
    std::vector<float> scores_sorted;
    for (auto idx : indices) {
        boxes_sorted.push_back(boxes[idx]);
        scores_sorted.push_back(scores[idx]);
    }
    boxes = boxes_sorted;
    scores = scores_sorted;
}

NCNN_PPOCRv4_Detector::NCNN_PPOCRv4_Detector(
    const std::string &param_path,
    const std::string &bin_path,
    double db_thresh,
    double box_thresh,
    int max_candidates,
    double unclip_ratio,
    bool use_dilation,
    const std::string &score_mode)
    : param_path_(param_path),
      bin_path_(bin_path),
      score_mode_(score_mode),
      postprocess_(db_thresh, box_thresh, max_candidates, unclip_ratio, use_dilation, score_mode),
      db_thresh_(db_thresh),
      box_thresh_(box_thresh),
      max_candidates_(max_candidates),
      unclip_ratio_(unclip_ratio),
      use_dilation_(use_dilation)
{
}

bool NCNN_PPOCRv4_Detector::load_model()
{
    int ret = net_.load_param(param_path_.c_str());
    if (ret != 0) return false;
    ret = net_.load_model(bin_path_.c_str());
    return (ret == 0);
}

void NCNN_PPOCRv4_Detector::resize_and_pad32(const cv::Mat &src, cv::Mat &dst, float &ratio_h, float &ratio_w) const
{
    int h = src.rows, w = src.cols;
    float max_side_len = 960.0;
    float ratio = 1.0;
    if (static_cast<float>(std::max(h, w)) > max_side_len) {
        ratio = max_side_len / static_cast<float>(std::max(h, w));
    }
    int resize_h = static_cast<int>(static_cast<float>(h) * ratio);
    int resize_w = static_cast<int>(static_cast<float>(w) * ratio);

    // Round to nearest multiple of 32, minimum 32
    resize_h = std::max(static_cast<int>(std::round(resize_h / 32.0) * 32), 32);
    resize_w = std::max(static_cast<int>(std::round(resize_w / 32.0) * 32), 32);

    cv::resize(src, dst, cv::Size(resize_w, resize_h));
    ratio_h = static_cast<float>(resize_h) / static_cast<float>(h);
    ratio_w = static_cast<float>(resize_w) / static_cast<float>(w);

}

void NCNN_PPOCRv4_Detector::preprocess(const cv::Mat &image, ncnn::Mat &in, float &ratio_h, float &ratio_w) const
{
    // Resize with padding as in Python
    cv::Mat resized;
    resize_and_pad32(image, resized, ratio_h, ratio_w);

    // Convert BGR to RGB if your model expects RGB
    // cv::cvtColor(resized, resized, cv::COLOR_BGR2RGB);

    // Convert to float32 and normalize (mean/std as per PaddleOCR)
    in = ncnn::Mat::from_pixels(resized.data, ncnn::Mat::PIXEL_BGR, resized.cols, resized.rows);

    float mean_vals[3] = {0.485f * 255.0f, 0.456f * 255.0f, 0.406f * 255.0f};
    float norm_vals[3] = {1.0f / (0.229f * 255.0f), 1.0f / (0.224f * 255.0f), 1.0f / (0.225f * 255.0f)};
    in.substract_mean_normalize(mean_vals, norm_vals);
}

void NCNN_PPOCRv4_Detector::detect(const cv::Mat &image,
                                   std::vector<std::vector<cv::Point>> &boxes,
                                   std::vector<float> &scores)
{
    boxes.clear();
    scores.clear();

    // 1. Preprocess input (resize/pad/normalize)
    ncnn::Mat in;
    float ratio_h, ratio_w;
    preprocess(image, in, ratio_h, ratio_w);

    // 2. Run inference
    ncnn::Extractor ex = net_.create_extractor();
    ex.input("in0", in);         // Use in0
    ncnn::Mat out;
    ex.extract("out0", out);     // Use out0

    // 3. Convert NCNN Mat to cv::Mat (float32)
    int out_h = out.h;
    int out_w = out.w;
    cv::Mat pred_map(out_h, out_w, CV_32FC1, static_cast<void*>(out.data));

    // 4. Binarize to bitmap (thresh)
    cv::Mat bitmap;
    cv::threshold(pred_map, bitmap, db_thresh_, 1, cv::THRESH_BINARY);
    bitmap.convertTo(bitmap, CV_8UC1); // 0 or 1

    // 5. Run postprocess (outputs boxes and scores)
    postprocess_.boxes_from_bitmap(pred_map, bitmap, image.cols, image.rows, scores, boxes);

    // 6. Filter, order, and clip boxes
    std::vector<std::vector<cv::Point>> boxes_filtered;
    std::vector<float> scores_filtered;
    for (size_t i = 0; i < boxes.size(); ++i) {
        // Box order (PaddleOCR-style)
        std::vector<cv::Point> ordered = order_points_clockwise(boxes[i]);
        std::vector<cv::Point> clipped = clip_det_res(ordered, image.rows, image.cols);
        int rect_width = static_cast<int>(cv::norm(clipped[0] - clipped[1]));
        int rect_height = static_cast<int>(cv::norm(clipped[0] - clipped[3]));
        if (rect_width <= 3 || rect_height <= 3)
            continue;
        boxes_filtered.push_back(clipped);
        scores_filtered.push_back(scores[i]);
    }

    // Step 9: Sort boxes by Y then X (top to bottom, left to right)
    sort_boxes_yx(boxes_filtered, scores_filtered);

    // Output
    boxes = boxes_filtered;
    scores = scores_filtered;
}

// Helper: reorder box points clockwise (as in PaddleOCR)
std::vector<cv::Point> NCNN_PPOCRv4_Detector::order_points_clockwise(const std::vector<cv::Point>& pts)
{
    // This assumes input is 4 points, as from minAreaRect
    std::vector<cv::Point> sorted = pts;
    // Sort by x
    std::sort(sorted.begin(), sorted.end(), [](const cv::Point& a, const cv::Point& b) {
        return a.x < b.x;
    });
    std::vector<cv::Point> leftMost(sorted.begin(), sorted.begin() + 2);
    std::vector<cv::Point> rightMost(sorted.begin() + 2, sorted.end());
    std::sort(leftMost.begin(), leftMost.end(), [](const cv::Point& a, const cv::Point& b) {
        return a.y < b.y;
    });
    std::sort(rightMost.begin(), rightMost.end(), [](const cv::Point& a, const cv::Point& b) {
        return a.y < b.y;
    });
    cv::Point tl = leftMost[0], bl = leftMost[1], tr = rightMost[0], br = rightMost[1];
    return {tl, tr, br, bl};
}

// Helper: clip box to image boundaries
std::vector<cv::Point> NCNN_PPOCRv4_Detector::clip_det_res(const std::vector<cv::Point>& pts, int img_height, int img_width)
{
    std::vector<cv::Point> clipped = pts;
    for (auto& p : clipped) {
        p.x = std::min(std::max(p.x, 0), img_width - 1);
        p.y = std::min(std::max(p.y, 0), img_height - 1);
    }
    return clipped;
}
