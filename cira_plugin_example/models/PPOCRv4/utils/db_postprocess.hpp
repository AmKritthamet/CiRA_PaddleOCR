#ifndef DB_POSTPROCESS_HPP
#define DB_POSTPROCESS_HPP

#include <opencv2/core/core.hpp>
#include <vector>
#include <string>

class DBPostProcess {
public:
    DBPostProcess(double thresh, double box_thresh, int max_candidates, double unclip_ratio, bool use_dilation, const std::string& score_mode);
    void boxes_from_bitmap(
        const cv::Mat& pred,
        const cv::Mat& bitmap,
        int dest_width, int dest_height,
        std::vector<float>& box_scores,
        std::vector<std::vector<cv::Point>>& boxes);

private:
    double thresh_;
    double box_thresh_;
    int max_candidates_;
    double unclip_ratio_;
    bool use_dilation_;
    std::string score_mode_;

    std::vector<cv::Point> unclip(const std::vector<cv::Point>& box);
    double box_score_fast(const cv::Mat &bitmap, const std::vector<cv::Point>& box);
    double box_score_slow(const cv::Mat &bitmap, const std::vector<cv::Point>& contour);
};

#endif // DB_POSTPROCESS_HPP
