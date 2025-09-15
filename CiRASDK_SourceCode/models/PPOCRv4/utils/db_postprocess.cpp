#include "db_postprocess.hpp"
#include <opencv2/imgproc/imgproc.hpp>
#include <cmath>
#include <numeric>
#include "clipper.hpp"

DBPostProcess::DBPostProcess(double thresh, double box_thresh, int max_candidates, double unclip_ratio, bool use_dilation, const std::string& score_mode)
    : thresh_(thresh),
      box_thresh_(box_thresh),
      max_candidates_(max_candidates),
      unclip_ratio_(unclip_ratio),
      use_dilation_(use_dilation),
      score_mode_(score_mode)
{}

void DBPostProcess::boxes_from_bitmap(
    const cv::Mat &pred,
    const cv::Mat &bitmap,
    int dest_width, int dest_height,
    std::vector<float>& box_scores,
    std::vector<std::vector<cv::Point>>& boxes){

    boxes.clear();
    box_scores.clear();
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;

    // --- Apply dilation if requested ---
    cv::Mat proc_bitmap = bitmap.clone();
    if (use_dilation_) {
        int ksize = 3; // Kernel size (3x3)
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(ksize, ksize));
        cv::dilate(proc_bitmap, proc_bitmap, kernel, cv::Point(-1, -1), 1);
    }

    // 1. Find contours from (possibly dilated) binarized prediction
    cv::findContours(proc_bitmap, contours, hierarchy, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);

    for (size_t i = 0; i < contours.size() && boxes.size() < static_cast<size_t>(max_candidates_); ++i) {
        // 2. Minimum area rect for each contour
        cv::RotatedRect min_rect = cv::minAreaRect(contours[i]);

        // Filter out tiny boxes (optional, you can tune this!)
        if (std::min(min_rect.size.width, min_rect.size.height) < 3)
            continue;

        // 3. Box score (choose fast or slow as set)
        double score = 0.0;
        if (score_mode_ == "slow")
            score = box_score_slow(pred, contours[i]);
        else
            score = box_score_fast(pred, contours[i]);
        if (score < box_thresh_)
            continue;

        // 4. Get box points (4 points) from min area rect
        cv::Point2f rect_points[4];
        min_rect.points(rect_points);
        std::vector<cv::Point> box(rect_points, rect_points + 4);

        // 5. Optionally unclip (expand) the box, returns possibly complex polygon
        std::vector<cv::Point> unclipped = unclip(box);
        if (unclipped.empty())
            continue;

        // --- Ensure final box is always 4 points: minAreaRect on unclipped polygon (PaddleOCR official style) ---
        if (unclipped.size() > 4) {
            cv::RotatedRect rect = cv::minAreaRect(unclipped);
            cv::Point2f pts[4];
            rect.points(pts);
            unclipped = std::vector<cv::Point>(pts, pts + 4);
        }

        // 6. Rescale coordinates to destination image size
        double scale_x = double(dest_width) / bitmap.cols;
        double scale_y = double(dest_height) / bitmap.rows;
        for (auto &pt : unclipped) {
            pt.x = static_cast<int>(std::round(pt.x * scale_x));
            pt.y = static_cast<int>(std::round(pt.y * scale_y));
        }

        boxes.push_back(unclipped);
        box_scores.push_back(static_cast<float>(score));
    }
    //return boxes;
}

std::vector<cv::Point> DBPostProcess::unclip(const std::vector<cv::Point>& box)
{
    using namespace ClipperLib;
    Path poly;
    for (const auto& pt : box)
        poly.emplace_back(IntPoint(pt.x, pt.y));

    // Compute area and perimeter (length) for offset distance
    double area = std::fabs(Area(poly));
    double length = 0.0;
    for (size_t i = 0; i < poly.size(); ++i) {
        auto& p1 = poly[i];
        auto& p2 = poly[(i + 1) % poly.size()];
        double dx = static_cast<double>(p1.X - p2.X);
        double dy = static_cast<double>(p1.Y - p2.Y);
        length += std::sqrt(dx * dx + dy * dy);
    }
    if (length < 1e-5) return std::vector<cv::Point>();

    double distance = area * unclip_ratio_ / length;

    // Perform polygon offset (expand)
    ClipperOffset offset;
    Paths solution;
    offset.AddPath(poly, jtRound, etClosedPolygon);
    offset.Execute(solution, distance);

    // Convert result back to vector<cv::Point>
    std::vector<cv::Point> result;
    if (!solution.empty()) {
        for (const auto& ipt : solution[0])
            result.emplace_back(static_cast<int>(ipt.X), static_cast<int>(ipt.Y));
    }
    return result;
}

double DBPostProcess::box_score_fast(const cv::Mat &bitmap, const std::vector<cv::Point>& box)
{
    // Find axis-aligned bounding rectangle (AABB)
    cv::Rect bbox = cv::boundingRect(box);

    // Crop the probability map to this rectangle
    cv::Mat crop = bitmap(bbox);

    // Compute the mean value within the rectangle (includes some background!)
    cv::Scalar meanVal = cv::mean(crop);

    return meanVal[0];
}

double DBPostProcess::box_score_slow(const cv::Mat &bitmap, const std::vector<cv::Point>& contour)
{
    // Create a mask of the same size as bitmap (8-bit, single channel)
    cv::Mat mask = cv::Mat::zeros(bitmap.size(), CV_8UC1);
    std::vector<std::vector<cv::Point>> contours = { contour };
    cv::fillPoly(mask, contours, cv::Scalar(1));

    // Compute the mean of the probability map inside the mask
    cv::Scalar meanVal = cv::mean(bitmap, mask);
    return meanVal[0];
}
