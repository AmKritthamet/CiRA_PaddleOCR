#ifndef ThreadPPOCRv4_HPP
#define ThreadPPOCRv4_HPP

#include <ros/ros.h>
#include <cira_lib_bernoulli/general/GlobalData.hpp>
#include <QThread>
#include <QJsonObject>
#include <QJsonArray>
#include <chrono>

#include <opencv2/opencv.hpp>

// Include your NCNN model
#include "PPOCRv4/utils/ncnn_ppocrv4_detector.hpp"
#include "PPOCRv4/utils/ncnn_ppocrv4_recognizer.hpp"

class ThreadPPOCRv4 : public QThread
{
  Q_OBJECT
public:

  QString name = "PPOCRv4";

  QJsonObject payload_js_data;
  QJsonObject output_js_data;
  QJsonObject param_js_data;

  bool isUseImage = true;
  cv::Mat mat_im;

  bool isHaveError = false;

  // Detector instance
  NCNN_PPOCRv4_Detector* detector = nullptr;
  // Recognizer instance
  NCNN_PPOCRv4_Recognizer* recognizer = nullptr;

  ThreadPPOCRv4();
  ~ThreadPPOCRv4();

  void run() override;

private:
  std::string model_param_path;
  std::string model_bin_path;
  std::string rec_model_param_path;
  std::string rec_model_bin_path;
  std::string dict_path;

  // Helper function to crop text regions from detection boxes
  std::vector<cv::Mat> crop_text_regions(const cv::Mat& img, const std::vector<std::vector<cv::Point>>& boxes);
};

#endif // ThreadPPOCRv4_HPP
