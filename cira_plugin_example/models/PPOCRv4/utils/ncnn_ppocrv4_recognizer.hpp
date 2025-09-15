#ifndef NCNN_PPOCRV4_RECOGNIZER_HPP
#define NCNN_PPOCRV4_RECOGNIZER_HPP

#include <opencv2/core/core.hpp>
#include <ncnn/net.h>
#include <string>
#include <vector>
#include <map>

class NCNN_PPOCRv4_Recognizer
{
public:
    // Constructor with model paths and post-processing options
    NCNN_PPOCRv4_Recognizer(const std::string &param_path,
                            const std::string &bin_path,
                            const std::string &char_dict_path = "",
                            bool remove_space = false,
                            bool remove_punctuation = false);

    // Destructor
    ~NCNN_PPOCRv4_Recognizer() = default;

    // Load the model
    bool load_model();

    // Main recognition function - returns pairs of (text, confidence)
    std::vector<std::pair<std::string, float>> recognize(const std::vector<cv::Mat> &img_list);

    // Single image recognition
    std::pair<std::string, float> recognize_single(const cv::Mat &img);

    // Post-process text based on settings
    std::string post_process_text(const std::string &text) const;

    // Setters for post-processing options
    void set_remove_space(bool remove) { remove_space_ = remove; }
    void set_remove_punctuation(bool remove) { remove_punctuation_ = remove; }

    // Getters
    bool get_remove_space() const { return remove_space_; }
    bool get_remove_punctuation() const { return remove_punctuation_; }
    size_t get_dict_size() const { return character_.size(); }

private:
    // NCNN network
    ncnn::Net net_;

    // Model paths
    std::string param_path_;
    std::string bin_path_;
    std::string char_dict_path_;

    // Character dictionary
    std::vector<std::string> character_;
    std::map<std::string, int> char_to_idx_;
    std::map<int, std::string> idx_to_char_;

    // PP-OCRv4 recognition parameters
    std::vector<int> rec_image_shape_ = {3, 48, 320}; // [C, H, W]
    float mean_vals_[3] = {127.5f, 127.5f, 127.5f};
    float norm_vals_[3] = {1.0f/127.5f, 1.0f/127.5f, 1.0f/127.5f};

    // Post-processing options
    bool remove_space_;
    bool remove_punctuation_;

    // Helper functions
    void load_dict(const std::string &path);
    void load_default_dict();
    ncnn::Mat resize_norm_img(const cv::Mat &img);
    std::pair<std::string, float> ctc_decode(const std::vector<int> &preds_idx,
                                             const std::vector<float> &preds_prob);

    // Debug helper
    void print_debug_info() const;
};

#endif // NCNN_PPOCRV4_RECOGNIZER_HPP
