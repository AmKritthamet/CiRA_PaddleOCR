#include "ncnn_ppocrv4_recognizer.hpp"
#include <opencv2/imgproc/imgproc.hpp>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <cctype>
#include <iostream>

NCNN_PPOCRv4_Recognizer::NCNN_PPOCRv4_Recognizer(
    const std::string &param_path,
    const std::string &bin_path,
    const std::string &char_dict_path,
    bool remove_space,
    bool remove_punctuation)
    : param_path_(param_path),
      bin_path_(bin_path),
      char_dict_path_(char_dict_path),
      remove_space_(remove_space),
      remove_punctuation_(remove_punctuation)
{
    // Load dictionary during construction
    if (!char_dict_path.empty()) {
        load_dict(char_dict_path);
    } else {
        load_default_dict();
    }
}

bool NCNN_PPOCRv4_Recognizer::load_model()
{
    int ret = net_.load_param(param_path_.c_str());
    if (ret != 0) {
        std::cerr << "[ERROR] Failed to load recognizer param: " << param_path_ << std::endl;
        return false;
    }

    ret = net_.load_model(bin_path_.c_str());
    if (ret != 0) {
        std::cerr << "[ERROR] Failed to load recognizer bin: " << bin_path_ << std::endl;
        return false;
    }

    return true;
}

void NCNN_PPOCRv4_Recognizer::load_dict(const std::string &path)
{
    character_.clear();
    char_to_idx_.clear();
    idx_to_char_.clear();

    std::ifstream ifs(path);
    if (!ifs.is_open()) {
        std::cerr << "[WARNING] Dictionary file not found: " << path << std::endl;
        std::cerr << "[WARNING] Using default English dictionary" << std::endl;
        load_default_dict();
        return;
    }

    // Read from file
    std::string line;
    while (std::getline(ifs, line)) {
        if (!line.empty()) {
            // Remove any trailing \r or \n
            line.erase(std::remove(line.begin(), line.end(), '\r'), line.end());
            line.erase(std::remove(line.begin(), line.end(), '\n'), line.end());
            character_.push_back(line);
        }
    }
    ifs.close();

    // Ensure blank is at index 0
    auto blank_it = std::find(character_.begin(), character_.end(), "blank");
    if (blank_it != character_.end() && blank_it != character_.begin()) {
        character_.erase(blank_it);
    }
    if (character_.empty() || character_[0] != "blank") {
        character_.insert(character_.begin(), "blank");
    }

    // Fix for PP-OCRv4 English model expecting space at index 96
    if (character_.size() == 96) {
        character_.push_back(" "); // Add space at index 96
    }

    // Build mappings
    for (size_t i = 0; i < character_.size(); ++i) {
        char_to_idx_[character_[i]] = static_cast<int>(i);
        idx_to_char_[static_cast<int>(i)] = character_[i];
    }

    print_debug_info();
}

void NCNN_PPOCRv4_Recognizer::load_default_dict()
{
    character_.clear();
    char_to_idx_.clear();
    idx_to_char_.clear();

    // Add blank token first
    character_.push_back("blank");

    // Add space early
    character_.push_back(" ");

    // Numbers 0-9
    for (int i = 0; i < 10; ++i) {
        character_.push_back(std::to_string(i));
    }

    // Symbols in order from typical en_dict.txt
    std::vector<char> symbols = {
        ':', ';', '<', '=', '>', '?', '@',
        '[', '\\', ']', '^', '_', '`',
        '{', '|', '}', '~', '!', '"', '#', '$', '%', '&',
        '\'', '(', ')', '*', '+', ',', '-', '.', '/'
    };

    for (char c : symbols) {
        character_.push_back(std::string(1, c));
    }

    // Uppercase letters A-Z
    for (char c = 'A'; c <= 'Z'; ++c) {
        character_.push_back(std::string(1, c));
    }

    // Lowercase letters a-z
    for (char c = 'a'; c <= 'z'; ++c) {
        character_.push_back(std::string(1, c));
    }

    // Ensure we have 97 characters for PP-OCRv4 English
    while (character_.size() < 97) {
        character_.push_back("<UNK>");
    }

    // Build mappings
    for (size_t i = 0; i < character_.size(); ++i) {
        char_to_idx_[character_[i]] = static_cast<int>(i);
        idx_to_char_[static_cast<int>(i)] = character_[i];
    }
}

ncnn::Mat NCNN_PPOCRv4_Recognizer::resize_norm_img(const cv::Mat &img)
{
    int imgC = rec_image_shape_[0];
    int imgH = rec_image_shape_[1];
    int imgW = rec_image_shape_[2];

    int h = img.rows;
    int w = img.cols;
    float ratio = static_cast<float>(w) / static_cast<float>(h);

    int resized_w = static_cast<int>(imgH * ratio);
    resized_w = std::min(resized_w, imgW);
    resized_w = std::max(16, resized_w);

    cv::Mat resized;
    cv::resize(img, resized, cv::Size(resized_w, imgH));

    // Pad if needed
    if (resized_w < imgW) {
        int pad = imgW - resized_w;
        cv::copyMakeBorder(resized, resized, 0, 0, 0, pad, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
    }

    // Convert to NCNN Mat and normalize
    ncnn::Mat in = ncnn::Mat::from_pixels(resized.data, ncnn::Mat::PIXEL_BGR, imgW, imgH);
    in.substract_mean_normalize(mean_vals_, norm_vals_);

    return in;
}

std::pair<std::string, float> NCNN_PPOCRv4_Recognizer::ctc_decode(
    const std::vector<int> &preds_idx,
    const std::vector<float> &preds_prob)
{
    std::vector<std::string> char_list;
    std::vector<float> conf_list;

    int prev_idx = -1;

    for (size_t i = 0; i < preds_idx.size(); ++i) {
        int cur_idx = preds_idx[i];
        float cur_prob = preds_prob[i];

        // Skip blank token (index 0)
        if (cur_idx == 0) {
            prev_idx = cur_idx;
            continue;
        }

        // Skip consecutive duplicates
        if (prev_idx != -1 && cur_idx == prev_idx) {
            continue;
        }

        // Get character
        std::string ch;
        if (idx_to_char_.find(cur_idx) != idx_to_char_.end()) {
            ch = idx_to_char_[cur_idx];
        } else if (cur_idx < static_cast<int>(character_.size())) {
            ch = character_[cur_idx];
        } else {
            // Unknown character, skip
            prev_idx = cur_idx;
            continue;
        }

        char_list.push_back(ch);
        conf_list.push_back(cur_prob);
        prev_idx = cur_idx;
    }

    // Concatenate characters
    std::string text;
    for (const auto& ch : char_list) {
        text += ch;
    }

    // Calculate average confidence
    float confidence = 0.0f;
    if (!conf_list.empty()) {
        confidence = std::accumulate(conf_list.begin(), conf_list.end(), 0.0f) / conf_list.size();
    }

    return std::make_pair(text, confidence);
}

std::string NCNN_PPOCRv4_Recognizer::post_process_text(const std::string &text) const
{
    std::string result = text;

    if (remove_space_) {
        result.erase(std::remove(result.begin(), result.end(), ' '), result.end());
    }

    if (remove_punctuation_) {
        result.erase(
            std::remove_if(result.begin(), result.end(),
                [](unsigned char c) { return std::ispunct(c); }),
            result.end()
        );
    }

    return result;
}

std::pair<std::string, float> NCNN_PPOCRv4_Recognizer::recognize_single(const cv::Mat &img)
{
    if (img.empty()) {
        return std::make_pair("", 0.0f);
    }

    try {
        // Preprocess
        ncnn::Mat in = resize_norm_img(img);

        // Run inference
        ncnn::Extractor ex = net_.create_extractor();
        ex.input("in0", in);
        ncnn::Mat out;
        ex.extract("out0", out);

        // Get predictions
        int seq_len = out.h;
        int num_classes = out.w;

        std::vector<int> preds_idx;
        std::vector<float> preds_prob;

        // Find argmax for each timestep
        for (int t = 0; t < seq_len; ++t) {
            float max_prob = -1.0f;
            int max_idx = 0;

            const float* scores = out.row(t);
            for (int c = 0; c < num_classes; ++c) {
                if (scores[c] > max_prob) {
                    max_prob = scores[c];
                    max_idx = c;
                }
            }

            preds_idx.push_back(max_idx);
            preds_prob.push_back(max_prob);
        }

        // Decode
        auto result = ctc_decode(preds_idx, preds_prob);

        // Apply post-processing
        result.first = post_process_text(result.first);

        return result;

    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Recognition failed: " << e.what() << std::endl;
        return std::make_pair("", 0.0f);
    }
}

std::vector<std::pair<std::string, float>> NCNN_PPOCRv4_Recognizer::recognize(
    const std::vector<cv::Mat> &img_list)
{
    std::vector<std::pair<std::string, float>> results;
    results.reserve(img_list.size());

    for (const auto& img : img_list) {
        results.push_back(recognize_single(img));
    }

    return results;
}

void NCNN_PPOCRv4_Recognizer::print_debug_info() const
{
    static bool already_printed = false;
    if (already_printed) return;
    already_printed = true;

    std::cout << "===== Recognizer Info =====" << std::endl;
    std::cout << "Dictionary size: " << character_.size() << std::endl;
    std::cout << "First 5 chars: ";
    for (size_t i = 0; i < std::min(size_t(5), character_.size()); ++i) {
        std::cout << "'" << character_[i] << "' ";
    }
    std::cout << std::endl;

    // Check for space
    auto space_it = std::find(character_.begin(), character_.end(), " ");
    if (space_it != character_.end()) {
        std::cout << "Space found at index: " << std::distance(character_.begin(), space_it) << std::endl;
    }

    // Check index 96
    if (idx_to_char_.find(96) != idx_to_char_.end()) {
        std::cout << "Character at index 96: '" << idx_to_char_.at(96) << "'" << std::endl;
    }
    std::cout << "==========================" << std::endl;
}
