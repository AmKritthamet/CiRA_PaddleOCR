#include "ThreadPPOCRv4.hpp"
#include <QDebug>
#include <QDir>
#include <QFile>
#include <ros/package.h>

ThreadPPOCRv4::ThreadPPOCRv4() {
    static bool models_loaded = false;
    static bool first_debug = true;

    // Get the package path
    std::string pkg_path = ros::package::getPath("cira_plugin_example");

    if (pkg_path.empty()) {
        qDebug() << "[Error] Could not find package 'cira_plugin_example'";
        isHaveError = true;
        return;
    }

    // Set model paths
    model_param_path = pkg_path + "/models/PPOCRv4/models/ch_PP-OCRv4_det_infer.ncnn.param";
    model_bin_path = pkg_path + "/models/PPOCRv4/models/ch_PP-OCRv4_det_infer.ncnn.bin";
    rec_model_param_path = pkg_path + "/models/PPOCRv4/models/en_PP-OCRv4_rec_infer.ncnn.param";
    rec_model_bin_path = pkg_path + "/models/PPOCRv4/models/en_PP-OCRv4_rec_infer.ncnn.bin";
    dict_path = pkg_path + "/models/PPOCRv4/models/en_dict.txt";

    // Only verify files on first load
    if (first_debug) {
        first_debug = false;

        // Verify detector files
        if (!QFile::exists(QString::fromStdString(model_param_path))) {
            qDebug() << "[Error] Detector param file not found at:" << QString::fromStdString(model_param_path);
            isHaveError = true;
            return;
        } else {
            qDebug() << "[Detector] Param file found at:" << QString::fromStdString(model_param_path);
        }

        if (!QFile::exists(QString::fromStdString(model_bin_path))) {
            qDebug() << "[Error] Detector bin file not found at:" << QString::fromStdString(model_bin_path);
            isHaveError = true;
            return;
        } else {
            qDebug() << "[Detector] Bin file found at:" << QString::fromStdString(model_bin_path);
        }

        // Verify recognizer files
        if (!QFile::exists(QString::fromStdString(rec_model_param_path))) {
            qDebug() << "[Error] Recognizer param file not found at:" << QString::fromStdString(rec_model_param_path);
            isHaveError = true;
            return;
        } else {
            qDebug() << "[Recognizer] Param file found at:" << QString::fromStdString(rec_model_param_path);
        }

        if (!QFile::exists(QString::fromStdString(rec_model_bin_path))) {
            qDebug() << "[Error] Recognizer bin file not found at:" << QString::fromStdString(rec_model_bin_path);
            isHaveError = true;
            return;
        } else {
            qDebug() << "[Recognizer] Bin file found at:" << QString::fromStdString(rec_model_bin_path);
        }

        // Check dictionary file
        if (!QFile::exists(QString::fromStdString(dict_path))) {
            qDebug() << "[Warning] Dictionary file not found at:" << QString::fromStdString(dict_path);
            qDebug() << "[Warning] Will use default English dictionary";
        } else {
            qDebug() << "[Dictionary] File found at:" << QString::fromStdString(dict_path);
        }
    }

    // Always need to create new instances for each node
    detector = new NCNN_PPOCRv4_Detector(
        model_param_path,
        model_bin_path,
        0.3, 0.6, 1000, 1.5, false, "fast"
    );

    if (!detector->load_model()) {
        qDebug() << "[Error] Failed to load PP-OCRv4 detector model";
        isHaveError = true;
    } else if (!models_loaded) {
        qDebug() << "[Model Loading] PP-OCRv4 detector loaded successfully";
    }

    recognizer = new NCNN_PPOCRv4_Recognizer(
        rec_model_param_path,
        rec_model_bin_path,
        dict_path,
        false,
        false
    );

    if (!recognizer->load_model()) {
        qDebug() << "[Error] Failed to load PP-OCRv4 recognizer model";
        isHaveError = true;
    } else if (!models_loaded) {
        qDebug() << "[Model Loading] PP-OCRv4 recognizer loaded successfully";
        qDebug() << "[Dictionary] Loaded with" << recognizer->get_dict_size() << "characters";
    }

    models_loaded = true;
}

ThreadPPOCRv4::~ThreadPPOCRv4() {
    if (detector) {
        delete detector;
        detector = nullptr;
    }
    if (recognizer) {
        delete recognizer;
        recognizer = nullptr;
    }
}

std::vector<cv::Mat> ThreadPPOCRv4::crop_text_regions(const cv::Mat& img,
                                                      const std::vector<std::vector<cv::Point>>& boxes) {
    std::vector<cv::Mat> cropped_images;

    for (const auto& box : boxes) {
        if (box.size() != 4) continue;

        // Get bounding rectangle
        cv::Rect rect = cv::boundingRect(box);

        // Ensure rect is within image bounds
        rect.x = std::max(0, rect.x);
        rect.y = std::max(0, rect.y);
        rect.width = std::min(rect.width, img.cols - rect.x);
        rect.height = std::min(rect.height, img.rows - rect.y);

        if (rect.width > 0 && rect.height > 0) {
            // For better recognition, use perspective transform
            std::vector<cv::Point2f> src_pts;
            for (const auto& pt : box) {
                src_pts.push_back(cv::Point2f(static_cast<float>(pt.x), static_cast<float>(pt.y)));
            }

            // Calculate width and height
            float width = static_cast<float>(cv::norm(src_pts[0] - src_pts[1]));
            float height = static_cast<float>(cv::norm(src_pts[0] - src_pts[3]));

            // Define destination points
            std::vector<cv::Point2f> dst_pts = {
                cv::Point2f(0, 0),
                cv::Point2f(width, 0),
                cv::Point2f(width, height),
                cv::Point2f(0, height)
            };

            // Get perspective transform
            cv::Mat M = cv::getPerspectiveTransform(src_pts, dst_pts);

            // Warp image
            cv::Mat warped;
            cv::warpPerspective(img, warped, M, cv::Size(static_cast<int>(width), static_cast<int>(height)));

            cropped_images.push_back(warped);
        }
    }

    return cropped_images;
}

void ThreadPPOCRv4::run() {
    isHaveError = false;
    payload_js_data = QJsonObject();
    output_js_data = QJsonObject();

    // Check if detector and recognizer are initialized
    if (!detector || !recognizer || isHaveError) {
        QJsonObject error_jso;
        error_jso["error"] = "Models not loaded. Please check model files in package.";
        payload_js_data[name] = error_jso;
        isHaveError = true;
        return;
    }

    // Check if image is available
    if(isUseImage) {
        if(mat_im.empty()) {
            QJsonObject error_jso;
            error_jso["error"] = "No image";
            payload_js_data[name] = error_jso;
            isHaveError = true;
            return;
        }
    }

    // Get parameters from dialog
    double db_thresh = 0.3;
    double box_thresh = 0.6;
    double unclip_ratio = 1.5;
    bool use_dilation = false;
    std::string score_mode = "fast";
    bool draw_boxes = true;
    bool show_confidence = true;
    bool show_text = false;
    bool remove_space = false;
    bool remove_punctuation = false;
    bool auto_scale = true;
    double text_scale_factor = 1.0;
    bool use_rec_conf_filter = false;
    float rec_conf_thresh = 0.80f;

    // Read parameters from dialog
    if(param_js_data.contains("db_thresh"))
        db_thresh = param_js_data["db_thresh"].toDouble();
    if(param_js_data.contains("box_thresh"))
        box_thresh = param_js_data["box_thresh"].toDouble();
    if(param_js_data.contains("unclip_ratio"))
        unclip_ratio = param_js_data["unclip_ratio"].toDouble();
    if(param_js_data.contains("use_dilation"))
        use_dilation = param_js_data["use_dilation"].toBool();
    if(param_js_data.contains("score_mode"))
        score_mode = (param_js_data["score_mode"].toInt() == 0) ? "fast" : "slow";
    if(param_js_data.contains("draw_boxes"))
        draw_boxes = param_js_data["draw_boxes"].toBool();
    if(param_js_data.contains("show_confidence"))
        show_confidence = param_js_data["show_confidence"].toBool();
    if(param_js_data.contains("show_text"))
        show_text = param_js_data["show_text"].toBool();
    if(param_js_data.contains("remove_space"))
        remove_space = param_js_data["remove_space"].toBool();
    if(param_js_data.contains("remove_punct"))
        remove_punctuation = param_js_data["remove_punct"].toBool();
    if(param_js_data.contains("auto_scale_text"))
        auto_scale = param_js_data["auto_scale_text"].toBool();
    if(param_js_data.contains("text_scale_factor"))
        text_scale_factor = param_js_data["text_scale_factor"].toDouble();
    if(param_js_data.contains("use_rec_conf_filter"))
        use_rec_conf_filter = param_js_data["use_rec_conf_filter"].toBool();
    if(param_js_data.contains("rec_conf_thresh"))
        rec_conf_thresh = static_cast<float>(param_js_data["rec_conf_thresh"].toDouble());

    // Update recognizer post-processing settings
    recognizer->set_remove_space(remove_space);
    recognizer->set_remove_punctuation(remove_punctuation);

    // Recreate detector with new parameters
    if (detector) {
        delete detector;
        detector = new NCNN_PPOCRv4_Detector(
            model_param_path,
            model_bin_path,
            db_thresh,
            box_thresh,
            1000,
            unclip_ratio,
            use_dilation,
            score_mode
        );

        if (!detector->load_model()) {
            QJsonObject error_jso;
            error_jso["error"] = "Failed to reload detector model with new parameters";
            payload_js_data[name] = error_jso;
            isHaveError = true;
            return;
        }
    }

    // Run detection
    std::vector<std::vector<cv::Point>> boxes;
    std::vector<float> scores;

    try {
        auto start_time = std::chrono::high_resolution_clock::now();

        detector->detect(mat_im, boxes, scores);

        auto detect_time = std::chrono::high_resolution_clock::now();
        auto detect_duration = std::chrono::duration_cast<std::chrono::milliseconds>(detect_time - start_time);

        // Recognition part
        std::vector<std::pair<std::string, float>> rec_results;
        std::vector<cv::Mat> cropped_images;

        // Always run recognition if we need to filter by recognition confidence
        if ((show_text || use_rec_conf_filter) && !boxes.empty()) {
            // Crop text regions
            cropped_images = crop_text_regions(mat_im, boxes);

            // Run recognition
            rec_results = recognizer->recognize(cropped_images);
        }

        // Apply recognition confidence filter
        if (use_rec_conf_filter && !rec_results.empty()) {
            std::vector<std::vector<cv::Point>> filtered_boxes;
            std::vector<float> filtered_scores;
            std::vector<std::pair<std::string, float>> filtered_rec_results;

            for (size_t i = 0; i < boxes.size(); ++i) {
                if (i < rec_results.size() && rec_results[i].second >= rec_conf_thresh) {
                    filtered_boxes.push_back(boxes[i]);
                    filtered_scores.push_back(scores[i]);
                    filtered_rec_results.push_back(rec_results[i]);
                }
            }

            // Update with filtered results
            boxes = filtered_boxes;
            scores = filtered_scores;
            rec_results = filtered_rec_results;

            //qDebug() << "[RecConfFilter] Filtered to" << filtered_boxes.size() << "boxes (threshold:" << rec_conf_thresh << ")";
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

        // Prepare data arrays for output
        QJsonArray boxes_array;
        QJsonArray scores_array;
        QJsonArray texts_array;
        QJsonArray rec_scores_array;

        for (size_t i = 0; i < boxes.size(); ++i) {
            QJsonArray box_points;
            for (const auto& pt : boxes[i]) {
                QJsonArray point;
                point.append(pt.x);
                point.append(pt.y);
                box_points.append(point);
            }
            boxes_array.append(box_points);
            scores_array.append(static_cast<double>(scores[i]));

            // Add recognition results if available
            if (show_text && i < rec_results.size()) {
                texts_array.append(QString::fromStdString(rec_results[i].first));
                rec_scores_array.append(static_cast<double>(rec_results[i].second));
            }
        }

        // Draw visualization
        if (!boxes.empty()) {
            // Calculate font scale for text display
            double font_scale;
            int thickness;

            if (auto_scale) {
                double img_diagonal = std::sqrt(mat_im.cols * mat_im.cols + mat_im.rows * mat_im.rows);
                font_scale = (img_diagonal / 1500.0) * text_scale_factor;
                font_scale = std::max(0.3, std::min(font_scale, 3.0));
                thickness = std::max(1, static_cast<int>(font_scale * 3.0));
            } else {
                font_scale = text_scale_factor;
                thickness = std::max(1, static_cast<int>(font_scale * 3.2));
            }

            for (size_t i = 0; i < boxes.size(); ++i) {
                // Draw box if requested
                if (draw_boxes) {
                    std::vector<std::vector<cv::Point>> contours = {boxes[i]};
                    cv::drawContours(mat_im, contours, 0, cv::Scalar(255, 0, 0), 2);
                }

                // Draw recognized text if requested
                if (show_text && i < rec_results.size() && !rec_results[i].first.empty()) {
                    const std::string& text = rec_results[i].first;
                    float rec_conf = rec_results[i].second;

                    // Get top-left corner of the box
                    cv::Point text_pos = boxes[i][0]; // Top-left corner

                    // Adjust position to be slightly above the box
                    text_pos.y -= 5;
                    if (text_pos.y < 20) text_pos.y = 20;

                    // Remove white background drawing - DELETE THESE LINES:
                    // cv::Size text_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX,
                    //                                    font_scale, thickness, nullptr);
                    // cv::rectangle(mat_im,
                    //             cv::Point(text_pos.x - 2, text_pos.y - text_size.height - 2),
                    //             cv::Point(text_pos.x + text_size.width + 2, text_pos.y + 2),
                    //             cv::Scalar(255, 255, 255), -1);

                    // Draw text in purple (same as rec confidence)
                    cv::putText(mat_im, text, text_pos,
                              cv::FONT_HERSHEY_SIMPLEX, font_scale,
                              cv::Scalar(255, 0, 255), thickness); // Purple color

                    // Draw recognition confidence if both confidences are shown
                    if (show_confidence) {
                        cv::Size text_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX,
                                                           font_scale, thickness, nullptr);
                        std::string rec_conf_text = " (" + std::to_string(rec_conf).substr(0, 4) + ")";
                        cv::Point rec_conf_pos(text_pos.x + text_size.width, text_pos.y);
                        cv::putText(mat_im, rec_conf_text, rec_conf_pos,
                                  cv::FONT_HERSHEY_SIMPLEX, font_scale * 0.8,
                                  cv::Scalar(255, 0, 255), thickness); // Purple
                    }
                }

                // Draw detection confidence score
                if (show_confidence && (!show_text || (show_text && draw_boxes))) {
                    // Show detection confidence on the right side of box
                    std::string score_text = std::to_string(scores[i]).substr(0, 4);

                    // Get text size first
                    cv::Size text_size = cv::getTextSize(score_text, cv::FONT_HERSHEY_SIMPLEX,
                                                       font_scale, thickness, nullptr);

                    // Find the middle point of the right edge
                    cv::Point top_right = boxes[i][1];
                    cv::Point bottom_right = boxes[i][2];

                    int mid_x = (top_right.x + bottom_right.x) / 2;
                    int mid_y = (top_right.y + bottom_right.y) / 2;

                    int offset_x = 3;
                    cv::Point text_pos(mid_x + offset_x, mid_y + text_size.height/2);

                    // Check bounds
                    if (text_pos.x + text_size.width > mat_im.cols) {
                        text_pos.x = mid_x - text_size.width - offset_x;
                    }
                    if (text_pos.y - text_size.height < 0) {
                        text_pos.y = text_size.height + 5;
                    }
                    if (text_pos.y > mat_im.rows - 5) {
                        text_pos.y = mat_im.rows - 5;
                    }

                    cv::putText(mat_im, score_text, text_pos,
                              cv::FONT_HERSHEY_SIMPLEX, font_scale,
                              cv::Scalar(255, 0, 0), thickness);
                }
            }
        }

        // Create separate objects for payload and output
        QJsonObject payload_jso;

        // Payload contains only the essential data
        payload_jso["counts"] = static_cast<int>(boxes.size());
        payload_jso["process_time_ms"] = static_cast<int>(total_duration.count());

        // Add all recognized texts to results array
        if (show_text && !rec_results.empty()) {
            QJsonArray results_array;
            for (const auto& rec_result : rec_results) {
                results_array.append(QString::fromStdString(rec_result.first));
            }
            payload_jso["results"] = results_array;
        } else {
            // Empty array if no text recognition was performed
            payload_jso["results"] = QJsonArray();
        }

        // Everything else goes to output for debugging
        QJsonObject output_jso;
        output_jso["boxes"] = boxes_array;
        output_jso["detection_scores"] = scores_array;
        output_jso["detection_time_ms"] = static_cast<int>(detect_duration.count());
        output_jso["total_time_ms"] = static_cast<int>(total_duration.count());
        output_jso["count"] = static_cast<int>(boxes.size());
        output_jso["success"] = true;

        if (show_text) {
            output_jso["texts"] = texts_array;
            output_jso["recognition_scores"] = rec_scores_array;
            output_jso["recognition_time_ms"] = static_cast<int>(total_duration.count() - detect_duration.count());
        }

        output_jso["parameters"] = param_js_data;
        output_jso["msg"] = QString("Detected %1 text regions in %2ms").arg(boxes.size()).arg(total_duration.count());

        payload_js_data[name] = payload_jso;
        output_js_data[name] = output_jso;

    } catch (const std::exception& e) {
        QJsonObject error_jso;
        error_jso["error"] = QString("Processing failed: %1").arg(e.what());
        error_jso["success"] = false;
        payload_js_data[name] = error_jso;
        output_js_data[name] = error_jso;
        isHaveError = true;
        return;
    }
}
