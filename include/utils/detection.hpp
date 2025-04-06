#pragma once

#include <opencv2/opencv.hpp>

namespace Framer {
/**
 * @brief Struct to represent a detection.
 */
struct Detection {
    cv::Rect box;
    float conf{};
    int classId{};
};
}
