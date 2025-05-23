/*
    Copyright (c) 2024-2025 Abdalrahman M. Amer
    Copyright (c) 2025 APSS-Official

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
*/
#pragma once

#include "utils/utils.hpp"
#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>

#include <algorithm>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include <utils/segmentation.hpp>
#include <utils/debug.hpp>
#include <utils/scopedtimer.hpp>

// ============================================================================
// Utility Namespace
// ============================================================================
namespace utils {

// Left for now, will see this one later.
inline void letterBox(const cv::Mat &image,
                      cv::Mat &outImage,
                      const cv::Size &newShape,
                      const cv::Scalar &color     = cv::Scalar(114, 114, 114),
                      bool auto_       = true,
                      bool scaleFill   = false,
                      bool scaleUp     = true,
                      int stride       = 32) {
    float r = std::min((float)newShape.height / (float)image.rows,
                       (float)newShape.width  / (float)image.cols);
    if (!scaleUp) {
        r = std::min(r, 1.0f);
    }

    int newW = static_cast<int>(std::round(image.cols * r));
    int newH = static_cast<int>(std::round(image.rows * r));

    int dw = newShape.width  - newW;
    int dh = newShape.height - newH;

    if (auto_) {
        dw = dw % stride;
        dh = dh % stride;
    }
    else if (scaleFill) {
        newW = newShape.width;
        newH = newShape.height;
        dw = 0;
        dh = 0;
    }

    cv::Mat resized;
    cv::resize(image, resized, cv::Size(newW, newH), 0, 0, cv::INTER_LINEAR);

    int top = dh / 2;
    int bottom = dh - top;
    int left = dw / 2;
    int right = dw - left;
    cv::copyMakeBorder(resized, outImage, top, bottom, left, right, cv::BORDER_CONSTANT, color);
}

} // namespace utils

// ============================================================================
// YOLO11Segmentor Class
// ============================================================================
class YOLO11Segmentor {
public:
    YOLO11Segmentor(const std::string &modelPath,
                       const std::string &labelsPath,
                       bool useGPU = false);

    // Main API
    std::vector<Framer::Segmentation> segment(const cv::Mat &image,
                                      float confThreshold = 0.4f,
                                      float iouThreshold  = 0.45f);

    // Draw results
    void drawSegmentationsAndBoxes(cv::Mat &image,
                                   const std::vector<Framer::Segmentation> &results,
                                   float maskAlpha = 0.5f, const float conf_threshold = 0.4f) const;

    void drawSegmentations(cv::Mat &image,
                           const std::vector<Framer::Segmentation> &results,
                           float maskAlpha = 0.5f, const float confThreshold = 0.4f) const;
    // Accessors
    const std::vector<std::string> &getClassNames()  const { return classNames;  }
    const std::vector<cv::Scalar>  &getClassColors() const { return classColors; }

private:
    // Helpers
    cv::Mat preprocess(const cv::Mat &image,
                       float *&blobPtr,
                       std::vector<int64_t> &inputTensorShape);

    std::vector<Framer::Segmentation> postprocess(const cv::Size &origSize,
                                          const cv::Size &letterboxSize,
                                          const std::vector<Ort::Value> &outputs,
                                          float confThreshold,
                                          float iouThreshold);

    void printBenchMarks();

private:
    Ort::Env           env;
    Ort::SessionOptions sessionOptions;
    Ort::Session       session{nullptr};

    bool     isDynamicInputShape{false};
    cv::Size inputImageShape; 

    std::vector<Ort::AllocatedStringPtr> inputNameAllocs;
    std::vector<const char*>             inputNames;
    std::vector<Ort::AllocatedStringPtr> outputNameAllocs;
    std::vector<const char*>             outputNames;

    size_t numInputNodes  = 0;
    size_t numOutputNodes = 0;

    std::vector<std::string> classNames;
    std::vector<cv::Scalar>  classColors;

    mutable std::mutex m_mtx;
    std::priority_queue<std::string> m_benchmarkQueue;
};

inline YOLO11Segmentor::YOLO11Segmentor(const std::string &modelPath,
                                              const std::string &labelsPath,
                                              bool useGPU)
    : env(ORT_LOGGING_LEVEL_WARNING, "YOLOv11Seg") 
{
    sessionOptions.SetIntraOpNumThreads(std::min(6, static_cast<int>(std::thread::hardware_concurrency())));
    sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    std::vector<std::string> providers = Ort::GetAvailableProviders();
    if (useGPU && std::find(providers.begin(), providers.end(), "CUDAExecutionProvider") != providers.end()) {
        OrtCUDAProviderOptions cudaOptions;
        sessionOptions.AppendExecutionProvider_CUDA(cudaOptions);
        INFO_PRINT("[INFO] Using GPU (CUDA) for YOLOv11 Seg inference.\n");
    } else {
        INFO_PRINT("[INFO] Using CPU for YOLOv11 Seg inference.\n");
    }

#ifdef _WIN32
    std::wstring w_modelPath(modelPath.begin(), modelPath.end());
    session = Ort::Session(env, w_modelPath.c_str(), sessionOptions);
#else
    session = Ort::Session(env, modelPath.c_str(), sessionOptions);
#endif

    numInputNodes  = session.GetInputCount();
    numOutputNodes = session.GetOutputCount();

    Ort::AllocatorWithDefaultOptions allocator;

    // Input
    {
        auto inNameAlloc = session.GetInputNameAllocated(0, allocator);
        inputNameAllocs.emplace_back(std::move(inNameAlloc));
        inputNames.push_back(inputNameAllocs.back().get());

        auto inTypeInfo = session.GetInputTypeInfo(0);
        auto inShape    = inTypeInfo.GetTensorTypeAndShapeInfo().GetShape();

        if (inShape.size() == 4) {
            if (inShape[2] == -1 || inShape[3] == -1) {
                isDynamicInputShape = true;
                inputImageShape = cv::Size(640, 640); // Fallback if dynamic
            } else {
                inputImageShape = cv::Size(static_cast<int>(inShape[3]), static_cast<int>(inShape[2]));
            }
        } else {
            throw std::runtime_error("Model input is not 4D! Expect [N, C, H, W].");
        }
    }

    // Outputs
    if (numOutputNodes != 2) {
        throw std::runtime_error("Expected exactly 2 output nodes: output0 and output1.");
    }

    for (size_t i = 0; i < numOutputNodes; ++i) {
        auto outNameAlloc = session.GetOutputNameAllocated(i, allocator);
        outputNameAllocs.emplace_back(std::move(outNameAlloc));
        outputNames.push_back(outputNameAllocs.back().get());
    }

    classNames  = Framer::getClassNames(labelsPath);
    classColors = Framer::generateColors(classNames);

    INFO_PRINT("[INFO] YOLOv11Seg loaded: " << modelPath << std::endl
              << "      Input shape: " << inputImageShape 
              << (isDynamicInputShape ? " (dynamic)" : "") << std::endl
              << "      #Outputs   : " << numOutputNodes << std::endl
              << "      #Classes   : " << classNames.size() << std::endl);
}

inline cv::Mat YOLO11Segmentor::preprocess(const cv::Mat &image,
                                              float *&blobPtr,
                                              std::vector<int64_t> &inputTensorShape)
{
    ScopedTimer timer("preprocess", &m_benchmarkQueue);

    cv::Mat letterboxImage;
    utils::letterBox(image, letterboxImage, inputImageShape,
                     cv::Scalar(114,114,114), /*auto_=*/isDynamicInputShape,
                     /*scaleFill=*/false, /*scaleUp=*/true, /*stride=*/32);

    // Update if dynamic
    inputTensorShape[2] = static_cast<int64_t>(letterboxImage.rows);
    inputTensorShape[3] = static_cast<int64_t>(letterboxImage.cols);

    letterboxImage.convertTo(letterboxImage, CV_32FC3, 1.0f/255.0f);

    size_t size = static_cast<size_t>(letterboxImage.rows) * static_cast<size_t>(letterboxImage.cols) * 3;
    blobPtr = new float[size];

    std::vector<cv::Mat> channels(3);
    for (int c = 0; c < 3; ++c) {
        channels[c] = cv::Mat(letterboxImage.rows, letterboxImage.cols, CV_32FC1,
                              blobPtr + c * (letterboxImage.rows * letterboxImage.cols));
    }
    cv::split(letterboxImage, channels);

    return letterboxImage;
}

std::vector<Framer::Segmentation> YOLO11Segmentor::postprocess(
    const cv::Size &origSize,
    const cv::Size &letterboxSize,
    const std::vector<Ort::Value> &outputs,
    float confThreshold,
    float iouThreshold) 
{
    ScopedTimer timer("postprocess", &m_benchmarkQueue);

    std::vector<Framer::Segmentation> results;

    // Validate outputs size
    if (outputs.size() < 2) {
        throw std::runtime_error("Insufficient outputs from the model. Expected at least 2 outputs.");
    }

    // Extract outputs
    const float* output0_ptr = outputs[0].GetTensorData<float>();
    const float* output1_ptr = outputs[1].GetTensorData<float>();

    // Get shapes
    auto shape0 = outputs[0].GetTensorTypeAndShapeInfo().GetShape(); // [1, 116, num_detections]
    auto shape1 = outputs[1].GetTensorTypeAndShapeInfo().GetShape(); // [1, 32, maskH, maskW]

    if (shape1.size() != 4 || shape1[0] != 1 || shape1[1] != 32)
        throw std::runtime_error("Unexpected output1 shape. Expected [1, 32, maskH, maskW].");

    const size_t num_features = shape0[1]; // e.g 80 class + 4 bbox parms + 32 seg masks = 116 
    const size_t num_detections = shape0[2];

    // Early exit if no detections
    if (num_detections == 0)
    {
        return results;
    }

    const int numClasses = static_cast<int>(num_features - 4 - 32); // Corrected number of classes

    // Validate numClasses
    if (numClasses <= 0)
    {
        throw std::runtime_error("Invalid number of classes.");
    }

    const int numBoxes = static_cast<int>(num_detections);
    const int maskH = static_cast<int>(shape1[2]);
    const int maskW = static_cast<int>(shape1[3]);

    // Constants from model architecture
    constexpr int BOX_OFFSET = 0;
    constexpr int CLASS_CONF_OFFSET = 4;
    const int MASK_COEFF_OFFSET = numClasses + CLASS_CONF_OFFSET;

    // 1. Process prototype masks
    // Store all prototype masks in a vector for easy access
    std::vector<cv::Mat> prototypeMasks;
    prototypeMasks.reserve(32);
    for (int m = 0; m < 32; ++m) {
        // Each mask is maskH x maskW
        cv::Mat proto(maskH, maskW, CV_32F, const_cast<float*>(output1_ptr + m * maskH * maskW));
        prototypeMasks.emplace_back(proto.clone()); // Clone to ensure data integrity
    }

    // 2. Process detections
    std::vector<cv::Rect> boxes;
    boxes.reserve(numBoxes);
    std::vector<float> confidences;
    confidences.reserve(numBoxes);
    std::vector<int> classIds;
    classIds.reserve(numBoxes);
    std::vector<std::vector<float>> maskCoefficientsList;
    maskCoefficientsList.reserve(numBoxes);

    for (int i = 0; i < numBoxes; ++i) {
        // Extract box coordinates
        float xc = output0_ptr[BOX_OFFSET * numBoxes + i];
        float yc = output0_ptr[(BOX_OFFSET + 1) * numBoxes + i];
        float w = output0_ptr[(BOX_OFFSET + 2) * numBoxes + i];
        float h = output0_ptr[(BOX_OFFSET + 3) * numBoxes + i];

        // Convert to xyxy format
        cv::Rect box{
            static_cast<int>(std::round(xc - w / 2.0f)),
            static_cast<int>(std::round(yc - h / 2.0f)),
            static_cast<int>(std::round(w)),
            static_cast<int>(std::round(h))
        };

        // Get class confidence
        float maxConf = 0.0f;
        int classId = -1;
        for (int c = 0; c < numClasses; ++c) {
            float conf = output0_ptr[(CLASS_CONF_OFFSET + c) * numBoxes + i];
            if (conf > maxConf) {
                maxConf = conf;
                classId = c;
            }
        }

        if (maxConf < confThreshold) continue;

        // Store detection
        boxes.push_back(box);
        confidences.push_back(maxConf);
        classIds.push_back(classId);

        // Store mask coefficients
        std::vector<float> maskCoeffs(32);
        for (int m = 0; m < 32; ++m) {
            maskCoeffs[m] = output0_ptr[(MASK_COEFF_OFFSET + m) * numBoxes + i];
        }
        maskCoefficientsList.emplace_back(std::move(maskCoeffs));
    }

    // Early exit if no boxes after confidence threshold
    if (boxes.empty()) {
        return results;
    }

    // 3. Apply NMS
    std::vector<int> nmsIndices;
    Framer::NMSBoxes(boxes, confidences, confThreshold, iouThreshold, nmsIndices);

    if (nmsIndices.empty()) {
        return results;
    }

    // 4. Prepare final results
    results.reserve(nmsIndices.size());

    // Calculate letterbox parameters
    const float gain = std::min(static_cast<float>(letterboxSize.height) / origSize.height,
                                static_cast<float>(letterboxSize.width) / origSize.width);
    const int scaledW = static_cast<int>(origSize.width * gain);
    const int scaledH = static_cast<int>(origSize.height * gain);
    const float padW = (letterboxSize.width - scaledW) / 2.0f;
    const float padH = (letterboxSize.height - scaledH) / 2.0f;

    // Precompute mask scaling factors
    const float maskScaleX = static_cast<float>(maskW) / letterboxSize.width;
    const float maskScaleY = static_cast<float>(maskH) / letterboxSize.height;

    for (const int idx : nmsIndices) {
        Framer::Segmentation seg;
        seg.box = boxes[idx];
        seg.conf = confidences[idx];
        seg.classId = classIds[idx];

        // 5. Scale box to original image
        seg.box = Framer::scaleCoords(letterboxSize, seg.box, origSize, true);

        // 6. Process mask
        const auto& maskCoeffs = maskCoefficientsList[idx];

        // Linear combination of prototype masks
        cv::Mat finalMask = cv::Mat::zeros(maskH, maskW, CV_32F);
        for (int m = 0; m < 32; ++m) {
            finalMask += maskCoeffs[m] * prototypeMasks[m];
        }

        // Apply sigmoid activation
        finalMask = Framer::sigmoid(finalMask);

        // Crop mask to letterbox area with a slight padding to avoid border issues
        int x1 = static_cast<int>(std::round((padW - 0.1f) * maskScaleX));
        int y1 = static_cast<int>(std::round((padH - 0.1f) * maskScaleY));
        int x2 = static_cast<int>(std::round((letterboxSize.width - padW + 0.1f) * maskScaleX));
        int y2 = static_cast<int>(std::round((letterboxSize.height - padH + 0.1f) * maskScaleY));

        // Ensure coordinates are within mask bounds
        x1 = std::max(0, std::min(x1, maskW - 1));
        y1 = std::max(0, std::min(y1, maskH - 1));
        x2 = std::max(x1, std::min(x2, maskW));
        y2 = std::max(y1, std::min(y2, maskH));

        // Handle cases where cropping might result in zero area
        if (x2 <= x1 || y2 <= y1) {
            // Skip this mask as cropping is invalid
            continue;
        }

        cv::Rect cropRect(x1, y1, x2 - x1, y2 - y1);
        cv::Mat croppedMask = finalMask(cropRect).clone(); // Clone to ensure data integrity

        // Resize to original dimensions
        cv::Mat resizedMask;
        cv::resize(croppedMask, resizedMask, origSize, 0, 0, cv::INTER_LINEAR);

        // Threshold and convert to binary
        cv::Mat binaryMask;
        cv::threshold(resizedMask, binaryMask, 0.5, 255.0, cv::THRESH_BINARY);
        binaryMask.convertTo(binaryMask, CV_8U);

        // Crop to bounding box
        cv::Mat finalBinaryMask = cv::Mat::zeros(origSize, CV_8U);
        cv::Rect roi(seg.box.x, seg.box.y, seg.box.width, seg.box.height);
        roi &= cv::Rect(0, 0, binaryMask.cols, binaryMask.rows); // Ensure ROI is within mask
        if (roi.area() > 0) {
            binaryMask(roi).copyTo(finalBinaryMask(roi));
        }

        seg.mask = finalBinaryMask;
        results.push_back(seg);
    }

    return results;
}

inline void YOLO11Segmentor::printBenchMarks()
{
    std::string result;
    while(!m_benchmarkQueue.empty()) {
        auto top = m_benchmarkQueue.top();
        m_benchmarkQueue.pop();
        result = result + top + (m_benchmarkQueue.empty() ? "" : ", ");
    }

    DEBUG_PRINT("Speed: " + result);
}

inline void YOLO11Segmentor::drawSegmentationsAndBoxes(cv::Mat &image,
                                                          const std::vector<Framer::Segmentation> &results,
                                                          float maskAlpha, const float conf_threshold) const
{
    for (const auto &seg : results) {
        if (seg.conf < conf_threshold) {
            continue;
        }
        cv::Scalar color = classColors[seg.classId % classColors.size()];

        // -----------------------------
        // 1. Draw Bounding Box
        // -----------------------------
        cv::rectangle(image,
                      cv::Point(seg.box.x, seg.box.y),
                      cv::Point(seg.box.x + seg.box.width, seg.box.y + seg.box.height),
                      color, 2);

        // -----------------------------
        // 2. Draw Label
        // -----------------------------
        std::string label = classNames[seg.classId] + " " + std::to_string(static_cast<int>(seg.conf * 100)) + "%";
        int baseLine = 0;
        double fontScale = 0.5;
        int thickness = 1;
        cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, fontScale, thickness, &baseLine);
        int top = std::max(seg.box.y, labelSize.height + 5);
        cv::rectangle(image,
                      cv::Point(seg.box.x, top - labelSize.height - 5),
                      cv::Point(seg.box.x + labelSize.width + 5, top),
                      color, cv::FILLED);
        cv::putText(image, label,
                    cv::Point(seg.box.x + 2, top - 2),
                    cv::FONT_HERSHEY_SIMPLEX,
                    fontScale,
                    cv::Scalar(255, 255, 255),
                    thickness);

        // -----------------------------
        // 3. Apply Segmentation Mask
        // -----------------------------
        if (!seg.mask.empty()) {
            // Ensure the mask is single-channel
            cv::Mat mask_gray;
            if (seg.mask.channels() == 3) {
                cv::cvtColor(seg.mask, mask_gray, cv::COLOR_BGR2GRAY);
            } else {
                mask_gray = seg.mask.clone();
            }

            // Threshold the mask to binary (object: 255, background: 0)
            cv::Mat mask_binary;
            cv::threshold(mask_gray, mask_binary, 127, 255, cv::THRESH_BINARY);

            // Create a colored version of the mask
            cv::Mat colored_mask;
            cv::cvtColor(mask_binary, colored_mask, cv::COLOR_GRAY2BGR);
            colored_mask.setTo(color, mask_binary); // Apply color where mask is present

            // Blend the colored mask with the original image
            cv::addWeighted(image, 1.0, colored_mask, maskAlpha, 0, image);
        }
    }
}


inline void YOLO11Segmentor::drawSegmentations(cv::Mat &image,
                                                  const std::vector<Framer::Segmentation> &results,
                                                  float maskAlpha, const float conf_threshold) const
{
    for (const auto &seg : results) {
        if (seg.conf < conf_threshold) {
            continue;
        }
        cv::Scalar color = classColors[seg.classId % classColors.size()];

        // -----------------------------
        // Draw Segmentation Mask Only
        // -----------------------------
        if (!seg.mask.empty()) {
            // Ensure the mask is single-channel
            cv::Mat mask_gray;
            if (seg.mask.channels() == 3) {
                cv::cvtColor(seg.mask, mask_gray, cv::COLOR_BGR2GRAY);
            } else {
                mask_gray = seg.mask.clone();
            }

            // Threshold the mask to binary (object: 255, background: 0)
            cv::Mat mask_binary;
            cv::threshold(mask_gray, mask_binary, 127, 255, cv::THRESH_BINARY);

            // Create a colored version of the mask
            cv::Mat colored_mask;
            cv::cvtColor(mask_binary, colored_mask, cv::COLOR_GRAY2BGR);
            colored_mask.setTo(color, mask_binary); // Apply color where mask is present

            // Blend the colored mask with the original image
            cv::addWeighted(image, 1.0, colored_mask, maskAlpha, 0, image);
        }
    }
}

inline std::vector<Framer::Segmentation> YOLO11Segmentor::segment(const cv::Mat &image,
                                                             float confThreshold,
                                                             float iouThreshold)
{
    float *blobPtr = nullptr;
    std::vector<int64_t> inputShape = {1, 3, inputImageShape.height, inputImageShape.width};
    cv::Mat letterboxImg = preprocess(image, blobPtr, inputShape);

    size_t inputSize = Framer::vectorProduct(inputShape);
    std::vector<float> inputVals(blobPtr, blobPtr + inputSize);
    delete[] blobPtr;

    Ort::MemoryInfo memInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
        memInfo,
        inputVals.data(),
        inputSize,
        inputShape.data(),
        inputShape.size()
        );

    std::vector<Ort::Value> outputs;
    {
        ScopedTimer timer("inference", &m_benchmarkQueue);
        outputs = session.Run(
            Ort::RunOptions{nullptr},
            inputNames.data(),
            &inputTensor,
            numInputNodes,
            outputNames.data(),
            numOutputNodes);
    }

    std::vector<Framer::Segmentation> segmentations;
    cv::Size letterboxSize(static_cast<int>(inputShape[3]), static_cast<int>(inputShape[2]));
    segmentations = postprocess(image.size(), letterboxSize, outputs, confThreshold, iouThreshold);

#ifdef FVERBOSE
    printBenchMarks();
#endif

    return segmentations;
}

