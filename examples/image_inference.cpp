/**
 * @file image_inference.cpp
 * @brief Object detection in a static image using YOLO models (v5, v7, v8, v9, v10, v11, v12).
 * 
 * This file implements an object detection application that utilizes YOLO 
 * (You Only Look Once) models, specifically versions 5, 7, 8, 9, 10, 11 and 12. 
 * The application loads a specified image, processes it to detect objects, 
 * and displays the results with bounding boxes around detected objects.
 *
 * The application supports the following functionality:
 * - Loading a specified image from disk.
 * - Initializing the YOLO detector with the desired model and labels.
 * - Detecting objects within the image.
 * - Drawing bounding boxes around detected objects and displaying the result.
 *
 * Configuration parameters can be adjusted to suit specific requirements:
 * - `isGPU`: Set to true to enable GPU processing for improved performance; 
 *   set to false for CPU processing.
 * - `labelsPath`: Path to the class labels file (e.g., COCO dataset).
 * - `imagePath`: Path to the image file to be processed (e.g., dogs.jpg).
 * - `modelPath`: Path to the desired YOLO model file (e.g., ONNX format).
 *
 * The application can be extended to use different YOLO versions by modifying 
 * the model path and the corresponding detector class.
 *
 * Usage Instructions:
 * 1. Compile the application with the necessary OpenCV and YOLO dependencies.
 * 2. Ensure that the specified image and model files are present in the 
 *    provided paths.
 * 3. Run the executable to initiate the object detection process.
 *
 * @note The code includes commented-out sections to demonstrate how to switch 
 * between different YOLO models and image inputs.
 *
 * Author: Abdalrahman M. Amer, www.linkedin.com/in/abdalrahman-m-amer
 * Date: 29.09.2024
 */

// Include necessary headers
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <string>

#include "det/YOLO11.hpp"
#include "exports.h"


int main(int argc, char* argv[])
{
    // Paths to the model, labels, and test image
    std::string modelPath;
    std::string labelsPath;
    std::string imagePath;

    // Usage: camera_inference.exe <model_path> <labels_file_path> <image_path>
    if (argc < 4 && !EXAMPLES_ASSETS_DIR_ENABLED) {
        std::cerr << "Usage: image_inference.exe <model_path> <labels_file_path> <image_path>\n";
        return 1;
    }

    if (argc == 4) {
        modelPath = argv[1];
        labelsPath = argv[2];
        imagePath = argv[3];
    } else {
        std::string assets_dir = std::string(EXAMPLES_ASSETS_DIR);
        modelPath = assets_dir + "/../../scripts/yolo11n.onnx";
        labelsPath = assets_dir + "/coco.txt";
        imagePath = assets_dir + "/img1.jpg";
        std::cout << "Selecting assets from " + assets_dir + " directory." << std::endl;
    }

    // Initialize the YOLO detector with the chosen model and labels
    bool isGPU = false; // Set to false for CPU processing

    Framer::YOLO11Detector detector(modelPath, labelsPath, isGPU); // Uncomment for YOLOv12


    // Load an image
    cv::Mat image = cv::imread(imagePath);
    if (image.empty())
    {
        std::cerr << "Error: Could not open or find the image!\n";
        return -1;
    }

    // Enforcing a small preview. Because of the image size I'm trying to load (3000x4000)
    cv::resize(image, image, cv::Size(480, 640));

    // Detect objects in the image
    std::vector<Framer::Detection> results = detector.detect(image);

    // Draw bounding boxes on the image
    detector.drawBoundingBox(image, results); // simple bbox drawing
    // detector.drawBoundingBoxMask(image, results); // Uncomment for mask drawing

    // Display the image
    cv::imshow("Detections", image);
    cv::waitKey(0); // Wait for a key press to close the window

    return 0;
}
