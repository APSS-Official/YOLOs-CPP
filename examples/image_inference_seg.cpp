#include <iostream>
#include <string>

#include <opencv2/highgui/highgui.hpp>

#include <seg/YOLO11Seg.hpp>
#include "exports.h"

int main(int argc, char* argv[])
{
    // Paths to the model, labels, and test image
    std::string modelPath;
    std::string labelsPath;
    std::string imagePath;

    // Usage: image_inference_seg.exe <model_path> <labels_file_path> <image_path>
    if (argc < 4 && !EXAMPLES_ASSETS_DIR_ENABLED) {
        std::cerr << "Usage: image_inference_seg.exe <model_path> <labels_file_path> <image_path>\n";
        return 1;
    }

    if (argc == 4) {
        modelPath = argv[1];
        labelsPath = argv[2];
        imagePath = argv[3];
    } else {
        std::string assets_dir = std::string(EXAMPLES_ASSETS_DIR);
        modelPath = assets_dir + "/../../scripts/yolo11n-seg.onnx";
        labelsPath = assets_dir + "/coco.txt";
        imagePath = assets_dir + "/img2.jpg";
        std::cout << "Selecting assets from " + assets_dir + " directory." << std::endl;
    }

    // Initialize the YOLO segmentor with the chosen model and labels
    bool isGPU = false; // Set to false for CPU processing
    YOLO11Segmentor segmentor(modelPath, labelsPath, isGPU); // Uncomment for YOLOv12


    // Load an image
    cv::Mat image = cv::imread(imagePath);
    if (image.empty())
    {
        std::cerr << "Error: Could not open or find the image!\n";
        return -1;
    }

    // Enforcing a small preview. Because of the image size I'm trying to load (3000x4000)
    cv::resize(image, image, cv::Size(480, 640));

    // Segment objects in the image
    std::vector<Framer::Segmentation> results = segmentor.segment(image);

    // Draw bounding boxes on the image
    segmentor.drawSegmentationsAndBoxes(image, results); // simple bbox drawing

    // Display the image
    cv::imshow("Segmentations", image);
    cv::waitKey(0); // Wait for a key press to close the window

    return 0;
}
