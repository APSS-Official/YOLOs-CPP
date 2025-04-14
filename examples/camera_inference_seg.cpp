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
#include <iostream>
#include <vector>
#include <thread>
#include <atomic>

#include <opencv2/highgui/highgui.hpp>

#include <seg/YOLO11Seg.hpp>
#include <utils/utils.hpp>
#include "exports.h"

// Include the bounded queue
#include <utils/boundedthreadsafequeue.hpp>

int main(int argc, char* argv[])
{
    // Paths to the model, labels, video source
    std::string modelPath;
    std::string labelsPath;
    int videoSource;

    // Usage: camera_inference_seg.exe <model_path> <labels_file_path> <video_source>
    if (argc < 4 && !EXAMPLES_ASSETS_DIR_ENABLED) {
        std::cerr << "Usage: camera_inference_seg.exe <model_path> <labels_file_path> <video_source>\n";
        return 1;
    }

    if (argc == 4) {
        modelPath = argv[1];
        labelsPath = argv[2];
        videoSource = std::stoi(argv[3]);
    } else {
        std::string assets_dir = std::string(EXAMPLES_ASSETS_DIR);
        modelPath = assets_dir + "/../../scripts/yolo11n-seg.onnx";
        labelsPath = assets_dir + "/coco.txt";
        videoSource = 0; // your usb cam device
        std::cout << "Selecting assets from " + assets_dir + " directory." << std::endl;
    }

    // Configuration parameters
    const bool isGPU = true;

    // Initialize YOLO segmentor
    YOLO11Segmentor segmentor(modelPath, labelsPath, isGPU);


    // Open video capture
    cv::VideoCapture cap;
    cap.open(videoSource); // Specify V4L2 backend for better performance
    if (!cap.isOpened())
    {
        std::cerr << "Error: Could not open the camera!\n";
        return -1;
    }

    // Set camera properties
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 720);
    cap.set(cv::CAP_PROP_FPS, 30);

    // Initialize queues with bounded capacity
    const size_t max_queue_size = 2; // Double buffering
    BoundedThreadSafeQueue<cv::Mat> frameQueue(max_queue_size);
    BoundedThreadSafeQueue<std::pair<cv::Mat, std::vector<Framer::Segmentation>>> processedQueue(max_queue_size);
    std::atomic<bool> stopFlag(false);

    // Producer thread: Capture frames
    std::thread producer([&]() {
        cv::Mat frame;
        while (!stopFlag.load() && cap.read(frame))
        {
            if (!frameQueue.enqueue(frame))
                break; // Queue is finished
        }
        frameQueue.set_finished();
    });

    // Consumer thread: Process frames
    std::thread consumer([&]() {
        cv::Mat frame;
        while (!stopFlag.load() && frameQueue.dequeue(frame))
        {
            // Perform detection
            std::vector<Framer::Segmentation> segmentations = segmentor.segment(frame);

            // Enqueue processed frame
            if (!processedQueue.enqueue(std::make_pair(frame, segmentations)))
                break;
        }
        processedQueue.set_finished();
    });

    std::pair<cv::Mat, std::vector<Framer::Segmentation>> item;

    #ifdef __APPLE__
    // For macOS, ensure UI runs on the main thread
    while (!stopFlag.load() && processedQueue.dequeue(item))
    {
        cv::Mat displayFrame = item.first;
        segmentor.drawSegmentationsAndBoxes(displayFrame, item.second);

        cv::imshow("Segmentations", displayFrame);
        if (cv::waitKey(1) == 'q')
        {
            stopFlag.store(true);
            frameQueue.set_finished();
            processedQueue.set_finished();
            break;
        }
    }
    #else
    // Display thread: Show processed frames
    std::thread displayThread([&]() {
        while (!stopFlag.load() && processedQueue.dequeue(item))
        {
            cv::Mat displayFrame = item.first;
            // segmentor.drawBoundingBox(displayFrame, item.second);
            segmentor.drawSegmentationsAndBoxes(displayFrame, item.second);

            // Display the frame
            cv::imshow("Segmentations", displayFrame);
            // Use a small delay and check for 'q' key press to quit
            if (cv::waitKey(1) == 'q') {
                stopFlag.store(true);
                frameQueue.set_finished();
                processedQueue.set_finished();
                break;
            }
        }
    });
    displayThread.join();
    #endif

    // Join all threads
    producer.join();
    consumer.join();

    // Release resources
    cap.release();
    cv::destroyAllWindows();

    return 0;
}
