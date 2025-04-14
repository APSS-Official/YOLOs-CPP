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
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/videoio.hpp>
#include <iostream>
#include <string>
#include <thread>
#include <queue>
#include <mutex>
#include <atomic>
#include <condition_variable>

// #include "det/YOLO5.hpp"  // Uncomment for YOLOv5
// #include "det/YOLO7.hpp"  // Uncomment for YOLOv7
// #include "det/YOLO8.hpp"  // Uncomment for YOLOv8
// #include "det/YOLO9.hpp"  // Uncomment for YOLOv9
// #include "det/YOLO10.hpp" // Uncomment for YOLOv10
// #include "det/YOLO11.hpp" // Uncomment for YOLOv11
#include "det/YOLO11.hpp" // Uncomment for YOLOv12
#include "exports.h"


// Thread-safe queue implementation
template <typename T>
class SafeQueue {
public:
    SafeQueue() : q(), m(), c() {}

    // Add an element to the queue.
    void enqueue(T t) {
        std::lock_guard<std::mutex> lock(m);
        q.push(t);
        c.notify_one();
    }

    // Get the first element from the queue.
    bool dequeue(T& t) {
        std::unique_lock<std::mutex> lock(m);
        while (q.empty()) {
            if (finished) return false;
            c.wait(lock);
        }
        t = q.front();
        q.pop();
        return true;
    }

    void setFinished() {
        std::lock_guard<std::mutex> lock(m);
        finished = true;
        c.notify_all();
    }

private:
    std::queue<T> q;
    mutable std::mutex m;
    std::condition_variable c;
    bool finished = false;
};

int main(int argc, char* argv[])
{
    // Paths to the model, labels, input video, and output video
    std::string modelPath;
    std::string labelsPath;
    std::string videoPath;
    std::string outputPath;

    // Usage: video_inference.exe <model_path> <labels_file_path> <video_input_source> <video_output_source>
    if (argc < 5 && !EXAMPLES_ASSETS_DIR_ENABLED) {
        std::cerr << "Usage: video_inference.exe <model_path> <labels_file_path> <video_input_source> <video_output_source>\n";
        return 1;
    }

    if (argc == 5) {
        modelPath = argv[1];
        labelsPath = argv[2];
        videoPath = argv[3];
        outputPath = argv[4];
    } else {
        std::string assets_dir = std::string(EXAMPLES_ASSETS_DIR);
        modelPath = assets_dir + "/../../scripts/yolo11n.onnx";
        labelsPath = assets_dir + "/coco.txt";
        videoPath = assets_dir + "/vid.mp4";
        outputPath = assets_dir + "/../../../scan.mp4"; // Outside the project dir
        std::cout << "Selecting assets from " + assets_dir + " directory." << std::endl;
    }

    // Initialize the YOLO detector
    bool isGPU = true; // Set to false for CPU processing
    // YOLO9Detector detector(modelPath, labelsPath, isGPU); // YOLOv9
    // YOLO11Detector detector(modelPath, labelsPath, isGPU); // YOLOv11
    Framer::YOLO11Detector detector(modelPath, labelsPath, isGPU); // YOLOv12


    // Open the video file
    cv::VideoCapture cap(videoPath);
    if (!cap.isOpened())
    {
        std::cerr << "Error: Could not open or find the video file!\n";
        return -1;
    }

    // Get video properties
    int frameWidth = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int frameHeight = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    int fps = static_cast<int>(cap.get(cv::CAP_PROP_FPS));
    int fourcc = static_cast<int>(cap.get(cv::CAP_PROP_FOURCC)); // Get codec of input video

    // Create a VideoWriter object to save the output video with the same codec
    cv::VideoWriter out(outputPath, fourcc, fps, cv::Size(frameWidth, frameHeight), true);
    if (!out.isOpened())
    {
        std::cerr << "Error: Could not open the output video file for writing!\n";
        return -1;
    }

    // Thread-safe queues and processing...
    // Thread-safe queues
    SafeQueue<cv::Mat> frameQueue;
    SafeQueue<std::pair<int, cv::Mat>> processedQueue;

    // Flag to indicate processing completion
    std::atomic<bool> processingDone(false);


    // Capture thread
    std::thread captureThread([&]() {
        cv::Mat frame;
        int frameCount = 0;
        while (cap.read(frame))
        {
            frameQueue.enqueue(frame.clone()); // Clone to ensure thread safety
            frameCount++;
        }
        frameQueue.setFinished();
    });

    // Processing thread
    std::thread processingThread([&]() {
        cv::Mat frame;
        int frameIndex = 0;
        while (frameQueue.dequeue(frame))
        {
            // Detect objects in the frame
            std::vector<Framer::Detection> results = detector.detect(frame);

            // Draw bounding boxes on the frame
            detector.drawBoundingBoxMask(frame, results); // Uncomment for mask drawing

            // Enqueue the processed frame
            processedQueue.enqueue(std::make_pair(frameIndex++, frame));
        }
        processedQueue.setFinished();
    });

    // Writing thread
    std::thread writingThread([&]() {
        std::pair<int, cv::Mat> processedFrame;
        while (processedQueue.dequeue(processedFrame))
        {
            out.write(processedFrame.second);
        }
    });

    // Wait for all threads to finish
    captureThread.join();
    processingThread.join();
    writingThread.join();

    // Release resources
    cap.release();
    out.release();
    cv::destroyAllWindows();

    std::cout << "Video processing completed successfully. Saved " << outputPath << std::endl;

    return 0;
}
