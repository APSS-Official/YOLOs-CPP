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

/**
 * @file ScopedTimer.hpp
 * @brief Header file for timing utilities.
 * 
 * This file defines the ScopedTimer class, which measures the duration of a 
 * code block for performance profiling. When TIMING_MODE is defined, it records
 * the time taken for execution and prints it to the standard output upon 
 * destruction; otherwise, it provides an empty implementation.
 */

#include <chrono>
#include <iostream>
#include <string>
#include <queue>

#include "tools/Config.hpp" // Include the config file to access the flags
#include "tools/Debug.hpp"

#ifdef FBENCHMARK
class ScopedTimer {
public:
    /**
     * @brief Constructs a ScopedTimer to measure the duration of a named code block.
     * @param name The name of the code block being timed.
     */
    ScopedTimer(const std::string &stage, std::priority_queue<std::string> *benchmarkQueue = nullptr, int appendPriority = -1)
        : m_stage(stage)
        , m_benchmarkQueue(benchmarkQueue)
        , m_priority(appendPriority)
        , m_startTime(std::chrono::high_resolution_clock::now())
    {
        static int priority_counter = 1;
        if (appendPriority < 1)
            m_priority = priority_counter++;
    }
    
    /**
     * @brief Destructor that calculates and prints the elapsed time.
     */
    ~ScopedTimer() {
        auto stop = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> duration = stop - m_startTime;

        std::string benchmarkString = m_stage + " (" + std::to_string(duration.count()) + "ms)";

        if (m_benchmarkQueue){
            m_benchmarkQueue->push(benchmarkString);
        } else {
#ifdef FVERBOSE
            DEBUG_PRINT(benchmarkString);
#endif
        }
    }

private:
    std::string m_stage; ///< The name of the timed function.
    std::chrono::time_point<std::chrono::high_resolution_clock> m_startTime; ///< Start time point.
    int m_priority = 0;
    std::priority_queue<std::string> *m_benchmarkQueue;
};
#else
class ScopedTimer {
public:
    ScopedTimer(const std::string &name, std::priority_queue<std::string> *benchmarkQueue = nullptr, int appendPriority = -1) {}
    ~ScopedTimer() {}
};
#endif // TIMING_MODE
