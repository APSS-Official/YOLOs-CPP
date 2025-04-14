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
 * @file Debug.hpp
 * @brief Header file for debugging utilities.
 * 
 * This file provides macros to enable or disable debug printing based on the
 * configuration. When DEBUG_MODE is defined, debug messages are printed to the
 * standard output; otherwise, they are ignored.
 */

#ifdef FVERBOSE
#define DEBUG_PRINT(x) std::cout << x << std::endl
#define INFO_PRINT(x) std::cout << x << std::endl
#else
#define DEBUG_PRINT(x)
#define INFO_PRINT(x)
#endif

#define ERROR_PRINT(x) std::cerr << "Error: " << x << std::endl
#define FATAL_PRINT(x) std::cerr << "Fatal: " << x << std::endl; exit(1)
