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
