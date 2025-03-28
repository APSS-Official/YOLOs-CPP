cmake_minimum_required(VERSION 3.21.1)
cmake_policy(SET CMP0074 NEW)

project(Framer VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Some options
option(FBUILD_EXAMPLES "Build Examples" ON)
option(FGPU "Enable GPU support" OFF)
option(FBENCHMARK "Enable Benchmarks" OFF)
option(FVERBOSE "Enable Verbosity" OFF)

# Setting default value for Benchmarks if we are in Debug mode
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(FBENCHMARK ON)
    set(FVERBOSE ON)
endif()

# Add cmake dir to find .cmake files
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

# Packages
find_package(OpenCV 4.10.0 CONFIG REQUIRED) # Picked from vcpkg
find_package(OnnxRuntime 1.21.0 REQUIRED)   # Uses the cmake/FindOnnxRuntime.cmake, no support for GPU yet!

add_library(Framer INTERFACE)
add_library(Framer::Framer ALIAS Framer)

# Some definitions being exposed to C++, optionally defined.
target_compile_definitions(Framer INTERFACE
    $<$<BOOL:${FGPU}>:FHAS_GPU=1>
    $<$<BOOL:${FBENCHMARK}>:FHAS_BENCHMARKS=1>
    $<$<BOOL:${FVERBOSE}>:FVERBOSE=1>
    $<$<CONFIG:Debug>:FDEBUG>
    $<$<CONFIG:RelWithDebInfo>:FDEBUG>
    $<$<CONFIG:Release>:FRELEASE>
    $<$<CONFIG:MinSizeRel>:FRELEASE>
)
target_include_directories(Framer INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_compile_features(Framer INTERFACE cxx_std_20)
target_link_libraries(Framer INTERFACE
    ${OpenCV_LIBS}
    ${OnnxRuntime_LIBRARIES}
)

# Building examples
if (FBUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

if (${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_CURRENT_SOURCE_DIR})
    message(NOTICE "Framer is not part of any parent project, defining targets now.")
    include(GNUInstallDirs)
    include(CMakePackageConfigHelpers)

    install(TARGETS Framer
        EXPORT FramerTargets
        INCLUDES DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
    )

    # For find_package
    install(EXPORT FramerTargets
        FILE FramerTargets.cmake
        NAMESPACE Framer::
        DESTINATION "${CMAKE_INSTALL_DATADIR}/Framer"
    )

    install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

    configure_package_config_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/cmake/FramerConfig.cmake.in
        ${CMAKE_CURRENT_BINARY_DIR}/FramerConfig.cmake
        INSTALL_DESTINATION ${CMAKE_INSTALL_DATADIR}/Framer
    )

    write_basic_package_version_file(
        ${CMAKE_CURRENT_BINARY_DIR}/FramerConfigVersion.cmake
        VERSION ${PROJECT_VERSION}
        COMPATIBILITY SameMajorVersion
    )

    install(FILES
        ${CMAKE_CURRENT_BINARY_DIR}/FramerConfig.cmake
        ${CMAKE_CURRENT_BINARY_DIR}/FramerConfigVersion.cmake
        DESTINATION ${CMAKE_INSTALL_DATADIR}/Framer
    )
else()
    message(NOTICE "Framer is part of the parent project: ${PROJECT_NAME}")
endif()
