# NOTE: Must be added as a sub-directory

add_executable(camera_inference camera_inference.cpp)
add_executable(camera_inference_seg camera_inference_seg.cpp)
add_executable(image_inference	image_inference.cpp)
add_executable(image_inference_seg	image_inference_seg.cpp)
add_executable(video_inference	video_inference.cpp)
add_executable(video_inference_seg	video_inference_seg.cpp)

set(EXAMPLES_INCLUDE_DIRS ${CMAKE_SOURCE_DIR}/include "${OnnxRuntime_INCLUDE_DIRS}")
set(EXAMPLES_LINK_LIBRARIES Framer::Framer ${OpenCV_LIBS} ${OnnxRuntime_LIBRARIES})

# Export EXAMPLES_DIR (the current directory
set(EXAMPLES_ASSETS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/assets")
configure_file(exports.h.in exports.h @ONLY)

target_include_directories(camera_inference PRIVATE ${EXAMPLES_INCLUDE_DIRS} ${CMAKE_CURRENT_BINARY_DIR})
target_include_directories(camera_inference_seg PRIVATE ${EXAMPLES_INCLUDE_DIRS} ${CMAKE_CURRENT_BINARY_DIR})
target_include_directories(image_inference PRIVATE ${EXAMPLES_INCLUDE_DIRS} ${CMAKE_CURRENT_BINARY_DIR})
target_include_directories(image_inference_seg PRIVATE ${EXAMPLES_INCLUDE_DIRS} ${CMAKE_CURRENT_BINARY_DIR})
target_include_directories(video_inference PRIVATE ${EXAMPLES_INCLUDE_DIRS} ${CMAKE_CURRENT_BINARY_DIR})
target_include_directories(video_inference_seg PRIVATE ${EXAMPLES_INCLUDE_DIRS} ${CMAKE_CURRENT_BINARY_DIR})

target_compile_features(camera_inference PRIVATE ${EXAMPLES_COMPILE_FEATURES})
target_compile_features(camera_inference_seg PRIVATE ${EXAMPLES_COMPILE_FEATURES})
target_compile_features(image_inference PRIVATE ${EXAMPLES_COMPILE_FEATURES})
target_compile_features(image_inference_seg PRIVATE ${EXAMPLES_COMPILE_FEATURES})
target_compile_features(video_inference PRIVATE ${EXAMPLES_COMPILE_FEATURES})
target_compile_features(video_inference_seg PRIVATE ${EXAMPLES_COMPILE_FEATURES})

target_link_libraries(camera_inference PRIVATE ${EXAMPLES_LINK_LIBRARIES})
target_link_libraries(camera_inference_seg PRIVATE ${EXAMPLES_LINK_LIBRARIES})
target_link_libraries(image_inference PRIVATE ${EXAMPLES_LINK_LIBRARIES})
target_link_libraries(image_inference_seg PRIVATE ${EXAMPLES_LINK_LIBRARIES})
target_link_libraries(video_inference PRIVATE ${EXAMPLES_LINK_LIBRARIES})
target_link_libraries(video_inference_seg PRIVATE ${EXAMPLES_LINK_LIBRARIES})
