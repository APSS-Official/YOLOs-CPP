# File responsible for ONNXRuntime

set(ONNXRUNTIME_VERSION 1.21.0 CACHE STRING "ONNX Runtime version to use")

# Platform specific binaries
if (FGPU)
    set(ONNXRUNTIME_ACCELARATOR "-gpu")
else()
    set(ONNXRUNTIME_ACCELARATOR "")
endif()

if (WIN32)
    set(ONNXRUNTIME_FILE_NAME "onnxruntime-win-x64${ONNXRUNTIME_ACCELARATOR}-${ONNXRUNTIME_VERSION}.zip")
else()
    message(FATAL_ERROR "Framer's OnnxRuntime support for other platforms/accelarators isn't managed yet. Please submit an issue or customize the file at ${CMAKE_CURRENT_LIST_FILE}")
endif()

# You can also specify a custom link through -DONNXRUNTIME_URL during configuration.
if (NOT ONNXRUNTIME_URL)
    set(ONNXRUNTIME_URL "https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/${ONNXRUNTIME_FILE_NAME}")
endif()

include(FetchContent)
set(FETCHCONTENT_QUIET OFF)
FetchContent_Declare(onnx
    DOWNLOAD_EXTRACT_TIMESTAMP true
    # Keep this URL last or we get a surprise (source: https://stackoverflow.com/questions/74996365/cmake-error-at-least-one-entry-of-url-is-a-path-invalid-in-a-list?)
    URL ${ONNXRUNTIME_URL}
)
FetchContent_MakeAvailable(onnx)
FetchContent_GetProperties(onnx)

find_path(OnnxRuntime_INCLUDE_DIR
    onnxruntime_cxx_api.h
    HINTS "${onnx_SOURCE_DIR}/include")
find_library(OnnxRuntime_LIBRARY
    NAMES onnxruntime
    HINTS "${onnx_SOURCE_DIR}/lib")
find_library(OnnxRuntime_PROVIDERS_SHARED_LIBRARY
    NAMES onnxruntime_providers_shared
    HINTS "${onnx_SOURCE_DIR}/lib")

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(OnnxRuntime REQUIRED_VARS
    OnnxRuntime_INCLUDE_DIR
    OnnxRuntime_LIBRARY
    OnnxRuntime_PROVIDERS_SHARED_LIBRARY
    VERSION_VAR "${ONNXRUNTIME_VERSION}"
)

if (OnnxRuntime_FOUND)
    set(OnnxRuntime_LIBRARIES ${OnnxRuntime_LIBRARY} ${OnnxRuntime_PROVIDERS_SHARED_LIBRARY})
    set(OnnxRuntime_INCLUDE_DIRS ${OnnxRuntime_INCLUDE_DIR})
endif()
