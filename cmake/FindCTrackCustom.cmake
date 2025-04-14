# I hate to do this, but I got no other choice at the moment, aight.
# We fetch, configure, build and install the ctrack and use find_package on it.
# WARNING: This file is named FindCTrackCustom, because the name FindCtrack will cause a
#           recursive call to this file, repeatedely due to find_package(ctrack ...).


include(FetchContent)
set(FETCHCONTENT_QUIET OFF)
FetchContent_Declare(ctrack
    GIT_REPOSITORY https://github.com/Compaile/ctrack.git
    GIT_TAG v1.0.2
    GIT_PROGRESS ON
)
FetchContent_MakeAvailable(ctrack)
FetchContent_GetProperties(ctrack)

# Configure CTRACK
message(STATUS "Configuring CTrack")
execute_process(COMMAND ${CMAKE_COMMAND}
    -G ${CMAKE_GENERATOR}
    -S ${ctrack_SOURCE_DIR}
    -B ${ctrack_BINARY_DIR}
    -DDISABLE_EXAMPLES=ON
    -DENABLE_WARNINGS=ON
)

# Build CTRACK
message(STATUS "Building CTrack")
execute_process(COMMAND ${CMAKE_COMMAND}
    --build ${ctrack_BINARY_DIR}
    --config Release)

# INSTALL CTRACK
set(ctrack_INSTALL_DIR ${CMAKE_BINARY_DIR}/ctrack)
message(STATUS "Installing CTrack")
execute_process(COMMAND ${CMAKE_COMMAND}
    --install ${ctrack_BINARY_DIR}
    --prefix=${ctrack_INSTALL_DIR}
    --config=Release
)

list(APPEND CMAKE_PREFIX_PATH ${ctrack_INSTALL_DIR})

find_package(ctrack REQUIRED)

# Now we should link to ctrack::ctrack in the main file
