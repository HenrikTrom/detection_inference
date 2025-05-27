cmake_minimum_required(VERSION 3.16)
project(detection_inference VERSION 1.0.0)
set(DETECTION_INFERENCE_VERSION ${PROJECT_VERSION})
set(CMAKE_CXX_STANDARD 17)

option(DEBUG_TIME_LOGGING "Enable cout for logging runtime" ON)

set(CMAKE_BUILD_TYPE Release)
# set(CMAKE_BUILD_TYPE Debug)
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  set(DEBUG_TIME_LOGGING ON CACHE BOOL "Enable cout for logging runtime" FORCE)
else()
  set(OPTIMIZATION_FLAGS $ENV{CPP_OPTIMIZATIONS})
  set(
    CMAKE_CXX_FLAGS 
    "${CMAKE_CXX_FLAGS} ${OPTIMIZATION_FLAGS}"
  )
endif()

message(STATUS "DEBUG_TIME_LOGGING is ${DEBUG_TIME_LOGGING}")

include(cmake/ccache.cmake)
set(CONFIG_DIR ${CMAKE_CURRENT_SOURCE_DIR}/cfg)

set(BATCH_SIZE 5 CACHE STRING "Fixed batch size for inference")
if(DEFINED ENV{BATCH_SIZE})
    set(BATCH_SIZE $ENV{BATCH_SIZE})
    message(STATUS "Environment variable 'BATCH_SIZE' found: Setting BATCH_SIZE to ${BATCH_SIZE}")
else()
    message(WARNING "Environment variable 'BATCH_SIZE' is not found. Using default BATCH_SIZE of ${BATCH_SIZE}")
endif()

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/cmake/config.h.in
  ${CMAKE_CURRENT_SOURCE_DIR}/src/config.h
)

link_directories(/usr/local/lib)
if(POLICY CMP0146)
  cmake_policy(SET CMP0146 OLD)
endif()
set(OPENCV_VERSION $ENV{OPENCV_VERSION})
message(STATUS "OpenCV version: ${OPENCV_VERSION}")
find_package(OpenCV ${OPENCV_VERSION} REQUIRED)

find_package(Eigen3 REQUIRED NO_MODULE)
find_package(tensorrt-cpp-api REQUIRED)
find_package(cpp_utils REQUIRED)
find_package(fmt REQUIRED)

# ============ LIBRARY ============
add_library(${PROJECT_NAME} STATIC
  src/config_parser.cpp
  src/detection.cpp
  src/stages.cpp
  src/utils.cpp
)

target_include_directories(
  ${PROJECT_NAME} PUBLIC ${OpenCV_INCLUDE_DIRS} ${tensorrt-cpp-api_INCLUDE_DIRS}
)
target_link_libraries(
  ${PROJECT_NAME} 
  PUBLIC 
  ${OpenCV_LIBS}
  ${CMAKE_THREAD_LIBS_INIT} 
  Eigen3::Eigen
  cpp_utils 
  tensorrt-cpp-api
)
# ============ EXECUTABLES ============
# BENCHMARK

add_executable(inference_benchmark src/apps/inference_benchmark.cpp)
if(DEBUG_TIME_LOGGING)
  target_compile_definitions(inference_benchmark PUBLIC USE_DEBUG_TIME_LOGGING=1)
  target_compile_definitions(${PROJECT_NAME} PUBLIC USE_DEBUG_TIME_LOGGING=1)
endif()
target_link_libraries(inference_benchmark ${PROJECT_NAME})

# VIDEO INFERENCE EXPORT

add_executable(video_inference_export src/apps/video_inference_export.cpp)
if(DEBUG_TIME_LOGGING)
  target_compile_definitions(video_inference_export PUBLIC USE_DEBUG_TIME_LOGGING=1)
  target_compile_definitions(${PROJECT_NAME} PUBLIC USE_DEBUG_TIME_LOGGING=1)
endif()
target_link_libraries(video_inference_export ${PROJECT_NAME})

# BBOX OVERLAY

add_executable(bbox_overlay src/apps/bbox_overlay.cpp)
if(DEBUG_TIME_LOGGING)
  target_compile_definitions(bbox_overlay PUBLIC USE_DEBUG_TIME_LOGGING=1)
  target_compile_definitions(${PROJECT_NAME} PUBLIC USE_DEBUG_TIME_LOGGING=1)
endif()
target_link_libraries(bbox_overlay ${PROJECT_NAME})

# ============ INSTALLATION ============
include(GNUInstallDirs)
install(TARGETS ${PROJECT_NAME}
  EXPORT ${PROJECT_NAME}Targets
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  INCLUDES DESTINATION include
)
# installs the export file (the “recipe”) created in the previous step into a standard location within your install tree (for find_package)
install(EXPORT ${PROJECT_NAME}Targets
  # NAMESPACE ${PROJECT_NAME}::
  DESTINATION lib/cmake/${PROJECT_NAME}
)

include(CMakePackageConfigHelpers)
configure_package_config_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Config.cmake.in"
  "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
  INSTALL_DESTINATION lib/cmake/${PROJECT_NAME}
)
write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake"
  VERSION $DETECTION_INFERENCE_VERSION
  COMPATIBILITY AnyNewerVersion
)
# build tree usage
export(EXPORT ${PROJECT_NAME}Targets
  FILE "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Targets.cmake"
  # NAMESPACE ${PROJECT_NAME}::
)

install(FILES
  "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
  "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake"
  DESTINATION lib/cmake/${PROJECT_NAME}
)
# Installs all header files (*.h) from your include/ folder to: ${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME}
install(DIRECTORY src/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME}
  FILES_MATCHING PATTERN "*.hpp" PATTERN "*.h"
) 
