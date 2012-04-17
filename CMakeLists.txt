project(finger-tracking)

cmake_minimum_required(VERSION 2.4.0)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-cpp")

# needed for VTK library
if(COMMAND cmake_policy)
    cmake_policy(SET CMP0003 NEW)
endif(COMMAND cmake_policy)

# enable debugging, and show all warnings
set(CMAKE_CXX_FLAGS "-g -Wall")

# OpenNI
include_directories("/usr/include/ni")
set(LIBS ${LIBS} OpenNI)

FIND_PACKAGE(OpenCV REQUIRED)

set(SRCS
    main.cpp
    SkeletonSensor.cpp
    log.cpp
)

add_executable(finger-tracking ${SRCS})

target_link_libraries(finger-tracking ${LIBS} ${OpenCV_LIBS})
