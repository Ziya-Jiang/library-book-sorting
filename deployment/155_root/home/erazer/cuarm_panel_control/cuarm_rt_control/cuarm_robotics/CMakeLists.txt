cmake_minimum_required(VERSION 3.2.0)
project(cuarm_robotics)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if (CMAKE_VERSION VERSION_LESS "3.2.0")
    set(CMAKE_INCLUDE_CURRENT_DIR ON)
endif()

# if(NOT CMAKE_BUILD_TYPE)
#     set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose Release or Debug" FORCE)
# endif()
# MESSAGE(${CMAKE_BUILD_TYPE})

###################################Find Packages###########################################
if (WIN32)
    message("OS: Windows")
    include_directories($ENV{INCLUDE_PATH})
    link_directories($ENV{LIBRARY_PATH})
    add_definitions(-DYAML_CPP_STATIC_DEFINE) #Yaml-cpp setting under Windows
    add_definitions(-DNOMINMAX) #Pinocchio setting under Windows
elseif (UNIX AND NOT APPLE)
    message("OS: Linux")
    find_package(yaml-cpp)
    add_compile_options(-ggdb3)
endif()

if (yaml-cpp_VERSION VERSION_GREATER 0.7.0)
    set(yaml_library yaml-cpp::yaml-cpp)
else()
    set(yaml_library yaml-cpp)
endif()

find_package(Eigen3 3.4 REQUIRED)
find_package(pinocchio)
find_package(Boost)

###################################Add subdirectory#################################################
cmake_path(GET PROJECT_SOURCE_DIR PARENT_PATH PARENT_DIR)

message("ROBOT_PLATFORM_UTILS_DIR: ${ROBOT_PLATFORM_UTILS_DIR}")
if(NOT DEFINED ROBOT_PLATFORM_UTILS_DIR)
    set(ROBOT_PLATFORM_UTILS_DIR "${PARENT_DIR}")
endif()
message("ROBOT_PLATFORM_UTILS_DIR: ${ROBOT_PLATFORM_UTILS_DIR}")

if(NOT DEFINED ROBOT_PLATFORM_UTILS_BINARY_DIR)
    set(ROBOT_PLATFORM_UTILS_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}")
endif()

# Check if CONFIG_PREFIX_PATH is already defined
if(NOT DEFINED CONFIG_PREFIX_PATH OR CONFIG_PREFIX_PATH STREQUAL "")
    if(DEFINED CUARM_CONFIGURATION_DIR AND NOT CUARM_CONFIGURATION_DIR STREQUAL "")
        set(CONFIG_PREFIX_PATH "${CUARM_CONFIGURATION_DIR}/cuarm_configuration")
    else()
        cmake_path(GET PARENT_DIR PARENT_PATH PARENT_PARENT_DIR)
        set(CONFIG_PREFIX_PATH "${PARENT_PARENT_DIR}/cuarm_configuration")
    endif()
endif()

if(NOT TARGET robot::platform_utils)
    add_subdirectory(${ROBOT_PLATFORM_UTILS_DIR}/robot_platform_utils/cpp ${ROBOT_PLATFORM_UTILS_BINARY_DIR}/robot_platform_utils)
endif()
# ###################################Add source files to lib###########################################
# file(GLOB SRC_FILES "src/*.cpp" "src/robot/*.cpp" "src/robotics_library/*.cpp")
# add_library(cuarm_robotics
#     ${SRC_FILES}
# )
# target_link_libraries(cuarm_robotics PUBLIC 
#     -lpthread
#     ${yaml_library}
#     ${PINOCCHIO_LIBRARIES}
#     ${Boost_LIBRARIES}
#     robot::platform_utils
# )
# target_include_directories(cuarm_robotics PUBLIC 
#     ${Boost_INCLUDE_DIR}
#     ${EIGEN3_INCLUDE_DIRS}
#     ${PINOCCHIO_INCLUDE_DIRS}
#     ${PROJECT_SOURCE_DIR}/src
#     ${ROBOT_PLATFORM_UTILS_DIR}
# )
# target_compile_definitions(cuarm_robotics PUBLIC
#     EIGEN_QUATERNIONBASE_PLUGIN="${PROJECT_SOURCE_DIR}/src/3rd_party/Eigen_QuaternionBase_Plugin.hpp" 
#     EIGEN_MATRIX_PLUGIN="${PROJECT_SOURCE_DIR}/src/3rd_party/Matrix_initializer_list.h"
# )
# ############################################################################
# 1. Base Robotics Library (cuarm_robotics::robotics)
# ############################################################################
file(GLOB ROBOTICS_SRC "src/robotics/robot/*.cpp" "src/robotics/robotics_library/*.cpp")
add_library(robotics_lib 
    ${ROBOTICS_SRC}
    src/robotics/robotics_base.cpp
    src/robotics/robotics.cpp
)
add_library(cuarm_robotics::robotics ALIAS robotics_lib)

target_link_libraries(robotics_lib PUBLIC 
    Threads::Threads  # Modern way to link pthread
    ${yaml_library}
    ${PINOCCHIO_LIBRARIES}
    ${Boost_LIBRARIES}
    robot::platform_utils
    -Wl,--whole-archive robot::platform_utils -Wl,--no-whole-archive # <-- MERGE INTO ONE LIB
)

target_include_directories(robotics_lib PUBLIC 
    ${Boost_INCLUDE_DIR}
    ${EIGEN3_INCLUDE_DIRS}
    ${PINOCCHIO_INCLUDE_DIRS}
    ${PROJECT_SOURCE_DIR}/src
    ${ROBOT_PLATFORM_UTILS_DIR}
)

target_compile_definitions(robotics_lib PUBLIC
    EIGEN_QUATERNIONBASE_PLUGIN="${PROJECT_SOURCE_DIR}/src/3rd_party/Eigen_QuaternionBase_Plugin.hpp" 
    EIGEN_MATRIX_PLUGIN="${PROJECT_SOURCE_DIR}/src/3rd_party/Matrix_initializer_list.h"
)

# Precompile Headers to prevent the compiler from reading Pinocchio's 100k+ lines of templates for every cpp file.
target_precompile_headers(robotics_lib PRIVATE
    <Eigen/Dense>
    <pinocchio/fwd.hpp>
    <pinocchio/multibody/model.hpp>
    <pinocchio/multibody/data.hpp>
    <pinocchio/spatial/se3.hpp>
    <pinocchio/spatial/explog.hpp>
    <pinocchio/algorithm/joint-configuration.hpp>
    <pinocchio/algorithm/kinematics.hpp>
    <pinocchio/algorithm/jacobian.hpp>
    <pinocchio/algorithm/frames.hpp>
    <pinocchio/spatial/force.hpp>
    <pinocchio/parsers/urdf.hpp>
    <pinocchio/algorithm/rnea.hpp>
    <pinocchio/algorithm/crba.hpp>
    <pinocchio/algorithm/center-of-mass.hpp>
    <pinocchio/algorithm/model.hpp>
)

# ############################################################################
# 2. Planner Library (cuarm_robotics::planner)
# ############################################################################

file(GLOB PLANNER_SRC "src/*.cpp")
add_library(planner_lib ${PLANNER_SRC})
add_library(cuarm_robotics::planner ALIAS planner_lib)

target_link_libraries(planner_lib PUBLIC 
    cuarm_robotics::robotics
)
target_include_directories(planner_lib PUBLIC 
    ${PROJECT_SOURCE_DIR}/src
)
###################################Generate Executable######################################
#Generate Executable if this CMakeLists isn't a subdirectory 
if (CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    # set (NAME_LISTS "cuarmV2_2" "cuarmV3_0" "euarm" "eyou_dual_v1")
    set (NAME_LISTS "dual_v1")
    foreach(VAR ${NAME_LISTS})
        add_executable(${VAR}_robotics
            test/main.cpp
            # test/mujoco_sim.cpp
        )
        target_compile_definitions(${VAR}_robotics PRIVATE 
            CONFIG_PREFIX_PATH="${CONFIG_PREFIX_PATH}/${VAR}"
            PROJECT_SOURCE_DIR=\"${PROJECT_SOURCE_DIR}\" 
        )
        target_link_libraries(${VAR}_robotics PRIVATE cuarm_robotics::planner)
    endforeach()
endif()