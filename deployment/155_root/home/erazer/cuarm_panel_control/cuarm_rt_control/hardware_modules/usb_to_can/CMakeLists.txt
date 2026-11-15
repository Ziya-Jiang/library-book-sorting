cmake_minimum_required(VERSION 3.2)
project(usb_to_can)

###################################OS Settings############################
if(WIN32)
    # Windows-specific handling
    if(CMAKE_SIZEOF_VOID_P EQUAL 8)
        message(STATUS "Building for Windows x64")
        set(ARCHI_PATH /windows/x64)
    elseif(CMAKE_SIZEOF_VOID_P EQUAL 4)
        message(STATUS "Building for Windows x86")
        set(ARCHI_PATH /windows/x86)
    endif()
    set(MAIN_TARGET ${PROJECT_SOURCE_DIR}/test/windows_main.cpp)
elseif(UNIX AND NOT APPLE)
    # Linux-specific handling
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64" OR CMAKE_SYSTEM_PROCESSOR MATCHES "arm64")
        message(STATUS "Building for Linux ARM64")
        set(ARCHI_PATH /linux/arm64)
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64" OR CMAKE_SYSTEM_PROCESSOR MATCHES "amd64")
        message(STATUS "Building for Linux AMD64")
        set(ARCHI_PATH /linux/amd64)
    endif()
    set(MAIN_TARGET ${PROJECT_SOURCE_DIR}/test/linux_main.cpp)
else()
    message(WARNING "Unsupported platform")
endif()

###################################Add subdirectory#################################################
cmake_path(GET PROJECT_SOURCE_DIR PARENT_PATH PARENT_DIR)
add_subdirectory(${PARENT_DIR}/motor_data ${CMAKE_BINARY_DIR}/motor_data)
link_libraries(MOTOR_DATA_INTERFACE)

###################################Compile sources into an object library############################
set(USB_TO_CAN_LINK_LIBS 
    ${PROJECT_SOURCE_DIR}/libusb/${ARCHI_PATH}/libusb-1.0.so
    "-lm -lpthread -lrt -Wall" 
)

set (USB_TO_CAN_INCLUDE_DIRS
    ${PROJECT_SOURCE_DIR}/libusb
    ${PROJECT_SOURCE_DIR}/src
)

add_library(USB_TO_CAN_INTERFACE INTERFACE)
target_link_libraries(USB_TO_CAN_INTERFACE INTERFACE ${USB_TO_CAN_LINK_LIBS})
target_include_directories(USB_TO_CAN_INTERFACE INTERFACE ${USB_TO_CAN_INCLUDE_DIRS})
link_libraries(USB_TO_CAN_INTERFACE)

add_library(USB_TO_CAN_OBJ OBJECT
    ${PROJECT_SOURCE_DIR}/libusb/libusb.h
    ${PROJECT_SOURCE_DIR}/src/usb_to_can.h
    ${PROJECT_SOURCE_DIR}/src/usb_to_can.c
    ${PROJECT_SOURCE_DIR}/src/usb_can_rmd_motor.h
    ${PROJECT_SOURCE_DIR}/src/usb_can_rmd_motor.cpp
)
target_link_libraries(USB_TO_CAN_OBJ PRIVATE MOTOR_DATA_INTERFACE)

###################################Generate Executable###########################################
#Generate Executable if this CMakeLists isn't a subdirectory 
if (CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    add_executable(${PROJECT_NAME}
        $<TARGET_OBJECTS:MOTOR_DATA_OBJ> 
        $<TARGET_OBJECTS:USB_TO_CAN_OBJ> 
        ${MAIN_TARGET}
    )
endif()