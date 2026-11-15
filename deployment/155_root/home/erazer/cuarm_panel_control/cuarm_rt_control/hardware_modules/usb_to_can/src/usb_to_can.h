/*
 * This file contains data structs of motors
 */

#ifndef USB_TO_CAN_H_
#define USB_TO_CAN_H_

#include <stdint.h>
#include "libusb.h"
#include <time.h>
#include <pthread.h>
#include <math.h>

#define SPI_SIZE_EACH_USE 2
#define MOTOR_SIZE_EACH_SPI 6
#define USB_DATA_SIZE (SPI_SIZE_EACH_USE * MOTOR_SIZE_EACH_SPI)
#define USB_DATA_LENGTH (USB_DATA_SIZE * 12 + 4)
#define USB_DATA_CHECK_LENGTH (USB_DATA_LENGTH / 4 - 1)

#ifdef __cplusplus
extern "C"
{
#endif

typedef struct {
    uint32_t id[MOTOR_SIZE_EACH_SPI];
    uint8_t  data[MOTOR_SIZE_EACH_SPI][8];
} _spi_data_t;

typedef struct {
    _spi_data_t spi_data[SPI_SIZE_EACH_USE];
    uint32_t checksum;
} _usb_data_t;

typedef union {
    _usb_data_t usb_data;
    uint8_t usb_data_buff[USB_DATA_SIZE];
} _usb_data_u;

struct usb_to_can_board {
    libusb_context* ctx;
    struct libusb_transfer* transfer_in_motors;
    struct libusb_transfer* transfer_out_motors;
    pthread_mutex_t* usb_in_mutex;
    pthread_mutex_t* usb_out_mutex;
    libusb_device_handle* device_handle;
    _usb_data_u* usb_data_u;
    _usb_data_u* usb_cmd_u;
    _usb_data_t* control_cmd;
    _usb_data_t* control_data;
    uint16_t usb_vendor_id;
    uint16_t usb_product_id;
    uint8_t motors_endpoint_in;
    uint8_t motors_endpoint_out;
    struct timespec time_last_in;
    struct timespec time_now_in;
    struct timespec time_last_out;
    struct timespec time_now_out;
};

void usb_motors_in_cbf_wrapper(struct libusb_transfer* _transfer);
void usb_motors_out_cbf_wrapper(struct libusb_transfer* _transfer);

void init_usb_to_can_board(struct usb_to_can_board* board, uint16_t vendor_id, uint16_t product_id, uint8_t motors_epin, uint8_t motors_epout);
void destroy_usb_to_can_board(struct usb_to_can_board* board);
void USB2CAN_SetBuffer(struct usb_to_can_board* board, _usb_data_t* _control_cmd, _usb_data_t* _controller_data);
void USB2CAN_Start_Transfer_Ans(struct usb_to_can_board* board);
void motor_epin_callback(struct usb_to_can_board* board, struct libusb_transfer* _transfer);
void motor_epout_callback(struct usb_to_can_board* board, struct libusb_transfer* _transfer);
void lock_in_mutex(struct usb_to_can_board* board);
void lock_out_mutex(struct usb_to_can_board* board);
void unlock_out_mutex(struct usb_to_can_board* board);
void unlock_in_mutex(struct usb_to_can_board* board);
void Deal_Usb_In_Data(struct usb_to_can_board* board);
void Deal_Usb_Out_Cmd(struct usb_to_can_board* board);

#ifdef __cplusplus
}
#endif

#endif
