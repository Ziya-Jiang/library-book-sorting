#include "usb_to_can.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <string.h>

static uint32_t data_checksum(const uint32_t* data_to_check, uint32_t check_length)
{
    uint32_t t = 0;
    for (int i = 0; i < check_length; i++) {
        t = t ^ data_to_check[i];
    }
    return t;
}

void init_usb_to_can_board(struct usb_to_can_board* board, uint16_t vendor_id, uint16_t product_id, uint8_t motors_epin, uint8_t motors_epout)
{
    board->usb_cmd_u = (_usb_data_u*)malloc(sizeof(_usb_data_u));
    board->usb_data_u = (_usb_data_u*)malloc(sizeof(_usb_data_u));
    board->usb_in_mutex = (pthread_mutex_t*)malloc(sizeof(pthread_mutex_t));
    board->usb_out_mutex = (pthread_mutex_t*)malloc(sizeof(pthread_mutex_t));

    pthread_mutex_init(board->usb_in_mutex, NULL);
    pthread_mutex_init(board->usb_out_mutex, NULL);

    clock_gettime(CLOCK_MONOTONIC, &board->time_last_out);
    clock_gettime(CLOCK_MONOTONIC, &board->time_now_out);
    clock_gettime(CLOCK_MONOTONIC, &board->time_last_in);
    clock_gettime(CLOCK_MONOTONIC, &board->time_now_in);

    libusb_init(&board->ctx);
    board->transfer_out_motors = libusb_alloc_transfer(0);
    board->transfer_in_motors = libusb_alloc_transfer(0);
    board->device_handle = libusb_open_device_with_vid_pid(board->ctx, vendor_id, product_id);
    if (libusb_kernel_driver_active(board->device_handle, 0x00)) {
        int success = libusb_detach_kernel_driver(board->device_handle, 0x00);
        if (success != 0) {
            fprintf(stderr, "Detach Driver Failed!\n");
            exit(EXIT_FAILURE);
        }
    }
    
    int claim_interface = libusb_claim_interface(board->device_handle, 0x00);
    if (claim_interface != 0) {
        fprintf(stderr, "Claim Driver Failed!\n");
        exit(EXIT_FAILURE);
    } else {
        printf("Claim USB Device OK!\n");
    }

    board->usb_vendor_id = vendor_id;
    board->usb_product_id = product_id;
    board->motors_endpoint_in = motors_epin;
    board->motors_endpoint_out = motors_epout;
}

void destroy_usb_to_can_board(struct usb_to_can_board* board)
{
    free(board->usb_cmd_u);
    free(board->usb_data_u);
    free(board->usb_in_mutex);
    free(board->usb_out_mutex);

    libusb_free_transfer(board->transfer_in_motors);
    libusb_free_transfer(board->transfer_out_motors);
    libusb_release_interface(board->device_handle, 0);
    libusb_close(board->device_handle);
    libusb_exit(board->ctx);
}

void USB2CAN_SetBuffer(struct usb_to_can_board* board, _usb_data_t* _control_cmd, _usb_data_t* _controller_data)
{
    board->control_cmd = _control_cmd;
    board->control_data = _controller_data;
}

void motor_epin_callback(struct usb_to_can_board* board, struct libusb_transfer* _transfer)
{
    if (_transfer->status != LIBUSB_TRANSFER_COMPLETED) {
        printf("Motor Ep81 IN Error! Transfer again!\n");
    } else {
        Deal_Usb_In_Data(board);
        clock_gettime(CLOCK_MONOTONIC, &board->time_now_in);
        double time_used = (board->time_now_in.tv_sec - board->time_last_in.tv_sec) * 1e6 +
                           (board->time_now_in.tv_nsec - board->time_last_in.tv_nsec) / 1e3;
        // printf("[Time Interval In]: %.2f us\n", time_used);
        board->time_last_in = board->time_now_in;
        libusb_submit_transfer(_transfer);
    }
}

void motor_epout_callback(struct usb_to_can_board* board, struct libusb_transfer* _transfer)
{
    if (_transfer->status != LIBUSB_TRANSFER_COMPLETED) {
        printf("Motor Ep01 OUT Error! Transfer again!\n");
    } else {
        Deal_Usb_Out_Cmd(board);
        clock_gettime(CLOCK_MONOTONIC, &board->time_now_out);
        double time_used = (board->time_now_out.tv_sec - board->time_last_out.tv_sec) * 1e6 +
                           (board->time_now_out.tv_nsec - board->time_last_out.tv_nsec) / 1e3;
        // printf("[Time Interval Out]: %.2f us\n", time_used);
        board->time_last_out = board->time_now_out;
        libusb_submit_transfer(_transfer);
    }
}

void Deal_Usb_In_Data(struct usb_to_can_board* board)
{
    uint8_t spi_id = 0;
    uint8_t module_id = 0;
    uint32_t t = data_checksum((uint32_t*)board->usb_data_u, USB_DATA_CHECK_LENGTH);
    if (board->usb_data_u->usb_data.checksum == t) {
        for (uint8_t i = 0; i < 2 * SPI_SIZE_EACH_USE; i++) {
            spi_id = i / 2;
            module_id = i % 2;
            for (uint8_t j = 0; j < 3; ++j) {
                board->control_data->spi_data[spi_id].id[module_id * 3 + j] = board->usb_data_u->usb_data.spi_data[spi_id].id[module_id * 3 + j];
                for (uint8_t k = 0; k < 8; ++k)
                    board->control_data->spi_data[spi_id].data[module_id * 3 + j][k] = board->usb_data_u->usb_data.spi_data[spi_id].data[module_id * 3 + j][k];
            }
        }
    } else {
        printf("usb data checksum error\n");
    }
}

void Deal_Usb_Out_Cmd(struct usb_to_can_board* board)
{
    uint8_t spi_id = 0;
    uint8_t module_id = 0;
    for (uint8_t i = 0; i < 2 * SPI_SIZE_EACH_USE; i++) {
        spi_id = i / 2;
        module_id = i % 2;
        for (uint8_t j = 0; j < 3; ++j) {
            board->usb_cmd_u->usb_data.spi_data[spi_id].id[module_id * 3 + j] = board->control_cmd->spi_data[spi_id].id[module_id * 3 + j];
            for (uint8_t k = 0; k < 8; ++k)
                board->usb_cmd_u->usb_data.spi_data[spi_id].data[module_id * 3 + j][k] = board->control_cmd->spi_data[spi_id].data[module_id * 3 + j][k];
        }
    }
    board->usb_cmd_u->usb_data.checksum = data_checksum((uint32_t*)board->usb_cmd_u, USB_DATA_CHECK_LENGTH);
}

void USB2CAN_Start_Transfer_Ans(struct usb_to_can_board* board)
{
    libusb_fill_interrupt_transfer(board->transfer_out_motors, board->device_handle, board->motors_endpoint_out,
                                   board->usb_cmd_u->usb_data_buff, USB_DATA_LENGTH, usb_motors_out_cbf_wrapper, board, 0);
    libusb_fill_interrupt_transfer(board->transfer_in_motors, board->device_handle, board->motors_endpoint_in,
                                   board->usb_data_u->usb_data_buff, USB_DATA_LENGTH, usb_motors_in_cbf_wrapper, board, 0);
    libusb_submit_transfer(board->transfer_in_motors);
    libusb_submit_transfer(board->transfer_out_motors);
    if ((!board->transfer_out_motors->status) & (!board->transfer_in_motors->status)) {
        printf("[Good] All endpoints start transferring!\n");
    } else {
        printf("[Bad] Some endpoints not work\n");
        exit(EXIT_FAILURE);
    }
}

void usb_motors_in_cbf_wrapper(struct libusb_transfer* _transfer)
{
    struct usb_to_can_board* temp = (struct usb_to_can_board*)_transfer->user_data;
    motor_epin_callback(temp, _transfer);
}

void usb_motors_out_cbf_wrapper(struct libusb_transfer* _transfer)
{
    struct usb_to_can_board* temp = (struct usb_to_can_board*)_transfer->user_data;
    motor_epout_callback(temp, _transfer);
}

void lock_in_mutex(struct usb_to_can_board* board)
{
    pthread_mutex_lock(board->usb_in_mutex);
}

void lock_out_mutex(struct usb_to_can_board* board)
{
    pthread_mutex_lock(board->usb_out_mutex);
}

void unlock_out_mutex(struct usb_to_can_board* board)
{
    pthread_mutex_unlock(board->usb_out_mutex);
}

void unlock_in_mutex(struct usb_to_can_board* board)
{
    pthread_mutex_unlock(board->usb_in_mutex);
}
