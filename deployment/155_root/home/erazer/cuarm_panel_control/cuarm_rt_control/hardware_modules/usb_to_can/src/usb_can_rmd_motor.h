#ifndef USB_CAN_RMD_MOTOR_H
#define USB_CAN_RMD_MOTOR_H
#include "usb_to_can.h"
#include "motor_data.h"

#define BRAKE2_OPEN 0xA2
#define BRAKE2_CLOSE 0xA1
#define BRAKE2_DISCHARGE 0xA3
#define BRAKE1_OPEN 0xA5
#define BRAKE1_CLOSE 0xA4
#define BRAKE1_DISCHARGE 0xA6
const uint16_t usb2can_vendor_id = 0x1111;
const uint16_t usb2can_product_id = 0x2222;
const uint8_t  motors_ep_in = 0x81;
const uint8_t  motors_ep_out = 0x01;

typedef struct {
    float torque[2]={0,};
    float encoder[2];
} sensor_data;

void can_rmd_set_position(unsigned char buffer[], float value, motor_data *motor);
void can_rmd_set_velocity(unsigned char buffer[], float value, motor_data *motor);
void can_rmd_set_torque(unsigned char buffer[], float value);
void can_rmd_decoding(const unsigned char buffer[], motor_data *motor);
void can_read_torque_sensor(unsigned char buffer[]);
void can_brake_control(unsigned char buffer[], unsigned char cmd);
void can_sensor_decoding(const unsigned char buffer[], sensor_data *sensor);
#endif