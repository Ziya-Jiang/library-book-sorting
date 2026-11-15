#include "usb_can_rmd_motor.h"
#include <stdio.h>

void can_rmd_set_position(unsigned char buffer[], float value, motor_data *motor)
{   
    int k = (RMD_V999 == motor->type)? 6 : 1;
    int val = (int) (value * 100 * k);
    buffer[0] = 0xA4;
    for (int i = 1; i <= 3; i++)
        buffer[i] = 0x00;       
    buffer[4] = (unsigned char)(val & 0XFF);
    buffer[5] = (unsigned char)((val >> 8) & 0XFF);
    buffer[6] = (unsigned char)((val >> 16) & 0XFF);
    buffer[7] = (unsigned char)((val >> 24) & 0XFF);
}

void can_rmd_set_velocity(unsigned char buffer[], float value, motor_data *motor)
{
    int k = (RMD_V999 == motor->type)? 6 : 1;
    int val = (int) (value * 100 * k);
    buffer[0] = 0xA2;
    for (int i = 1; i <= 3; i++)
        buffer[i] = 0x00;       
    buffer[4] = (unsigned char)(val & 0XFF);
    buffer[5] = (unsigned char)((val >> 8) & 0XFF);
    buffer[6] = (unsigned char)((val >> 16) & 0XFF);
    buffer[7] = (unsigned char)((val >> 24) & 0XFF);    
}

void can_rmd_set_torque(unsigned char buffer[], float value)
{
    int val = (int) (value * 100);
    buffer[0] = 0xA1;
    for (int i = 1; i <= 3; i++)
        buffer[i] = 0x00;       
    buffer[4] = (unsigned char)(val & 0XFF);
    buffer[5] = (unsigned char)((val >> 8) & 0XFF);
    buffer[6] = 0x00;
    buffer[7] = 0x00;    
}

void can_read_torque_sensor(unsigned char buffer[])
{
    buffer[0] = 0x55;
    for (int i = 1; i < 8; i++)
        buffer[i] = 0x00;
}

void can_brake_control(unsigned char buffer[], unsigned char cmd)
{
    buffer[0] = 0x0B;
    buffer[1] = cmd;
    for (int i = 2; i < 8; i++)
        buffer[i] = 0x00;
}

//RMD V1 motor
void can_rmd_decoding(const unsigned char buffer[], motor_data *motor)
{
    if (motor->resolution <= 0 || motor->gear_ratio <= 0) {
        printf("ERROR: This motor is not initialzied \n");
    }

    if (RMD_V1 == motor->type) {
        motor->torque   = (int16_t)((buffer[3] << 8) + buffer[2]) *  33.0 / 2048;
        motor->velocity = (int16_t)((buffer[5] << 8) + buffer[4]) / motor->gear_ratio;
        motor->position = (int16_t)((buffer[7] << 8) + buffer[6]) * 360.0 / motor->gear_ratio / motor->resolution;
    } else if (RMD_V3 == motor->type) {
        motor->torque   = (int16_t)((buffer[3] << 8) + buffer[2]) * 0.01;
        motor->velocity = (int16_t)((buffer[5] << 8) + buffer[4]);
        motor->position = (int16_t)((buffer[7] << 8) + buffer[6]);
    } else if (RMD_V4 == motor->type ||  (RMD_V4_1 == motor->type)) {
        motor->torque   = (int16_t)((buffer[3] << 8) + buffer[2]) * 33.0 / 2048;
        motor->velocity = (int16_t)((buffer[5] << 8) + buffer[4]);
        motor->position = (int16_t)((buffer[7] << 8) + buffer[6]) * 360.0 / motor->gear_ratio / motor->resolution;
    } else if (RMD_V999 == motor->type){
        motor->torque   = (int16_t)((buffer[3] << 8) + buffer[2]) * 33.0 / 2048 / 1.6;
        motor->velocity = (int16_t)((buffer[5] << 8) + buffer[4]) * 360.0 / motor->gear_ratio / motor->resolution;
        motor->position = (int16_t)((buffer[7] << 8) + buffer[6]) * 360.0 / motor->gear_ratio / motor->resolution;
    }
   
   
    motor->raw_position = motor->position;
    if (0 == motor->has_first_data) {
        motor->last_position = motor->position;
        motor->circle = 0;
        motor->has_first_data = 1;
    } else {
        if (RMD_V4_1 == motor->type){
            if (motor->position + motor->circle * 360.0 - motor->last_position < -175.0)
                motor->circle ++;
            else if (motor->position + motor->circle * 360.0 - motor->last_position > 175.0)
                motor->circle --;
            motor->position = motor->position + motor->circle * 360.0;
        }else{
            if (motor->position + motor->circle * 360.0 / motor->gear_ratio - motor->last_position < -180.0 / motor->gear_ratio)
                motor->circle ++;
            else if (motor->position + motor->circle * 360.0 / motor->gear_ratio - motor->last_position > 180.0 / motor->gear_ratio)
                motor->circle --;
            motor->position = motor->position + motor->circle * 360.0 / motor->gear_ratio;
        }
        motor->last_position = motor->position;
    }
}

void can_sensor_decoding(const unsigned char buffer[], sensor_data *sensor)
{
    // for (int i = 0;i<8;++i)
    //     printf("%x ", buffer[i]);
    // printf("\n");
    float Enc_res = 65536.f;
    int16_t adc16bits_1 = ((int16_t)buffer[0] << 8) | ((int16_t)buffer[1]);
    int16_t adc16bits_2 = ((int16_t)buffer[2] << 8) | ((int16_t)buffer[3]);
    int16_t Enc16bits_1 = ((int16_t)buffer[4] << 8) | ((int16_t)buffer[5]);
    int16_t Enc16bits_2 = ((int16_t)buffer[6] << 8) | ((int16_t)buffer[7]);

    sensor->torque[0]  = adc16bits_1 * 0.001;
    sensor->torque[1]  = adc16bits_2 * 0.001;
    sensor->encoder[0] = Enc16bits_1 * 360 / Enc_res;
    sensor->encoder[1] = Enc16bits_2 * 360 / Enc_res;
}


