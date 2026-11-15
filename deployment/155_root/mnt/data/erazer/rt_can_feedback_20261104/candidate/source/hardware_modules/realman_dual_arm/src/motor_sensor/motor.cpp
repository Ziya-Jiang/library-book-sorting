#include "motor.h"
#include <chrono>

namespace {
int32_t read_i32_le(const uint8_t* bytes) {
    const uint32_t raw = uint32_t(bytes[0]) | (uint32_t(bytes[1]) << 8) |
        (uint32_t(bytes[2]) << 16) | (uint32_t(bytes[3]) << 24);
    int32_t value;
    std::memcpy(&value, &raw, sizeof(value));
    return value;
}
double feedback_time() {
    return std::chrono::duration<double>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}
}

  

#define TIMEOUT_MS 10

// Constructor
WhjMotor::WhjMotor(uint32_t motor_id, int can_dev_index, double gear_ratio, int sign, double init_mdh_rad)
    : motor_id_(motor_id), pos_(0.0f), vel_(0.0f), cur_(0.0f), last_pos_(0.0),
      can_dev_index_(can_dev_index), gear_ratio_(gear_ratio), sign_(sign), init_mdh_rad_(init_mdh_rad) {}

// Member function implementations
int WhjMotor::set_IAP() {
    return write_register(ADDR_IAP_FLAG, 0x00, 3);
}

int WhjMotor::enable() {
    return write_register(ADDR_ENABLE_FLAG, 0x01, 3);
}

int WhjMotor::disable() {
    return write_register(ADDR_ENABLE_FLAG, 0x00, 3);
}

int WhjMotor::clear_error() {
    return write_register(ADDR_CLEAR_ERROR, 0x01, 3);
}

int WhjMotor::select_mode(uint8_t control_mode) {
    return write_register(ADDR_WORK_MODE, control_mode, 3);
}

int WhjMotor::set_velocity(double vel_rad_s) {
    uint32_t can_id = motor_id_ + 0x300;
    double vel_rpm = vel_rad_s * 30.0 / 3.141592653589793;
    int32_t vel_units = (int32_t)(vel_rpm * 500);
    return servo_control(can_id, vel_units);
}

int WhjMotor::set_current(double cur_amp) {
    uint32_t can_id = motor_id_ + 0x400;
    int32_t cur_units = (int32_t)(cur_amp * 1000);
    return servo_control(can_id, cur_units);
}

int WhjMotor::set_absposition(double pos_rad) {
    double pos_deg = pos_rad * 180.0 / 3.141592653589793;
    uint32_t can_id = motor_id_ + 0x200;
    int32_t pos_units = (int32_t)(pos_deg * 10000);
    return servo_control(can_id, pos_units);
}

int WhjMotor::read_status(void) {
    uint32_t can_id = motor_id_ + 0x600;
    uint8_t dlc = 0;
    uint8_t send_data;
    uint32_t resp_Id = 0;
    uint8_t resp_max_data[64];
    uint8_t resp_dlc = 0;
    int send_status = canfd_send(can_dev_index_, 0, can_id, &send_data, dlc);
    control_success_ = 0;
    if (send_status != 1) {
        std::ostringstream oss;
        oss << "Failed to send read_status data to motor ID: " << (int)motor_id_;
        std::cerr << oss.str() << std::endl;
        LOG_ERROR(oss.str());
        return -1;
    }
    return 0;
}

int WhjMotor::write_register(uint8_t addr, uint8_t values, uint8_t dlc) {
    uint8_t write_cmd = 0x02;
    uint8_t send_data[3] = {write_cmd, addr, values};
    uint32_t resp_Id = 0;
    uint8_t resp_max_data[64];
    uint8_t resp_dlc = 0;
    write_success_ = 0; 
    int send_status = canfd_send(can_dev_index_, 0, motor_id_, send_data, dlc);
    if (send_status != 1) {
        std::ostringstream oss;
        oss << "Failed to send write_register data to motor ID: " << (int)motor_id_;
        std::cerr << oss.str() << std::endl;
        LOG_ERROR(oss.str());
        return -1;
    }
    return 0;
}

int WhjMotor::decode_motor_data(uint8_t* data) {
    pos_ = read_i32_le(data + 8) * 0.0001 * 3.141592653589793 / 180.0;
    const double velocity = read_i32_le(data + 4) * 0.02 * 3.141592653589793 / 30.0;
    vel_ = status_velocity_.servo(pos_, velocity, feedback_time());
    cur_ = read_i32_le(data) * 0.001;
    enable_status_ = (int(data[13]) << 8) | int(data[12]);
    error_code_ = (int(data[15]) << 8) | int(data[14]);
    ++feedback_sequence_;
    return 0;
}

int WhjMotor::decode_motor_data2(uint8_t* data) {
    pos_ = read_i32_le(data + 8) * 0.0001 * 3.141592653589793 / 180.0;
    vel_ = status_velocity_.observe(pos_, feedback_time());
    error_code_ = (int(data[1]) << 8) | int(data[0]);
    enable_status_ = int(data[6]);
    brake_status_ = int(data[7]);
    cur_ = read_i32_le(data + 12) * 0.001;
    ++feedback_sequence_;
    return 0;
}

int WhjMotor::servo_control(uint32_t can_id, int32_t controlTarget) {
    uint8_t send_data[4];
    send_data[0] = (uint8_t)(controlTarget & 0xFFFF);
    send_data[1] = (uint8_t)((controlTarget >> 8) & 0xFFFF);
    send_data[2] = (uint8_t)((controlTarget >> 16) & 0xFFFF);
    send_data[3] = (uint8_t)((controlTarget >> 24) & 0xFFFF);
    uint8_t dlc = 4;
    uint32_t resp_Id = 0;
    uint8_t resp_max_data[64];
    uint8_t resp_dlc = 0;
    control_success_ = 0;
    int send_status = canfd_send(can_dev_index_, 0, can_id, send_data, dlc);
    if (send_status != 1) {
        std::ostringstream oss;
        oss<< "Failed to send servo_control data to motor ID: " << (int)motor_id_;
        std::cerr << oss.str() << std::endl;
        LOG_ERROR(oss.str());
        return -1;
    }
    return 0;
}