#pragma once
// Independent dual-arm profile. Startup reads actual positions without moving
// fingers. Runtime position is explicitly COMMAND ECHO, not grasp confirmation.
#include "src/gripper/abstract_gripper.h"
#include <fcntl.h>
#include <sys/file.h>
#include <sys/ioctl.h>
#include <termios.h>
#include <poll.h>
#include <unistd.h>
#include <chrono>
#include <mutex>
#include <memory>
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <thread>
#include <string>
#include <iostream>

namespace dual_arm {
// Continuous J6 only: accept stable left-ID1 startup measurements down to zero.
// Measurement envelope only; commands remain within the configured device range.
constexpr int kLeftStartupUndershoot = 100;
constexpr int kStartupStableSpan = 5;
// ID1's measured closed-end position may settle just below its commanded
// lower endpoint. This is a startup measurement tolerance, NOT a command range.
inline int startup_reference(int id, const std::vector<int>& positions,
                             int device_lo, int device_hi, float command_lo, float command_hi) {
    if (positions.empty() || device_lo >= device_hi || !std::isfinite(command_lo) ||
        !std::isfinite(command_hi) || command_lo >= command_hi ||
        command_lo < device_lo || command_hi > device_hi)
        throw std::runtime_error("Invalid gripper device/configured limits");
    auto span=std::minmax_element(positions.begin(),positions.end());
    const int actual=positions.back();
    if(positions.size()!=3 || *span.second-*span.first>kStartupStableSpan)
        throw std::runtime_error("Initial gripper position is unstable");
    if (*span.first >= command_lo && *span.second <= command_hi) return actual;
    // Only the observed left-ID1 lower endpoint is covered; upper/right errors
    // and a genuinely moving or widely out-of-range encoder remain faults.
    if (id==1 && device_lo==100 && device_hi==4000 &&
        command_lo==100 && command_hi==4000 &&
        *span.first>=device_lo-kLeftStartupUndershoot && *span.second<=command_lo+kStartupStableSpan)
        return static_cast<int>(command_lo);
    throw std::runtime_error("Stable gripper position outside startup endpoint envelope: id="+
        std::to_string(id)+" actual="+std::to_string(actual)+
        " sample_min="+std::to_string(*span.first)+" sample_max="+std::to_string(*span.second)+
        " command_min="+std::to_string(command_lo)+" command_max="+std::to_string(command_hi));
}
inline std::vector<unsigned char> frame(int id, int instruction, const std::vector<unsigned char>& data) {
    if(id<1 || id>253 || data.size()>250) throw std::runtime_error("Invalid addressed JOHO frame");
    std::vector<unsigned char> out{255,255,(unsigned char)id,(unsigned char)(data.size()+2),(unsigned char)instruction};
    out.insert(out.end(),data.begin(),data.end()); unsigned sum=0;
    for(size_t i=2;i<out.size();++i)sum+=out[i];out.push_back((unsigned char)~sum);return out;
}
class Bus {
    int fd_=-1;
public:
    std::mutex mutex;
    explicit Bus(const std::string& path) {
        fd_=::open(path.c_str(),O_RDWR|O_NOCTTY|O_NONBLOCK|O_CLOEXEC);
        if(fd_<0)throw std::runtime_error("Cannot open JOHO bus");
        try {
            if(flock(fd_,LOCK_EX|LOCK_NB) || ioctl(fd_,TIOCEXCL))throw std::runtime_error("JOHO bus occupied");
            termios t{};if(tcgetattr(fd_,&t))throw std::runtime_error("JOHO termios read failed");
            cfmakeraw(&t);cfsetispeed(&t,B115200);cfsetospeed(&t,B115200);
            t.c_cflag|=CLOCAL|CREAD;t.c_cc[VMIN]=0;t.c_cc[VTIME]=0;
            if(tcsetattr(fd_,TCSANOW,&t))throw std::runtime_error("JOHO termios setup failed");
            tcflush(fd_,TCIFLUSH);
        } catch(...) {::close(fd_);fd_=-1;throw;}
    }
    ~Bus(){if(fd_>=0){ioctl(fd_,TIOCNXCL);::close(fd_);}}
    void send(int id,int instruction,const std::vector<unsigned char>& data) {
        auto p=frame(id,instruction,data);size_t n=0;
        const auto end=std::chrono::steady_clock::now()+std::chrono::milliseconds(10);
        while(n<p.size()) {
            auto count=::write(fd_,p.data()+n,p.size()-n);
            if(count>0)n+=count;
            else if(errno!=EAGAIN && errno!=EINTR)throw std::runtime_error("JOHO write failed");
            if(std::chrono::steady_clock::now()>=end)throw std::runtime_error("JOHO write deadline");
        }
    }
    std::vector<unsigned char> read(int id,int reg,int count) {
        tcflush(fd_,TCIFLUSH);send(id,2,{(unsigned char)reg,(unsigned char)count});
        std::vector<unsigned char> data;const auto end=std::chrono::steady_clock::now()+std::chrono::milliseconds(80);
        while(std::chrono::steady_clock::now()<end) {
            pollfd p{fd_,POLLIN,0};if(poll(&p,1,5)<=0)continue;
            unsigned char buf[64];auto n=::read(fd_,buf,sizeof(buf));if(n>0)data.insert(data.end(),buf,buf+n);
            if(data.size()>=size_t(count+6)) {
                if(data.size()!=size_t(count+6) || data[0]!=255 || data[1]!=245 || data[2]!=id || data[3]!=count+2 || data[4]!=0)
                    throw std::runtime_error("JOHO reply identity/length/protection fault");
                unsigned sum=0;for(size_t i=2;i<data.size();++i)sum+=data[i];
                if((sum&255)!=255)throw std::runtime_error("JOHO checksum fault");
                return {data.begin()+5,data.end()-1};
            }
        }
        throw std::runtime_error("JOHO read timeout");
    }
};
class Gripper : public AbstractGripper {
    std::shared_ptr<Bus> bus_;int id_;bool opened_=false;
public:
    Gripper(const std::string& name,std::shared_ptr<Bus> bus,int id):AbstractGripper(name,1,{{0,4000}}),bus_(bus),id_(id){}
    RetCode open() override {
        try {
            std::lock_guard<std::mutex> l(bus_->mutex);
            auto mode=bus_->read(id_,0x1c,1),torque=bus_->read(id_,0x28,1),limits=bus_->read(id_,0x09,4);
            if(mode[0]!=1 || torque[0]!=1)throw std::runtime_error("JOHO requires existing position mode and torque");
            int lo=(limits[0]<<8)|limits[1], hi=(limits[2]<<8)|limits[3];
            if(joint_hardware_limits_[0][0]<lo || joint_hardware_limits_[0][1]>hi)throw std::runtime_error("Configured gripper range exceeds device limits");
            std::vector<int> positions;
            for(int i=0;i<3;++i) {
                if(i)std::this_thread::sleep_for(std::chrono::milliseconds(20));
                auto pos=bus_->read(id_,0x38,2);positions.push_back((pos[0]<<8)|pos[1]);
            }
            std::cerr<<"JOHO startup id="<<id_<<" raw_positions=["<<positions[0]<<","<<positions[1]<<","<<positions[2]
                     <<"] device_limits=["<<lo<<","<<hi<<"] command_limits=["
                     <<joint_hardware_limits_[0][0]<<","<<joint_hardware_limits_[0][1]<<"]"<<std::endl;
            int reference=startup_reference(id_,positions,lo,hi,joint_hardware_limits_[0][0],joint_hardware_limits_[0][1]);
            if(reference!=positions.back())
                std::cerr<<"JOHO startup endpoint normalization id="<<id_<<" actual="<<positions.back()
                         <<" reference="<<reference<<" tolerance_counts="<<kLeftStartupUndershoot
                         <<" stable_span_counts="<<kStartupStableSpan
                         <<" command_range_unchanged=true position_write_sent=false feedback=normalized_command_reference"<<std::endl;
            last_set_command_[0]=reference;opened_=true;return RetCode::kSuccess;
        }catch(const std::exception& e){std::cerr<<"JOHO id="<<id_<<" initialization: "<<e.what()<<std::endl;return RetCode::kFailed;}
    }
    RetCode close() override {opened_=false;return RetCode::kSuccess;}
    RetCode setPosition(const std::vector<float>& val) override {
        if(!opened_ || val.size()!=1 || !std::isfinite(val[0]) || val[0]<joint_hardware_limits_[0][0] || val[0]>joint_hardware_limits_[0][1])return RetCode::kInvalidInput;
        auto command=roundCommand(val);if(isDuplicateCommand(command))return RetCode::kSuccess;
        int target=int(command[0]);
        try {std::lock_guard<std::mutex> l(bus_->mutex);bus_->send(id_,3,{0x2a,(unsigned char)(target>>8),(unsigned char)target,0x07,0xd0});}
        catch(...){return RetCode::kFailed;}
        last_set_command_[0]=target;return RetCode::kSuccess;
    }
    RetCode getPosition(std::vector<float>& val) override {if(!opened_ || val.size()!=1)return RetCode::kFailed;val[0]=last_set_command_[0];return RetCode::kSuccess;}
    std::vector<float> convertPanelHardware(const std::vector<float>& val) override {return {1.f-val.at(0)};}
};
}
