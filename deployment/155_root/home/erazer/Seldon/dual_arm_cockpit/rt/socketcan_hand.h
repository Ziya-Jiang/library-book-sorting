#pragma once
#include <algorithm>
#include "hand_can_transport.h"
#include <linux/can.h>
#include <linux/can/raw.h>
#include <sys/socket.h>
#include <sys/ioctl.h>
#include <net/if.h>
#include <poll.h>
#include <unistd.h>
#include <cerrno>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <chrono>

// Explicitly verified hand-only adapter. Never enumerate interfaces, run sudo,
// change interface state, or touch can_chassis. Setup is performed separately.
class SeldonSocketHandTransport final : public IHandCanTransport {
    int fd_=-1;
public:
    SeldonSocketHandTransport() {
        std::ifstream input("/sys/class/net/can1/device/../serial");std::string serial;input>>serial;
        if(serial!="004800315246571720323135")throw std::runtime_error("Hand CAN USB identity mismatch");
        fd_=::socket(PF_CAN,SOCK_RAW|SOCK_NONBLOCK,CAN_RAW);
        if(fd_<0)throw std::runtime_error("Cannot open dedicated hand CAN socket");
        ifreq req{};std::strncpy(req.ifr_name,"can1",IFNAMSIZ-1);
        if(::ioctl(fd_,SIOCGIFINDEX,&req)<0){::close(fd_);fd_=-1;throw std::runtime_error("Missing can1");}
        sockaddr_can addr{};addr.can_family=AF_CAN;addr.can_ifindex=req.ifr_ifindex;
        if(::bind(fd_,reinterpret_cast<sockaddr*>(&addr),sizeof(addr))<0){::close(fd_);fd_=-1;throw std::runtime_error("Cannot bind dedicated hand CAN socket");}
        can_filter filter{};filter.can_id=CAN_EFF_FLAG|2;filter.can_mask=CAN_EFF_FLAG|CAN_RTR_FLAG|0x3fff;
        if(::setsockopt(fd_,SOL_CAN_RAW,CAN_RAW_FILTER,&filter,sizeof(filter))<0){::close(fd_);fd_=-1;throw std::runtime_error("Cannot set hand CAN receive filter");}
    }
    ~SeldonSocketHandTransport()override {if(fd_>=0)::close(fd_);}
    int request(int hand_idx,uint32_t hand_id,const HandCanFrame& req,HandCanFrame& response,int timeout_ms)override {
        if(hand_idx!=0||hand_id!=2||!req.is_extended||req.dlc>8||(req.id&0x3fff)!=2||timeout_ms<=0)return HAND_CAN_FAILED_RECEIVE;
        // Drain only this socket's queue; no other CAN clients are affected.
        can_frame incoming{};while(::read(fd_,&incoming,sizeof(incoming))==sizeof(incoming)){}
        can_frame outgoing{};outgoing.can_id=CAN_EFF_FLAG|req.id;outgoing.can_dlc=req.dlc;std::memcpy(outgoing.data,req.data.data(),req.dlc);
        if(::write(fd_,&outgoing,sizeof(outgoing))!=sizeof(outgoing))return HAND_CAN_FAILED_RECEIVE;
        auto deadline=std::chrono::steady_clock::now()+std::chrono::milliseconds(timeout_ms);
        while(std::chrono::steady_clock::now()<deadline){
            int remaining=std::chrono::duration_cast<std::chrono::milliseconds>(deadline-std::chrono::steady_clock::now()).count();pollfd pfd{fd_,POLLIN,0};if(::poll(&pfd,1,std::max(1,remaining))<=0)continue;
            if(::read(fd_,&incoming,sizeof(incoming))!=sizeof(incoming))continue;
            if((incoming.can_id&(CAN_EFF_FLAG|CAN_RTR_FLAG|CAN_ERR_FLAG))!=CAN_EFF_FLAG||(incoming.can_id&CAN_EFF_MASK)!=req.id||incoming.can_dlc>8)continue;
            response.id=incoming.can_id&CAN_EFF_MASK;response.is_extended=true;response.dlc=incoming.can_dlc;std::memcpy(response.data.data(),incoming.data,incoming.can_dlc);return HAND_CAN_SUCCESS;
        }
        return HAND_CAN_FAILED_RECEIVE;
    }
    int waitForResponse(int,uint32_t,HandCanFrame&,int)override{return HAND_CAN_FAILED_RECEIVE;}
};
