#include "can_fd.h"
#include "zcan.h"

#define CANFD_DISCOVERY_LIMIT 16

typedef struct {
    int device_index;
    char serial[21];
} CANFD_DeviceIdentity;

static CANFD_DeviceIdentity discovered_devices[CANFD_DISCOVERY_LIMIT];
static int discovered_device_count = 0;
static int discovery_complete = 0;

static void canfd_discover_devices(void) {
    if (discovery_complete) return;
    discovery_complete = 1;

    int consecutive_misses = 0;
    for (int device_index = 0;
         device_index < CANFD_DISCOVERY_LIMIT;
         ++device_index) {
        if (!VCI_OpenDevice(USBCANFD, device_index, 0)) {
            ++consecutive_misses;
            if (discovered_device_count > 0 && consecutive_misses >= 2) break;
            continue;
        }
        consecutive_misses = 0;

        ZCAN_DEV_INF info = {0};
        if (VCI_ReadBoardInfo(USBCANFD, device_index, &info) &&
            discovered_device_count < CANFD_DISCOVERY_LIMIT) {
            CANFD_DeviceIdentity* identity =
                &discovered_devices[discovered_device_count++];
            identity->device_index = device_index;
            memcpy(identity->serial, info.sn, sizeof(info.sn));
            identity->serial[sizeof(identity->serial) - 1] = '\0';
            printf("CANFD device index %d serial %s\n",
                   identity->device_index, identity->serial);
        }
        VCI_CloseDevice(USBCANFD, device_index);
    }
}

int canfd_find_device_by_serial(const char* serial) {
    if (!serial || serial[0] == '\0') return -1;
    canfd_discover_devices();

    for (int i = 0; i < discovered_device_count; ++i) {
        if (strcmp(discovered_devices[i].serial, serial) == 0) {
            return discovered_devices[i].device_index;
        }
    }

    fprintf(stderr, "CANFD device serial not found: %s\n", serial);
    return -1;
}

// CANFD初始化（波特率固定为仲裁段1Mbps，数据段5Mbps）
int canfd_init(int dev_idx, int chn_idx) {
    // 打开设备
    if (!VCI_OpenDevice(USBCANFD, dev_idx, 0)) {
        fprintf(stderr, "Failed to open device\n");
        return -1;
    }

    // 固定波特率配置（仲裁段1Mbps，数据段5Mbps）
    ZCAN_INIT init_cfg = {
        .clk = 60000000,    // 设备时钟60MHz
        .mode = 0,          // 正常模式
        
        // 仲裁段配置（1Mbps）
        .aset = {
            .tseg1 = 2,    // 时间段1
            .tseg2 = 0,     // 时间段2
            .sjw = 0,       // 同步跳转宽度
            .smp = 0,       // 采样次数
            .brp = 11        // 波特率分频
        },
        
        // 数据段配置（5Mbps）
        .dset = {
            .tseg1 = 1,     // 时间段1
            .tseg2 = 0,     // 时间段2
            .sjw = 0,       // 同步跳转宽度
            .smp = 0,       // 采样次数
            .brp = 2        // 波特率分频
        }
    };

    // 初始化通道
    if (!VCI_InitCAN(USBCANFD, dev_idx, chn_idx, &init_cfg)) {
        fprintf(stderr, "Failed to initialize channel\n");
        VCI_CloseDevice(USBCANFD, dev_idx);
        return -1;
    }

    // 启动通道
    if (!VCI_StartCAN(USBCANFD, dev_idx, chn_idx)) {
        fprintf(stderr, "Failed to start channel\n");
        VCI_CloseDevice(USBCANFD, dev_idx);
        return -1;
    }

    // 默认启用终端电阻
    uint32_t term = 1;
    VCI_SetReference(USBCANFD, dev_idx, chn_idx, CMD_CAN_TRES, &term);

    return 0;
}

// CANFD发送单帧（参数不变）
int canfd_send(int dev_idx, int chn_idx, uint32_t id, 
               const uint8_t* data, uint8_t len) {
    if (len > CANFD_MAX_DATA_LEN) {
        fprintf(stderr, "Data length exceeds 64 bytes\n");
        return -1;
    }
    ZCAN_FD_MSG frame = {
        .hdr = {
            .inf = {
                .fmt = 1,       // CANFD格式
                .sef = 0,  // 扩展帧标识
                .brs = 1        // 启用加速传输
            },
            .id = id,
            .chn = (uint8_t)chn_idx,
            .len = len
        }
    };
    memcpy(frame.dat, data, len);
    // printf("send: can_id 0x%x, dlc 0x%x data: ",frame.hdr.id, frame.hdr.len);
    // for (int i = 0;i<frame.hdr.len;++i)
    // {
    //     printf("0x%x ",frame.dat[i]);
    // }
    // printf("\n");
    int send_status = VCI_TransmitFD(USBCANFD, dev_idx, chn_idx, &frame, 1);
    // printf("send status : %d\n", send_status);
    
    return send_status;
}

// CANFD接收单帧（参数不变）
int canfd_receive(int dev_idx, int chn_idx, uint32_t* id, 
                  uint8_t* data, uint8_t* len, 
                  int timeout_ms) {
    ZCAN_FD_MSG frame = {0};
    if (!id || !len) return -3;
    *id = 0; *len = 0;
    int received = VCI_ReceiveFD(USBCANFD, dev_idx, chn_idx, &frame, 1, timeout_ms);
    
    if (received < 0) return -1;
    if (received == 0) return -2;
    if (received != 1 || frame.hdr.len > CANFD_MAX_DATA_LEN) return -3;
    //printf("frame.hdr.id %x, frame.hdr.len %x\n", frame.hdr.id, frame.hdr.len);
    *id = frame.hdr.id;
    *len = frame.hdr.len;

    if (data) 
    {  
        memcpy(data, frame.dat, frame.hdr.len);
        // printf("receive: can_id 0x%x, dlc 0x%x data: ",frame.hdr.id, frame.hdr.len);
        // for (int i = 0;i<frame.hdr.len;++i)
        // {
        //     printf("0x%x ",data[i]);
        // }
        // printf("\n------------\n");
    }

    return 0;
}

// CANFD关闭设备（参数不变）
void canfd_close(int dev_idx, int chn_idx) {
    VCI_ResetCAN(USBCANFD, dev_idx, chn_idx);
    VCI_CloseDevice(USBCANFD, dev_idx);
}