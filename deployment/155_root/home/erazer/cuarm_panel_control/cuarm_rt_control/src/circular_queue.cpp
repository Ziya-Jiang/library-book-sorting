#include "circular_queue.h"

#include <mutex>

static constexpr int kSaveBufferSize = 200000;
static int queue_head = 0, queue_tail = 0, queue_size = 0;
static std::mutex queue_lock;
static save_data queue_buffer[kSaveBufferSize];

bool queue_insert(save_data new_node)
{   
    bool success = false;
    if (queue_lock.try_lock()) {
        if (queue_size < kSaveBufferSize) {
            queue_buffer[queue_head ++] = new_node;
            if (queue_head >= kSaveBufferSize) {
                queue_head = 0;
            }
            queue_size ++;
            success = true;
        }else {
            printf("The queue is full !!!\n");
        }
        queue_lock.unlock();
    }
    return success;
}

bool queue_poptop(save_data* new_node) 
{   
    bool success = false;
    if (queue_lock.try_lock()) {
        if (queue_size > 0) {
            queue_size--;
            *new_node = queue_buffer[queue_tail ++];
            if (queue_tail >= kSaveBufferSize) {
                queue_tail = 0;
            }
            success = true;
        }
        queue_lock.unlock();
    }
    return success;
}

int queue_get_size(){
    return queue_size;
}
