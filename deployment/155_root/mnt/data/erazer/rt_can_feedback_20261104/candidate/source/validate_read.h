#ifndef RT_VALIDATE_READ_H
#define RT_VALIDATE_READ_H
#include <cmath>
#include <vector>
inline bool valid_arm_sample(bool read_succeeded, int size,
    const std::vector<float>& position, const std::vector<float>& velocity,
    const std::vector<float>& torque) {
    if (!read_succeeded || size < 1 || position.size() != static_cast<unsigned>(size) ||
        velocity.size() != position.size() || torque.size() != position.size()) return false;
    for (int j=0; j<size; ++j)
        if (!std::isfinite(position[j]) || !std::isfinite(velocity[j]) || !std::isfinite(torque[j])) return false;
    return true;
}
#endif
