#ifndef CUARM_STATUS_VELOCITY_H
#define CUARM_STATUS_VELOCITY_H
#include <cmath>
#include <limits>

// 0x700 status packets contain position/current, NOT velocity. Estimate their
// velocity from measured position and monotonic receipt time. Never reuse a
// servo packet's velocity indefinitely. No motor command is sent to obtain it.
class StatusVelocity {
public:
    enum Source { unavailable, servo_report, position_difference };
    double observe(double position, double now) {
        if (!std::isfinite(position) || !std::isfinite(now)) {
            reset();
            return invalid();
        }
        if (!have_anchor_ || now <= last_time_ || now - last_time_ > 0.100) {
            reset();
            have_anchor_ = true;
            anchor_position_ = position;
            anchor_time_ = last_time_ = now;
            return invalid();
        }
        last_time_ = now;
        const double dt = now - anchor_time_;
        if (dt >= 0.020) {
            estimated_ = (position - anchor_position_) / dt;
            estimate_time_ = now;
            anchor_position_ = position;
            anchor_time_ = now;
        }
        if (std::isfinite(reported_) && now - report_time_ <= 0.020) {
            source_ = servo_report;
            return reported_;
        }
        if (std::isfinite(estimated_) && now - estimate_time_ <= 0.100) {
            source_ = position_difference;
            return estimated_;
        }
        source_ = unavailable;
        return invalid();
    }
    double servo(double position, double velocity, double now) {
        observe(position, now);
        if (!std::isfinite(position) || !std::isfinite(velocity) || !std::isfinite(now)) {
            reset();
            return invalid();
        }
        reported_ = velocity;
        report_time_ = now;
        source_ = servo_report;
        return velocity;
    }
    Source source() const { return source_; }
private:
    static double invalid() { return std::numeric_limits<double>::quiet_NaN(); }
    void reset() {
        have_anchor_ = false;
        reported_ = estimated_ = invalid();
        source_ = unavailable;
    }
    bool have_anchor_ = false;
    double anchor_position_ = 0, anchor_time_ = 0, last_time_ = 0;
    double reported_ = invalid(), report_time_ = 0;
    double estimated_ = invalid(), estimate_time_ = 0;
    Source source_ = unavailable;
};
#endif
