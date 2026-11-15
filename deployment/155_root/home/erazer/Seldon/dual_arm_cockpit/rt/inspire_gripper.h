#pragma once
#include <algorithm>
#include <cmath>
#include <memory>
#include <stdexcept>
#include "src/gripper/abstract_gripper.h"
#include "inspire_hand.h"
#include "socketcan_hand.h"

// Matches the original InspireHandWrapper panel order and conversion:
// thumbRotate, thumbFlex, index, middle, ring, little (0..1) <->
// little, ring, middle, index, thumbFlex, thumbRotate (1000..0).
// This adapter reads actual hand angles; commands are never reported as measured.
// Temporary left-hand placeholder: preserve the six-joint wire protocol while
// performing no CAN initialization and no physical hand commands.
class SeldonDisabledHand : public AbstractGripper {
public:
    explicit SeldonDisabledHand(const std::string& name)
        : AbstractGripper(name, 6, std::vector<std::array<float,2>>(6, {0,1})) {}
    RetCode open() override { return RetCode::kSuccess; }
    RetCode close() override { return RetCode::kSuccess; }
    RetCode setPosition(const std::vector<float>& values) override {
        if (values.size() != 6) return RetCode::kInvalidInput;
        for (float value : values) {
            if (!std::isfinite(value)) return RetCode::kInvalidInput;
        }
        return RetCode::kSuccess;  // Intentionally ignored: left hand is offline.
    }
    RetCode getPosition(std::vector<float>& values) override {
        if (values.size() != 6) return RetCode::kInvalidInput;
        std::fill(values.begin(), values.end(), 0.0f);
        return RetCode::kSuccess;
    }
};

class SeldonInspireGripper : public AbstractGripper {
    class CheckedTransport : public IHandCanTransport {
    public:
        std::shared_ptr<IHandCanTransport> inner;
        bool failed = false;
        explicit CheckedTransport(std::shared_ptr<IHandCanTransport> transport): inner(std::move(transport)) {}
        int request(int i, uint32_t id, const HandCanFrame& req, HandCanFrame& response, int timeout) override {
            int result = inner->request(i, id, req, response, std::min(timeout, 10));
            // A short read must not fabricate zeros for the missing joints.
            const bool read = ((req.id >> 26) & 7) == 0;
            if (result != HAND_CAN_SUCCESS || (read && response.dlc < req.data[0])) {
                failed = true;
                return HAND_CAN_FAILED_RECEIVE;
            }
            return result;
        }
        int waitForResponse(int i, uint32_t id, HandCanFrame& response, int timeout) override {
            int result = inner->waitForResponse(i, id, response, std::min(timeout, 10));
            if (result != HAND_CAN_SUCCESS) failed = true;
            return result;
        }
    };
    int ids_[2];
    std::shared_ptr<IHandCanTransport> injected_;
    std::shared_ptr<CheckedTransport> transport_;
    std::unique_ptr<InspireHand> hand_;
public:
    explicit SeldonInspireGripper(const std::string& name, int hand_id = 2,
                                 std::shared_ptr<IHandCanTransport> transport = nullptr)
        : AbstractGripper(name, 6, std::vector<std::array<float,2>>(6, {0,1000})),
          ids_{hand_id,1}, injected_(std::move(transport)) {}
    RetCode open() override {
        try {
            transport_ = std::make_shared<CheckedTransport>(injected_ ? injected_ : std::make_shared<SeldonSocketHandTransport>());
            hand_ = std::make_unique<InspireHand>(ids_, transport_);
            std::vector<float> measured(6);
            if (getPosition(measured) != RetCode::kSuccess) return RetCode::kFailed;
            // Seed duplicate suppression from feedback: taking control holds the hand.
            last_set_command_ = roundCommand(measured);
            return RetCode::kSuccess;
        } catch (const std::exception&) { return RetCode::kFailed; }
    }
    RetCode close() override { hand_.reset(); transport_.reset(); return RetCode::kSuccess; }
    RetCode setPosition(const std::vector<float>& values) override {
        if (!hand_ || values.size()!=6) return RetCode::kInvalidInput;
        for (float v: values) if (!std::isfinite(v) || v<0 || v>1000) return RetCode::kInvalidInput;
        auto rounded = roundCommand(values);
        if (isDuplicateCommand(rounded)) return RetCode::kSuccess;
        int16_t angles[6];
        for (int i=0; i<6; ++i) angles[i]=static_cast<int16_t>(rounded[i]);
        transport_->failed=false;
        hand_->setAngles(0, angles);
        if (transport_->failed) return RetCode::kFailed;
        last_set_command_=rounded;
        return RetCode::kSuccess;
    }
    RetCode getPosition(std::vector<float>& values) override {
        if (!hand_ || values.size()!=6) return RetCode::kInvalidInput;
        if (hand_->getAngles(0)!=HAND_CAN_SUCCESS) return RetCode::kFailed;
        for (int i=0; i<6; ++i) {
            float v=hand_->joint_angle_[0][i];
            if (v<0 || v>1000) return RetCode::kFailed;
            values[i]=v;
        }
        return RetCode::kSuccess;
    }
    std::vector<float> convertPanelHardware(const std::vector<float>& values) override {
        if (values.size()!=6) return {};
        std::vector<float> result(6);
        for (int i=0; i<6; ++i) result[i]=1.0f-values[5-i];
        return result;
    }
};
