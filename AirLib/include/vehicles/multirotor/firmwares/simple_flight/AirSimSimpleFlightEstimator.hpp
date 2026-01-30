// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef msr_airlib_AirSimSimpleFlightEstimator_hpp
#define msr_airlib_AirSimSimpleFlightEstimator_hpp

#include "firmware/interfaces/CommonStructs.hpp"
#include "AirSimSimpleFlightCommon.hpp"
#include "physics/Kinematics.hpp"
#include "physics/Environment.hpp"
#include "common/Common.hpp"
#include "common/ClockFactory.hpp"
#include <mutex>
#include <deque>

namespace msr
{
namespace airlib
{

    class AirSimSimpleFlightEstimator : public simple_flight::IStateEstimator
    {
    public:
        virtual ~AirSimSimpleFlightEstimator() {}

        void setGroundTruthKinematics(const Kinematics::State* kinematics, const Environment* environment)
        {
            std::lock_guard<std::mutex> lock(state_mutex_);
            gt_kinematics_ = kinematics;
            environment_ = environment;
            // Note: gt_kinematics_ is a live pointer updated by physics engine.
            // Ring buffer is populated continuously via recordGTSnapshot() in getters.
        }

        void setVIOKinematics(const Kinematics::State& vio_state)
        {
            std::lock_guard<std::mutex> lock(state_mutex_);

            // Find GT position at the time VIO was measured (t - L)
            TTimePoint now = ClockFactory::get()->nowNanos();
            Vector3r gt_at_vio_time = findGTAtTime(now, vio_delay_sec_);

            // Synchronized residual: pure sensor error only
            // ε = z_VIO(t-L) - x_GT(t-L)
            vio_position_residual_ = vio_state.pose.position - gt_at_vio_time;
            has_vio_data_ = true;
        }

        void setUseVIO(bool use_vio)
        {
            std::lock_guard<std::mutex> lock(state_mutex_);
            use_vio_ = use_vio;
        }

        bool isUsingVIO() const
        {
            std::lock_guard<std::mutex> lock(state_mutex_);
            return use_vio_;
        }

        // HYBRID STATE ESTIMATOR (Re-propagation)
        // - Position: x_GT(t) + ε  where ε = z_VIO(t-L) - x_GT(t-L)
        // - Orientation/angular: GT (fast, 333Hz)
        // - Velocity: GT (fast, for stable inner loop)

        virtual simple_flight::Axis3r getAngles() const override
        {
            std::lock_guard<std::mutex> lock(state_mutex_);
            simple_flight::Axis3r angles;
            VectorMath::toEulerianAngle(gt_kinematics_->pose.orientation,
                                        angles.pitch(),
                                        angles.roll(),
                                        angles.yaw());
            return angles;
        }

        virtual simple_flight::Axis3r getAngularVelocity() const override
        {
            std::lock_guard<std::mutex> lock(state_mutex_);
            const auto& angular = gt_kinematics_->twist.angular;

            simple_flight::Axis3r conv;
            conv.x() = angular.x();
            conv.y() = angular.y();
            conv.z() = angular.z();

            return conv;
        }

        virtual simple_flight::Axis3r getPosition() const override
        {
            std::lock_guard<std::mutex> lock(state_mutex_);
            recordGTSnapshot();
            if (use_vio_ && has_vio_data_) {
                // Re-propagation: x̂_t = x_GT(t) + ε
                Vector3r estimated = gt_kinematics_->pose.position + vio_position_residual_;
                return AirSimSimpleFlightCommon::toAxis3r(estimated);
            }
            return AirSimSimpleFlightCommon::toAxis3r(gt_kinematics_->pose.position);
        }

        virtual simple_flight::Axis3r transformToBodyFrame(const simple_flight::Axis3r& world_frame_val) const override
        {
            std::lock_guard<std::mutex> lock(state_mutex_);
            const Vector3r& vec = AirSimSimpleFlightCommon::toVector3r(world_frame_val);
            const Vector3r& trans = VectorMath::transformToBodyFrame(vec, gt_kinematics_->pose.orientation);
            return AirSimSimpleFlightCommon::toAxis3r(trans);
        }

        virtual simple_flight::Axis3r getLinearVelocity() const override
        {
            std::lock_guard<std::mutex> lock(state_mutex_);
            return AirSimSimpleFlightCommon::toAxis3r(gt_kinematics_->twist.linear);
        }

        virtual simple_flight::Axis4r getOrientation() const override
        {
            std::lock_guard<std::mutex> lock(state_mutex_);
            return AirSimSimpleFlightCommon::toAxis4r(gt_kinematics_->pose.orientation);
        }

        virtual simple_flight::GeoPoint getGeoPoint() const override
        {
            std::lock_guard<std::mutex> lock(state_mutex_);
            return AirSimSimpleFlightCommon::toSimpleFlightGeoPoint(environment_->getState().geo_point);
        }

        virtual simple_flight::GeoPoint getHomeGeoPoint() const override
        {
            std::lock_guard<std::mutex> lock(state_mutex_);
            return AirSimSimpleFlightCommon::toSimpleFlightGeoPoint(environment_->getHomeGeoPoint());
        }

        virtual simple_flight::KinematicsState getKinematicsEstimated() const override
        {
            std::lock_guard<std::mutex> lock(state_mutex_);
            recordGTSnapshot();
            simple_flight::KinematicsState state;

            // Position: GT + VIO residual (re-propagation)
            if (use_vio_ && has_vio_data_) {
                Vector3r estimated = gt_kinematics_->pose.position + vio_position_residual_;
                state.position = AirSimSimpleFlightCommon::toAxis3r(estimated);
            } else {
                state.position = AirSimSimpleFlightCommon::toAxis3r(gt_kinematics_->pose.position);
            }

            // Everything else: GT (fast feedback for stable control)
            state.linear_velocity = AirSimSimpleFlightCommon::toAxis3r(gt_kinematics_->twist.linear);
            state.orientation = AirSimSimpleFlightCommon::toAxis4r(gt_kinematics_->pose.orientation);
            state.angular_velocity = AirSimSimpleFlightCommon::toAxis3r(gt_kinematics_->twist.angular);
            state.linear_acceleration = AirSimSimpleFlightCommon::toAxis3r(gt_kinematics_->accelerations.linear);
            state.angular_acceleration = AirSimSimpleFlightCommon::toAxis3r(gt_kinematics_->accelerations.angular);

            return state;
        }

    private:
        // Record current GT position into the ring buffer.
        // Called from const getters (at controller frequency ~333Hz) to keep the buffer populated.
        // setGroundTruthKinematics() is only called once at init, so we cannot rely on it.
        void recordGTSnapshot() const
        {
            if (gt_kinematics_ == nullptr) return;
            TTimePoint now = ClockFactory::get()->nowNanos();
            // Avoid duplicate entries if called multiple times in the same tick
            if (!gt_history_.empty() && gt_history_.back().time == now) return;
            gt_history_.push_back({now, gt_kinematics_->pose.position});
            if (gt_history_.size() > MAX_GT_HISTORY) {
                gt_history_.pop_front();
            }
        }

        // Find GT position at (now - delay_sec) using ring buffer + linear interpolation
        // Position-only interpolation (Lerp) - no quaternion involved
        Vector3r findGTAtTime(TTimePoint now, float delay_sec) const
        {
            if (gt_history_.empty()) {
                return gt_kinematics_->pose.position;
            }

            TTimePoint target_time = now - static_cast<TTimePoint>(delay_sec * 1.0E9);

            // Search backwards for bracketing entries
            for (int i = static_cast<int>(gt_history_.size()) - 1; i >= 0; --i) {
                if (gt_history_[i].time <= target_time) {
                    if (i + 1 < static_cast<int>(gt_history_.size())) {
                        // Linear interpolation between gt_history_[i] and gt_history_[i+1]
                        double dt_total = static_cast<double>(gt_history_[i + 1].time - gt_history_[i].time);
                        double dt_target = static_cast<double>(target_time - gt_history_[i].time);
                        float alpha = (dt_total > 0) ? static_cast<float>(dt_target / dt_total) : 0.0f;
                        return gt_history_[i].position * (1.0f - alpha)
                             + gt_history_[i + 1].position * alpha;
                    }
                    return gt_history_[i].position;
                }
            }
            // target_time is older than all history entries
            return gt_history_.front().position;
        }

        struct TimestampedPosition {
            TTimePoint time;
            Vector3r position;
        };

        const Kinematics::State* gt_kinematics_ = nullptr;
        const Environment* environment_ = nullptr;

        // GT position history ring buffer (~1 sec at 333Hz)
        // mutable: updated from const getters via recordGTSnapshot()
        mutable std::deque<TimestampedPosition> gt_history_;
        static constexpr size_t MAX_GT_HISTORY = 333;

        // VIO re-propagation state
        Vector3r vio_position_residual_{0, 0, 0};
        float vio_delay_sec_ = 0.2f;
        bool use_vio_ = false;
        bool has_vio_data_ = false;

        mutable std::mutex state_mutex_;
    };
}
} //namespace
#endif
