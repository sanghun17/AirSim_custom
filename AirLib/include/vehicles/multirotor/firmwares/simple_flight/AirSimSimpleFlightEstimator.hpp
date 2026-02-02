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

    // =============================================================================
    // HYBRID STATE ESTIMATOR with VIO Re-propagation
    // =============================================================================
    //
    // When VIO is enabled, fuses VIO measurements with ground truth (GT) using
    // a re-propagation approach that compensates for VIO processing delay (~0.2s):
    //
    //   estimated(t) = GT(t) + [VIO(t-L) - GT(t-L)] - baseline
    //
    // where:
    //   t   = current time
    //   L   = vio_delay_sec_ (~0.2s, FAST-LIVO2 processing latency)
    //   VIO(t-L) = VIO measurement at the delayed time
    //   GT(t-L)  = GT state interpolated from ring buffer at delayed time
    //   baseline = residual captured at VIO enable, so effective correction starts at 0
    //
    // FRAME CONVENTIONS (all values in this class are NED world frame):
    //   - Position: NED (North, East, Down) world frame
    //   - Velocity: NED world frame
    //   - Orientation: NED quaternion (w, x, y, z) with Hamilton convention
    //   - The ROS wrapper handles all frame conversions before calling setVIOKinematics():
    //       Position:    camera_init FLU → ENU axis swap (y,x,-z) → NED + offset
    //       Velocity:    camera_init FLU → FRD (x,-y,-z) → R(GT_init) → NED
    //       Orientation: camera_init FLU quat → axis swap (w,y,x,-z) → offset mult → NED
    //   - Orientation and angular velocity always use GT (fast, ~333Hz from physics engine)
    //
    // CAUTION:
    //   - vio_delay_sec_ is hardcoded to 0.2s. Adjust if FAST-LIVO2 latency changes.
    //   - Ring buffer is populated from const getters at controller frequency (~333Hz).
    //     setGroundTruthKinematics() is only called once at init (sets the pointer).
    //   - Baseline mechanism: first VIO residual after enable becomes the baseline,
    //     absorbing any systematic VIO bias at the moment of enable.
    // =============================================================================

    class AirSimSimpleFlightEstimator : public simple_flight::IStateEstimator
    {
    public:
        virtual ~AirSimSimpleFlightEstimator() {}

        void setGroundTruthKinematics(const Kinematics::State* kinematics, const Environment* environment)
        {
            std::lock_guard<std::mutex> lock(state_mutex_);
            gt_kinematics_ = kinematics;
            environment_ = environment;
        }

        // Receive VIO measurement. All fields must be in NED world frame
        // (the ROS wrapper handles camera_init FLU → NED conversion).
        void setVIOKinematics(const Kinematics::State& vio_state)
        {
            std::lock_guard<std::mutex> lock(state_mutex_);

            // Find GT position and velocity at the time VIO was measured (t - L)
            TTimePoint now = ClockFactory::get()->nowNanos();
            Vector3r gt_pos_at_vio_time, gt_vel_at_vio_time;
            findGTAtTime(now, vio_delay_sec_, gt_pos_at_vio_time, gt_vel_at_vio_time);

            // Synchronized residuals: pure sensor error only
            // ε_pos = z_VIO_pos(t-L) - x_GT_pos(t-L)
            // ε_vel = z_VIO_vel(t-L) - v_GT_vel(t-L)
            vio_position_residual_ = vio_state.pose.position - gt_pos_at_vio_time;
            vio_velocity_residual_ = vio_state.twist.linear - gt_vel_at_vio_time;

            // Captures current drift as baseline so effective ε starts at 0
            if (vio_needs_baseline_) {
                vio_position_baseline_ = vio_position_residual_;
                vio_velocity_baseline_ = vio_velocity_residual_;
                vio_needs_baseline_ = false;
            }

            has_vio_data_ = true;
        }

        void setUseVIO(bool use_vio)
        {
            std::lock_guard<std::mutex> lock(state_mutex_);
            if (use_vio && !use_vio_) {
                // false→true: wait for first VIO callback to set baseline
                vio_needs_baseline_ = true;
                has_vio_data_ = false; // force GT until first VIO arrives
            }
            if (!use_vio && use_vio_) {
                // true→false: clean reset
                vio_position_baseline_ = Vector3r::Zero();
                vio_velocity_baseline_ = Vector3r::Zero();
                vio_needs_baseline_ = false;
                has_vio_data_ = false;
            }
            use_vio_ = use_vio;
        }

        bool isUsingVIO() const
        {
            std::lock_guard<std::mutex> lock(state_mutex_);
            return use_vio_;
        }

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
            // Position: GT + effective VIO residual (re-propagation with baseline)
            if (use_vio_ && has_vio_data_) {
                Vector3r effective = vio_position_residual_ - vio_position_baseline_;
                Vector3r estimated = gt_kinematics_->pose.position + effective;
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
            // Velocity: GT + effective VIO residual (re-propagation with baseline)
            if (use_vio_ && has_vio_data_) {
                Vector3r effective = vio_velocity_residual_ - vio_velocity_baseline_;
                Vector3r estimated = gt_kinematics_->twist.linear + effective;
                return AirSimSimpleFlightCommon::toAxis3r(estimated);
            }
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

            // Position: GT + effective VIO residual (re-propagation with baseline)
            if (use_vio_ && has_vio_data_) {
                Vector3r eff_pos = vio_position_residual_ - vio_position_baseline_;
                state.position = AirSimSimpleFlightCommon::toAxis3r(gt_kinematics_->pose.position + eff_pos);
            } else {
                state.position = AirSimSimpleFlightCommon::toAxis3r(gt_kinematics_->pose.position);
            }

            // Velocity: GT + effective VIO residual (re-propagation with baseline)
            if (use_vio_ && has_vio_data_) {
                Vector3r eff_vel = vio_velocity_residual_ - vio_velocity_baseline_;
                state.linear_velocity = AirSimSimpleFlightCommon::toAxis3r(gt_kinematics_->twist.linear + eff_vel);
            } else {
                state.linear_velocity = AirSimSimpleFlightCommon::toAxis3r(gt_kinematics_->twist.linear);
            }
            state.orientation = AirSimSimpleFlightCommon::toAxis4r(gt_kinematics_->pose.orientation);
            state.angular_velocity = AirSimSimpleFlightCommon::toAxis3r(gt_kinematics_->twist.angular);
            state.linear_acceleration = AirSimSimpleFlightCommon::toAxis3r(gt_kinematics_->accelerations.linear);
            state.angular_acceleration = AirSimSimpleFlightCommon::toAxis3r(gt_kinematics_->accelerations.angular);

            return state;
        }

    private:
        // Record current GT state into the ring buffer for delay compensation.
        // Called from const getters (at controller frequency ~333Hz).
        // gt_kinematics_ is a live pointer updated by physics engine each tick.
        void recordGTSnapshot() const
        {
            if (gt_kinematics_ == nullptr) return;
            TTimePoint now = ClockFactory::get()->nowNanos();
            // Avoid duplicate entries if called multiple times in the same tick
            if (!gt_history_.empty() && gt_history_.back().time == now) return;
            gt_history_.push_back({now, gt_kinematics_->pose.position, gt_kinematics_->twist.linear});
            if (gt_history_.size() > MAX_GT_HISTORY) {
                gt_history_.pop_front();
            }
        }

        // Find GT position and velocity at (now - delay_sec) using ring buffer + linear interpolation.
        void findGTAtTime(TTimePoint now, float delay_sec, Vector3r& out_position, Vector3r& out_velocity) const
        {
            if (gt_history_.empty()) {
                out_position = gt_kinematics_->pose.position;
                out_velocity = gt_kinematics_->twist.linear;
                return;
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
                        out_position = gt_history_[i].position * (1.0f - alpha)
                                     + gt_history_[i + 1].position * alpha;
                        out_velocity = gt_history_[i].velocity * (1.0f - alpha)
                                     + gt_history_[i + 1].velocity * alpha;
                        return;
                    }
                    out_position = gt_history_[i].position;
                    out_velocity = gt_history_[i].velocity;
                    return;
                }
            }
            // target_time is older than all history entries
            out_position = gt_history_.front().position;
            out_velocity = gt_history_.front().velocity;
        }

        struct TimestampedState {
            TTimePoint time;
            Vector3r position;
            Vector3r velocity;
        };

        const Kinematics::State* gt_kinematics_ = nullptr;
        const Environment* environment_ = nullptr;

        // GT position+velocity history ring buffer (~1 sec at 333Hz)
        // mutable: updated from const getters via recordGTSnapshot()
        mutable std::deque<TimestampedState> gt_history_;
        static constexpr size_t MAX_GT_HISTORY = 333;

        // VIO re-propagation state (all residuals in NED world frame)
        Vector3r vio_position_residual_{0, 0, 0};
        Vector3r vio_velocity_residual_{0, 0, 0};
        Vector3r vio_position_baseline_{0, 0, 0};
        Vector3r vio_velocity_baseline_{0, 0, 0};
        float vio_delay_sec_ = 0.2f;   // FAST-LIVO2 processing delay (hardcoded)
        bool use_vio_ = false;
        bool has_vio_data_ = false;
        bool vio_needs_baseline_ = false;

        mutable std::mutex state_mutex_;
    };
}
} //namespace
#endif
