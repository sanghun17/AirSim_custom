// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef msr_airlib_AirSimSimpleFlightEstimator_hpp
#define msr_airlib_AirSimSimpleFlightEstimator_hpp

#include "firmware/interfaces/CommonStructs.hpp"
#include "AirSimSimpleFlightCommon.hpp"
#include "physics/Kinematics.hpp"
#include "physics/Environment.hpp"
#include "common/Common.hpp"
#include <mutex>
#include <chrono>

namespace msr
{
namespace airlib
{

    class AirSimSimpleFlightEstimator : public simple_flight::IStateEstimator
    {
    public:
        AirSimSimpleFlightEstimator() = default;
        virtual ~AirSimSimpleFlightEstimator() = default;

        //====================================================================
        // VIO Support: Dual-source state estimation
        //====================================================================

        /**
         * Enable or disable VIO-based state estimation
         * @param use_vio: true = use VIO odometry, false = use ground truth (default)
         */
        void setUseVIO(bool use_vio)
        {
            {
                std::lock_guard<std::mutex> lock(state_mutex_);
                use_vio_ = use_vio;
            }
            // std::cout << "[VIO DEBUG ESTIMATOR] setUseVIO: " << use_vio << std::endl;
        }

        /**
         * Check if VIO mode is enabled
         */
        bool isUsingVIO() const
        {
            std::lock_guard<std::mutex> lock(state_mutex_);
            return use_vio_;
        }

        /**
         * Update VIO-based kinematics from external source (e.g., ROS via RPC)
         * @param vio_state: VIO state estimate
         */
        void setVIOKinematics(const Kinematics::State& vio_state)
        {
            // Fast path: update data with minimal lock time
            {
                std::lock_guard<std::mutex> lock(state_mutex_);
                vio_kinematics_ = vio_state;
                vio_last_update_time_ = std::chrono::steady_clock::now();
                has_vio_data_ = true;
            }

            // // Debug output AFTER releasing lock (throttled to avoid spam)
            // static int call_count = 0;
            // if (++call_count % 10 == 0) {  // Print every 10th call only
            //     std::cout << "[VIO DEBUG ESTIMATOR] VIO kinematics updated (#" << call_count << ")" << std::endl;
            // }
        }

        /**
         * Check if VIO data is available and recent
         * @param timeout_sec: Maximum age of VIO data (default: 2.0 seconds)
         * @return true if VIO data is valid
         */
        bool hasValidVIOData(double timeout_sec = 2.0) const
        {
            std::lock_guard<std::mutex> lock(state_mutex_);
            return hasValidVIODataUnsafe(timeout_sec);
        }

        //====================================================================
        // Ground Truth Setting (stores GT, doesn't replace VIO)
        //====================================================================

        /**
         * Set ground truth kinematics from physics simulation
         * This is called every frame by the simulation loop
         */
        void setGroundTruthKinematics(const Kinematics::State* kinematics, const Environment* environment)
        {
            std::lock_guard<std::mutex> lock(state_mutex_);
            gt_kinematics_ = kinematics;
            environment_ = environment;
        }

        //====================================================================
        // State Getters (select GT or VIO based on mode)
        //====================================================================

        virtual simple_flight::Axis3r getAngles() const override
        {
            const Kinematics::State* source = getKinematicsSource();
            simple_flight::Axis3r angles;
            VectorMath::toEulerianAngle(source->pose.orientation,
                                        angles.pitch(),
                                        angles.roll(),
                                        angles.yaw());
            return angles;
        }

        virtual simple_flight::Axis3r getAngularVelocity() const override
        {
            std::lock_guard<std::mutex> lock(state_mutex_);

            // TEMPORARY TEST: Always use GT angular velocity
            // VIO angular velocity seems incorrect (too small)
            // TODO: Fix VIO angular velocity frame/scaling
            if (gt_kinematics_) {
                const auto& anguler = gt_kinematics_->twist.angular;
                simple_flight::Axis3r conv;
                conv.x() = anguler.x();
                conv.y() = anguler.y();
                conv.z() = anguler.z();
                return conv;
            }

            // Fallback to VIO (should not happen if GT is available)
            const Kinematics::State* source = getKinematicsSource();
            const auto& anguler = source->twist.angular;

            simple_flight::Axis3r conv;
            conv.x() = anguler.x();
            conv.y() = anguler.y();
            conv.z() = anguler.z();

            return conv;
        }

        virtual simple_flight::Axis3r getPosition() const override
        {
            const Kinematics::State* source = getKinematicsSource();
            return AirSimSimpleFlightCommon::toAxis3r(source->pose.position);
        }

        virtual simple_flight::Axis3r transformToBodyFrame(const simple_flight::Axis3r& world_frame_val) const override
        {
            const Kinematics::State* source = getKinematicsSource();
            const Vector3r& vec = AirSimSimpleFlightCommon::toVector3r(world_frame_val);
            const Vector3r& trans = VectorMath::transformToBodyFrame(vec, source->pose.orientation);
            return AirSimSimpleFlightCommon::toAxis3r(trans);
        }

        virtual simple_flight::Axis3r getLinearVelocity() const override
        {
            const Kinematics::State* source = getKinematicsSource();
            return AirSimSimpleFlightCommon::toAxis3r(source->twist.linear);
        }

        virtual simple_flight::Axis4r getOrientation() const override
        {
            const Kinematics::State* source = getKinematicsSource();
            return AirSimSimpleFlightCommon::toAxis4r(source->pose.orientation);
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
            const Kinematics::State* source = getKinematicsSource();

            simple_flight::KinematicsState state;
            state.position = AirSimSimpleFlightCommon::toAxis3r(source->pose.position);
            state.orientation = AirSimSimpleFlightCommon::toAxis4r(source->pose.orientation);
            state.linear_velocity = AirSimSimpleFlightCommon::toAxis3r(source->twist.linear);
            state.angular_velocity = AirSimSimpleFlightCommon::toAxis3r(source->twist.angular);
            state.linear_acceleration = AirSimSimpleFlightCommon::toAxis3r(source->accelerations.linear);
            state.angular_acceleration = AirSimSimpleFlightCommon::toAxis3r(source->accelerations.angular);

            return state;
        }

    private:
        /**
         * Check if VIO data is valid without locking (call with lock held)
         */
        bool hasValidVIODataUnsafe(double timeout_sec = 2.0) const
        {
            if (!has_vio_data_) {
                return false;
            }

            auto now = std::chrono::steady_clock::now();
            std::chrono::duration<double> elapsed = now - vio_last_update_time_;
            return elapsed.count() < timeout_sec;
        }

        /**
         * Select kinematics source based on current mode
         * Returns VIO if enabled and valid, otherwise GT
         */
        const Kinematics::State* getKinematicsSource() const
        {
            std::lock_guard<std::mutex> lock(state_mutex_);

            // TEMPORARY FIX: Always use GT directly to avoid synchronization issues
            // Problem: VIO data from RPC is a delayed copy, but GT is always current via pointer
            // This causes orientation conflict between controller (using VIO copy) and physics (using GT)
            // Solution: Use GT pointer directly even in VIO mode for testing

            // SAFETY: gt_kinematics_ should never be null if setGroundTruthKinematics was called
            if (gt_kinematics_ == nullptr) {
                return &vio_kinematics_; // Fallback to VIO
            }

            return gt_kinematics_;  // Always return GT for now

            // ORIGINAL CODE (commented out to avoid sync issues):
            // If VIO mode enabled AND VIO data is valid, use VIO
            // if (use_vio_ && hasValidVIODataUnsafe()) {
            //     return &vio_kinematics_;
            // }
        }

        // Ground truth kinematics (from physics simulation)
        const Kinematics::State* gt_kinematics_ = nullptr;
        Kinematics::State gt_kinematics_copy_ = {};  // Copy to avoid race condition

        // VIO kinematics (from external odometry via RPC)
        Kinematics::State vio_kinematics_ = {};

        // Environment (always from GT)
        const Environment* environment_ = nullptr;

        // VIO mode flag
        bool use_vio_ = false;  // DEFAULT: Use ground truth

        // VIO data validity tracking
        bool has_vio_data_ = false;
        std::chrono::steady_clock::time_point vio_last_update_time_ = std::chrono::steady_clock::now();

        // Thread safety
        mutable std::mutex state_mutex_;
    };
}
} //namespace
#endif
