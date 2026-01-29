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
        virtual ~AirSimSimpleFlightEstimator() {}

        void setGroundTruthKinematics(const Kinematics::State* kinematics, const Environment* environment)
        {
            std::lock_guard<std::mutex> lock(state_mutex_);
            gt_kinematics_ = kinematics;
            environment_ = environment;
        }

        void setVIOKinematics(const Kinematics::State& vio_state)
        {
            std::lock_guard<std::mutex> lock(state_mutex_);
            vio_kinematics_ = vio_state;
            vio_last_update_time_ = std::chrono::steady_clock::now();
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
        bool hasValidVIODataUnsafe(double timeout_sec = 2.0) const
        {
            if (!has_vio_data_) {
                return false;
            }
            auto now = std::chrono::steady_clock::now();
            std::chrono::duration<double> elapsed = now - vio_last_update_time_;
            return elapsed.count() < timeout_sec;
        }

        const Kinematics::State* getKinematicsSource() const
        {
            std::lock_guard<std::mutex> lock(state_mutex_);

            // If VIO mode enabled AND VIO data is valid, use VIO
            if (use_vio_ && hasValidVIODataUnsafe()) {
                return &vio_kinematics_;
            }
            return gt_kinematics_;
        }

        const Kinematics::State* gt_kinematics_ = nullptr;
        Kinematics::State vio_kinematics_ = {};
        const Environment* environment_ = nullptr;

        bool use_vio_ = false;
        bool has_vio_data_ = false;
        std::chrono::steady_clock::time_point vio_last_update_time_ = std::chrono::steady_clock::now();

        mutable std::mutex state_mutex_;
    };
}
} //namespace
#endif
