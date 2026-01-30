import airsim
import sys

client = airsim.MultirotorClient()
client.confirmConnection()

# Default: P=0.25, I=0, D=0
# VIO delay causes oscillation with pure P -> add D for damping, reduce P

kp = float(sys.argv[1]) if len(sys.argv) > 1 else 0.25
ki = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0
kd = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0

print(f"Setting Position PID: P={kp}, I={ki}, D={kd}")

client.setPositionControllerGains(
    airsim.PositionControllerGains(
        x_gains=airsim.PIDGains(kp, ki, kd),
        y_gains=airsim.PIDGains(kp, ki, kd),
        z_gains=airsim.PIDGains(kp, ki, kd)
    )
)

print("Done! Gains applied.")
