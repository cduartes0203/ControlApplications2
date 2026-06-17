import gymnasium as gym
import highway_env
import numpy as np
import matplotlib.pyplot as plt

# Create the environment with human rendering to visualize the simulation
env = gym.make("highway-v0", render_mode="human")

# Configure the environment
env.unwrapped.configure({
    "action": {
        "type": "DiscreteMetaAction",
        "target_speeds": [0.0, 5.0, 10.0]
    },
    "simulation_frequency": 15,
    "policy_frequency": 1,
    "duration": 20, # Let it run for 20 seconds
    "vehicles_count": 0, # No other vehicles for this simple test
})

obs, info = env.reset()

# Force the initial speed to 0 and target speed to 10
env.unwrapped.vehicle.speed = 0.0
env.unwrapped.vehicle.target_speed = 10.0
env.unwrapped.vehicle.speed_index = 2

times = []
speeds = []

done = truncated = False
while not (done or truncated):
    # Action 1 is IDLE for DiscreteMetaAction (keeps current target speed)
    action = 1 
    obs, reward, done, truncated, info = env.step(action)
    
    speed = env.unwrapped.vehicle.speed
    time = env.unwrapped.time
    
    times.append(time)
    speeds.append(speed)
    
    # Render the environment
    env.render()

print("Simulation Finished!")
env.close()

# Plot the results
plt.figure(figsize=(10, 5))
plt.plot(times, speeds, label='Ego Vehicle Speed', linewidth=2)
plt.axhline(y=10.0, color='r', linestyle='--', label='Target Speed (10 m/s)')
plt.title('Vehicle Speed over Time')
plt.xlabel('Time (s)')
plt.ylabel('Speed (m/s)')
plt.legend()
plt.grid(True)
plt.savefig('speed_profile.png')
print("Speed profile saved to 'speed_profile.png'")

