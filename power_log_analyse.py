import pandas as pd
import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv("logs/power_log.csv")

# Convert timestamp to datetime
data['timestamp'] = pd.to_datetime(data['timestamp'])

# Make temperature and voltage numeric
data['temperature'] = pd.to_numeric(data['temperature'])
data['core_voltage'] = pd.to_numeric(data['core_voltage'])
data['cpu_frequency'] = pd.to_numeric(data['cpu_frequency'])

# Create a plot
fig, ax1 = plt.subplots(figsize=(12, 6))

# Plot temperature
ax1.set_xlabel('Time')
ax1.set_ylabel('Temperature (°C)', color='red')
ax1.plot(data['timestamp'], data['temperature'], color='red')
ax1.tick_params(axis='y', labelcolor='red')

# Create second y-axis
ax2 = ax1.twinx()
ax2.set_ylabel('Core Voltage (V)', color='blue')
ax2.plot(data['timestamp'], data['core_voltage'], color='blue')
ax2.tick_params(axis='y', labelcolor='blue')

# Create third y-axis for frequency
ax3 = ax1.twinx()
ax3.spines["right"].set_position(("axes", 1.1))  # Offset the right spine
ax3.set_ylabel('CPU Frequency (MHz)', color='green')
ax3.plot(data['timestamp'], data['cpu_frequency'], color='green')
ax3.tick_params(axis='y', labelcolor='green')

plt.title('Raspberry Pi Performance Metrics During AI Processing')
fig.tight_layout()
plt.savefig('performance_metrics.png')
plt.show()

# Calculate statistics
print("=== Performance Statistics ===")
print(f"Average temperature: {data['temperature'].mean():.2f}°C")
print(f"Max temperature: {data['temperature'].max():.2f}°C")
print(f"Average core voltage: {data['core_voltage'].mean():.4f}V")
print(f"Average CPU frequency: {data['cpu_frequency'].mean():.2f}MHz")
