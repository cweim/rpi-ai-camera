# power_monitor.py
import subprocess
import time
import datetime
import os

# Configuration
log_file = "power_log.csv"
duration_hours = 1  # Change as needed
log_interval_seconds = 5  # How often to record measurements

# Create logs directory if it doesn't exist
os.makedirs("logs", exist_ok=True)
log_path = os.path.join("logs", log_file)

# Initialize log file with headers
with open(log_path, "w") as f:
    f.write("timestamp,temperature,core_voltage,cpu_frequency\n")

print(f"Starting power monitoring for {duration_hours} hours...")
print(f"Log file: {log_path}")

start_time = time.time()
end_time = start_time + (duration_hours * 3600)

samples = 0
try:
    while time.time() < end_time:
        # Get temperature
        temp_output = subprocess.check_output(["vcgencmd", "measure_temp"]).decode()
        temp = temp_output.split("=")[1].split("'")[0]

        # Get core voltage
        volt_output = subprocess.check_output(["vcgencmd", "measure_volts core"]).decode()
        volts = volt_output.split("=")[1].split("V")[0]

        # Get CPU frequency (in MHz)
        freq_output = subprocess.check_output(["vcgencmd", "measure_clock arm"]).decode()
        freq_mhz = str(int(freq_output.split("=")[1]) / 1000000)

        # Log data
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(log_path, "a") as f:
            f.write(f"{timestamp},{temp},{volts},{freq_mhz}\n")

        samples += 1
        if samples % 12 == 0:  # Show status every ~minute (if interval is 5 seconds)
            elapsed_min = (time.time() - start_time) / 60
            remaining_min = (end_time - time.time()) / 60
            print(f"Running for {elapsed_min:.1f} min, {remaining_min:.1f} min remaining. Temp: {temp}°C, Voltage: {volts}V")

        # Sleep until next measurement
        time.sleep(log_interval_seconds)

except KeyboardInterrupt:
    print("\nMonitoring stopped by user")
finally:
    print(f"Monitoring complete. Log saved to {log_path}")
    elapsed_time = time.time() - start_time
    print(f"Total runtime: {elapsed_time/60:.2f} minutes, {samples} samples collected")
