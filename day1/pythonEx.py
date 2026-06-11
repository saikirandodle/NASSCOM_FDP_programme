DEVICE_TYPES = ['Temperature_Sensor', 'Humidity_Sensor', 'Pressure_Sensor', 'Light_Sensor']
LOCATIONS = ['Lab1', 'Lab2', 'OfficeA', 'OfficeB', 'Warehouse']
READING_RANGES = {
    'Temperature_Sensor': (18.0, 30.0), # Celsius
    'Humidity_Sensor': (30.0, 80.0),   # Percentage
    'Pressure_Sensor': (980.0, 1030.0),# hPa
    'Light_Sensor': (100, 1000)        # Lux
}

# --- Task 1: Python Fundamentals, Lists, Tuples, Sets, Dictionaries, Indexing & Slicing ---
print("\n--- Task 1: Data Generation and Basic Structures ---")

# 1.1 Generate a list of 100 simulated sensor readings.
#    Each reading should be a tuple: (timestamp_str, device_id, device_type, location, value).
#    - timestamp_str: Format as 'YYYY-MM-DD HH:MM:SS'
#    - device_id: A unique ID for each reading (e.g., 'DEV001', 'DEV002', etc., up to 100 unique IDs).
#    - device_type: Randomly chosen from DEVICE_TYPES.
#    - location: Randomly chosen from LOCATIONS.
#    - value: Random float within READING_RANGES for its device_type (round to 2 decimal places).
# Store these in a list called 'raw_sensor_data'.

import os
import random
from datetime import datetime, timedelta
def generate_random_timestamp(start, end):
    """Generate a random timestamp between start and end."""
    delta = end - start
    random_seconds = random.randint(0, int(delta.total_seconds()))
    return start + timedelta(seconds=random_seconds)
start_time = datetime(2024, 1, 1)
end_time = datetime(2024, 1, 31)
raw_sensor_data = []
for i in range(1, 101): 
    device_id = f'DEV{i:03d}'
    device_type = random.choice(DEVICE_TYPES)
    location = random.choice(LOCATIONS)
    value_range = READING_RANGES[device_type]
    value = round(random.uniform(*value_range), 2)
    timestamp_str = generate_random_timestamp(start_time, end_time).strftime('%Y-%m-%d %H:%M:%S')
    raw_sensor_data.append((timestamp_str, device_id, device_type, location, value))
print(f"Generated {len(raw_sensor_data)} sensor readings.")



# 1.2 From 'raw_sensor_data', create a list named 'temperature_readings' containing only the readings
#     where 'device_type' is 'Temperature_Sensor'.
#     Each item in 'temperature_readings' should be a dictionary with keys:
#     'timestamp', 'device_id', 'location', 'temperature_celsius'.

temperature_readings = []
for reading in raw_sensor_data:
    timestamp_str, device_id, device_type, location, value = reading
    if device_type == 'Temperature_Sensor':
        temperature_readings.append({
            'timestamp': timestamp_str,
            'device_id': device_id,
            'location': location,
            'temperature_celsius': value
        })
print(f"Extracted {len(temperature_readings)} temperature readings.", "Sample:", temperature_readings[:5])


# 1.3 Create a set named 'unique_locations' containing all unique locations from 'raw_sensor_data'.
unique_locations = set(reading[3] for reading in raw_sensor_data)
print(f"Unique locations: {len(unique_locations)}", "Locations:", unique_locations)

# 1.4 Create a dictionary named 'device_stats' where keys are 'device_type' and values are
#     a tuple: (min_value, max_value) observed for that device type across all readings.
#     Initialize with sensible large/small values.
device_stats = {device_type: (float('inf'), float('-inf')) for device_type in DEVICE_TYPES}
for reading in raw_sensor_data:
    _, _, device_type, _, value = reading
    min_val, max_val = device_stats[device_type]
    device_stats[device_type] = (min(min_val, value), max(max_val, value))
print("Device stats (min, max):")
for device_type, stats in device_stats.items():
    print(f"{device_type}: {stats}")


# 1.5: Advanced Indexing and Slicing (Lists and Strings)
print("\n--- Task 1.5: Advanced Indexing and Slicing (Lists and Strings) ---")

# Let's use 'raw_sensor_data' and create a sample string for these tasks.
sample_string = "PythonProgrammingIsFun"

# 1.5.1 List Slicing: Extract elements from the 10th to the 20th (inclusive) from 'raw_sensor_data'.
#       Store the result in 'sliced_readings_1'.
#       Then, extract every 5th reading from 'raw_sensor_data' starting from the beginning.
#       Store the result in 'sliced_readings_2'.
sliced_readings_1 = raw_sensor_data[9:20]  # 10th to 20th (inclusive)
sliced_readings_2 = raw_sensor_data[::5]    # Every 5th reading
print(f"Sliced readings (10th to 20th): {len(sliced_readings_1)}", "Sample:", sliced_readings_1[:3])
print(f"Sliced readings (every 5th): {len(sliced_readings_2)}", "Sample:", sliced_readings_2[:3])

# 1.5.2 String Slicing: Extract the substring "Programming" from 'sample_string'.
#       Store the result in 'substring_1'.
#       Then, extract the substring "nohtyP" (the reverse of "Python") from 'sample_string' using slicing.
#       Store the result in 'substring_2'.

substring_1 = sample_string[6:17]  # "Programming"
substring_2 = sample_string[5::-1]  # "nohtyP"  
print(f"Extracted substring 1: {substring_1}")
print(f"Extracted substring 2: {substring_2}")

# 1.5.3 Mixed Slicing: Access the 'device_type' of the 5th reading in 'raw_sensor_data' using indexing.
#       Store it in 'fifth_reading_device_type'.
#       Then, from 'fifth_reading_device_type', extract the last 6 characters (e.g., 'Sensor' from 'Temperature_Sensor').
#       Store it in 'device_type_suffix'.

fifth_reading_device_type = raw_sensor_data[4][2]  # 5th reading's device_type
device_type_suffix = fifth_reading_device_type[-6:]  # Last 6 characters
print(f"5th reading device type: {fifth_reading_device_type}")
print(f"Device type suffix: {device_type_suffix}")

# --- Task 2: Writing and Using Functions in Python ---
print("\n--- Task 2: Function Definitions and Usage ---")

# 2.1 Write a function `get_readings_by_location(readings_list, target_location)`
#     that takes the 'raw_sensor_data' list and a 'target_location' string.
#     It should return a list of all readings (as tuples) for that specific location.

def get_readings_by_location(readings_list, target_location):
    """Return a list of readings for the specified location."""
    return [reading for reading in readings_list if reading[3] == target_location]
# Example usage:
location_readings = get_readings_by_location(raw_sensor_data, 'Lab1')
print(f"Readings for Lab1: {len(location_readings)}", "Sample:", location_readings[:3])

# 2.2 Write a function `calculate_average_temperature(temp_data_list)`
#     that takes the 'temperature_readings' list (list of dictionaries).
#     It should calculate and return the average temperature. Return 0.0 if the list is empty.

def calculate_average_temperature(temp_data_list):
    """Calculate and return the average temperature from the list of temperature readings."""
    if not temp_data_list:
        return 0.0
    total_temp = sum(reading['temperature_celsius'] for reading in temp_data_list)
    return total_temp / len(temp_data_list)

average_temp = calculate_average_temperature(temperature_readings)
print(f"Average temperature: {average_temp:.2f} °C")

# 2.3 Write a function `generate_report_line(reading_dict)`
#     that takes a single temperature reading dictionary.
#     It should return a formatted string like:
#     "[{timestamp}] Device {device_id} at {location}: {temperature_celsius}°C"

def generate_report_line(reading_dict):
    """Generate a formatted report line for a temperature reading."""
    return f"[{reading_dict['timestamp']}] Device {reading_dict['device_id']} at {reading_dict['location']}: {reading_dict['temperature_celsius']}°C"
# Example usage:
if temperature_readings:
    report_line = generate_report_line(temperature_readings[0])
    print("Sample report line:", report_line)

    # --- Task 3: Working with Core Python Libraries (os, datetime, random) ---
print("\n--- Task 3: Core Library Usage ---")

# 3.1 Use the `os` module to create a new directory named 'sensor_reports'
#     in the current working directory, if it doesn't already exist.

# Your code for Task 3.1 here:
reports_dir = 'sensor_reports'
if not os.path.exists(reports_dir):
    os.makedirs(reports_dir)
    print(f"\nDirectory '{reports_dir}' created.")
else:
    print(f"\nDirectory '{reports_dir}' already exists.")
    
# 3.2 Calculate the duration between the first and last timestamp in 'raw_sensor_data'.
#     Convert string timestamps to datetime objects first.
#     Print the duration in days, hours, and minutes.

# Your code for Task 3.2 here:
if len(raw_sensor_data) > 1:
    first_timestamp_str = raw_sensor_data[0][0]
    last_timestamp_str = raw_sensor_data[-1][0]

    # Convert string to datetime object
    # The format string must exactly match the timestamp string
    date_format = '%Y-%m-%d %H:%M:%S'
    first_dt = datetime.strptime(first_timestamp_str, date_format)
    last_dt = datetime.strptime(last_timestamp_str, date_format)

    duration = last_dt - first_dt

    days = duration.days
    seconds = duration.total_seconds()
    hours = int(seconds // 3600) % 24 # Calculate remaining hours after full days
    minutes = int((seconds % 3600) // 60) # Calculate remaining minutes after full hours

    print(f"\nDuration of data collection: {days} days, {hours} hours, {minutes} minutes.")
else:
    print("\nNot enough data to calculate duration.")


# --- Task 4: File Handling in Python ---
print("\n--- Task 4: File I/O ---")

# 4.1 Write all 'temperature_readings' to a file named 'sensor_reports/temperature_report.txt'.
#     Each line in the file should be generated using the `generate_report_line` function.

# Your code for Task 4.1 here:
report_filename = os.path.join(reports_dir, 'temperature_report.txt')
try:
    with open(report_filename, 'w') as f:
        for reading_dict in temperature_readings:
            line = generate_report_line(reading_dict)
            f.write(line + '\n')
    print(f"\nTemperature report written to '{report_filename}'.")
except IOError as e:
    print(f"\nError writing file '{report_filename}': {e}")


# 4.2 Read the content of 'sensor_reports/temperature_report.txt' back into a list of strings
#     named 'read_report_lines'. Print the first 3 lines.
# Your code for Task 4.2 here:

read_report_lines = []
try:
    with open(report_filename, 'r') as f:
        read_report_lines = f.readlines()
    print(f"\nRead {len(read_report_lines)} lines from '{report_filename}'.")
    print("First 3 lines:")
    for line in read_report_lines[:3]:
        print(line.strip()) # Using strip() to remove the newline character
except IOError as e:
    print(f"\nError reading file '{report_filename}': {e}")
    
    print("\n--- Exercise Complete! ---")
print("Great job completing the intermediate Python exercise.")
print("Review your code and outputs to ensure correctness and understanding of all concepts.")
print("Consider adding more complex scenarios or user inputs for further practice.")