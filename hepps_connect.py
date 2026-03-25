import asyncio
import multiprocessing as mp
from bleak import BleakClient, BleakScanner
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Button
from collections import deque
import time
from datetime import datetime
import os
import json
import pandas as pd

DEVICE_NAME = "HePPS"  # Your ESP32 BLE device name
CHAR_UUID = "0000cf7a-0000-1000-8000-00805f9b34fb"  # Replace with your characteristic UUID

# File to store the last used name and blood pressure values
LAST_NAME_FILE = "last_name.json"
METADATA_FILE = os.path.join("data", "HEPPSBP", "metadata.csv")

def get_file_name():
    """Get participant name from user input or use the last used name as default"""
    default_name = "data"
    
    if os.path.exists(LAST_NAME_FILE):
        try:
            with open(LAST_NAME_FILE, 'r') as f:
                data = json.load(f)
                default_name = data.get('last_name', 'data')
        except:
            pass
    
    name = input(f"Enter participant name (default: {default_name}): ").strip()
    if not name:
        name = default_name
    
    try:
        with open(LAST_NAME_FILE, 'w') as f:
            json.dump({'last_name': name}, f)
    except:
        pass
    
    return name

def get_user_info(name):
    """Check if user exists in metadata (case-insensitive) and return demographics, else prompt user."""
    demographics = {}
    name_lower = name.lower()
    
    if os.path.exists(METADATA_FILE):
        try:
            df = pd.read_csv(METADATA_FILE)
            if 'Name' in df.columns:
                # Find matching names case-insensitively
                matches = df[df['Name'].astype(str).str.lower() == name_lower]
                if not matches.empty:
                    latest = matches.iloc[-1]
                    demographics['Age'] = latest.get('Age', '')
                    demographics['Sex'] = latest.get('Sex', '')
                    demographics['Height'] = latest.get('Height', '')
                    demographics['Weight'] = latest.get('Weight', '')
                    demographics['BMI'] = latest.get('BMI', '')
                    print(f"\nUser '{name}' found in records. Demographics loaded.")
                    return demographics
        except Exception as e:
            print(f"Error reading metadata: {e}")

    # If user doesn't exist or metadata file is missing
    print(f"\nUser '{name}' not found in metadata. Please enter demographic information:")
    
    demographics['Age'] = input("Enter Age: ").strip()
    demographics['Sex'] = input("Enter Sex (M/F): ").strip().upper()
    
    try:
        height = float(input("Enter Height (cm): ").strip())
        demographics['Height'] = height
    except ValueError:
        print("Invalid height. Setting to 0.")
        demographics['Height'] = 0.0
        
    try:
        weight = float(input("Enter Weight (kg): ").strip())
        demographics['Weight'] = weight
    except ValueError:
        print("Invalid weight. Setting to 0.")
        demographics['Weight'] = 0.0

    # Calculate BMI
    if demographics['Height'] > 0 and demographics['Weight'] > 0:
        height_m = demographics['Height'] / 100.0
        bmi = demographics['Weight'] / (height_m ** 2)
        demographics['BMI'] = round(bmi, 1)
    else:
        demographics['BMI'] = 0.0
        
    print(f"Calculated BMI: {demographics['BMI']}")
    return demographics

def get_blood_pressure_input():
    """Get blood pressure values from user input"""
    default_sbp = "120"
    default_dbp = "80"
    
    if os.path.exists(LAST_NAME_FILE):
        try:
            with open(LAST_NAME_FILE, 'r') as f:
                data = json.load(f)
                default_sbp = str(data.get('last_sbp', 120))
                default_dbp = str(data.get('last_dbp', 80))
        except:
            pass
    
    print("\n--- Final Data Input ---")
    sbp_input = input(f"Enter SBP (systolic blood pressure) (default: {default_sbp}): ").strip()
    if not sbp_input:
        sbp_input = default_sbp
    
    dbp_input = input(f"Enter DBP (diastolic blood pressure) (default: {default_dbp}): ").strip()
    if not dbp_input:
        dbp_input = default_dbp
    
    try:
        sbp = int(sbp_input)
        dbp = int(dbp_input)
        
        try:
            existing_data = {}
            if os.path.exists(LAST_NAME_FILE):
                with open(LAST_NAME_FILE, 'r') as f:
                    existing_data = json.load(f)
            
            existing_data['last_sbp'] = sbp
            existing_data['last_dbp'] = dbp
            
            with open(LAST_NAME_FILE, 'w') as f:
                json.dump(existing_data, f)
        except:
            pass
        
        return sbp, dbp
    except ValueError:
        print("Invalid input. Using default values.")
        return int(default_sbp), int(default_dbp)

def update_metadata_csv(name, session_timestamp, demographics, sbp, dbp):
    """Append one metadata row per recording session (Name + time)."""
    try:
        os.makedirs(os.path.dirname(METADATA_FILE), exist_ok=True)
        
        columns = ['ID', 'Name', 'time', 'Age', 'Sex', 'Height', 'Weight', 'BMI', 'SBP', 'DBP']
        
        if os.path.exists(METADATA_FILE):
            df = pd.read_csv(METADATA_FILE)
            # Ensure all required columns exist
            for col in columns:
                if col not in df.columns:
                    df[col] = ''
        else:
            df = pd.DataFrame(columns=columns)

        # Calculate new ID
        new_id = df['ID'].max() + 1 if not df.empty and df['ID'].notna().any() else 1
        
        new_row = {
            'ID': new_id,
            'Name': name,
            'time': session_timestamp,
            'Age': demographics.get('Age', ''),
            'Sex': demographics.get('Sex', ''),
            'Height': demographics.get('Height', ''),
            'Weight': demographics.get('Weight', ''),
            'BMI': demographics.get('BMI', ''),
            'SBP': sbp,
            'DBP': dbp
        }
        
        new_row_df = pd.DataFrame([new_row])
        df = pd.concat([df, new_row_df], ignore_index=True)
        
        # Ensure column order matches specifications
        df = df[columns]
        
        df.to_csv(METADATA_FILE, index=False)
        print(f"\nMetadata successfully updated in: {METADATA_FILE}")
        
    except Exception as e:
        print(f"Error updating metadata CSV: {e}")

def file_writer_process(file_queue, stop_event, participant_dir, session_timestamp, start_recording_event, data_saved_event):
    """Process for writing data to file"""
    received_batches = []
    start_time = None
    
    while not stop_event.is_set() or not file_queue.empty():
        try:
            batch_data = file_queue.get(timeout=1)
            timestamp, ain0_data, ain1_data, should_record = batch_data
            
            if should_record and start_recording_event.is_set():
                if start_time is None:
                    start_time = timestamp
                received_batches.append((timestamp, ain0_data, ain1_data))
        except:
            continue
    
    if received_batches:
        if not os.path.exists(participant_dir):
            os.makedirs(participant_dir, exist_ok=True)
        data_file_path = os.path.join(participant_dir, f"{session_timestamp}.csv")
        
        print("\nWriting data to file...")
        with open(data_file_path, "w") as f:
            f.write("timestamp,ain0,ain1\n")
            
            for i, (timestamp, ain0_data, ain1_data) in enumerate(received_batches):
                if i < len(received_batches) - 1:
                    next_timestamp = received_batches[i + 1][0]
                    time_interval = (next_timestamp - timestamp) / 30
                else:
                    if i > 0:
                        prev_timestamp = received_batches[i - 1][0]
                        time_interval = (timestamp - prev_timestamp) / 30
                    else:
                        time_interval = 0.001
                
                for j in range(30):
                    relative_timestamp = (timestamp - start_time) + (j * time_interval)
                    f.write(f"{relative_timestamp:.6f},{ain0_data[j]},{ain1_data[j]}\n")
        
        print(f"Data saved to {data_file_path}")
        data_saved_event.set()
    else:
        print("\nNo data received - no file created")

def ble_process(plot_queue, file_queue, stop_event, start_recording_event, manual_stop_event):
    """Process for BLE data collection"""
    
    async def notification_handler(sender, data):
        line = data.decode("utf-8").strip()
        current_time = time.time()
        samples = [int(x) for x in line.split(",") if x]
        
        if len(samples) == 60:
            ain0_data = samples[0:30]
            ain1_data = samples[30:60]
            should_record = start_recording_event.is_set()
            
            try:
                file_queue.put_nowait((current_time, ain0_data, ain1_data, should_record))
            except:
                pass
            
            try:
                plot_queue.put_nowait((ain0_data, ain1_data))
            except:
                pass
    
    async def ble_task():
        print("\nScanning for BLE devices...")
        try:
            devices = await BleakScanner.discover()
            address = None
            for d in devices:
                if d.name == DEVICE_NAME:
                    address = d.address
                    break
            
            if not address:
                print(f"Device '{DEVICE_NAME}' not found!")
                stop_event.set()
                return

            print(f"Connecting to {DEVICE_NAME} at {address}...")
            async with BleakClient(address) as client:
                if not client.is_connected:
                    print("Failed to connect to device!")
                    stop_event.set()
                    return
                    
                print("Connected. Subscribing to notifications...")
                await client.start_notify(CHAR_UUID, notification_handler)
                print("Collecting data. Use Start/Stop buttons in the plot window to control recording...")
                
                start_time = time.time()
                while time.time() - start_time < 120:
                    if manual_stop_event.is_set():
                        print("Manual stop triggered")
                        break
                    await asyncio.sleep(0.1)
                
                await client.stop_notify(CHAR_UUID)

            print("BLE data collection completed.")
        except Exception as e:
            print(f"BLE connection error: {e}")
        finally:
            stop_event.set()
    
    asyncio.run(ble_task())

def plotting_process(plot_queue, stop_event, start_recording_event, manual_stop_event):
    """Process for real-time plotting with interactive buttons"""
    local_ain0 = deque(maxlen=1000)
    local_ain1 = deque(maxlen=1000)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    plt.subplots_adjust(bottom=0.15)
    
    ax_start = plt.axes([0.3, 0.02, 0.15, 0.05])
    ax_stop = plt.axes([0.55, 0.02, 0.15, 0.05])
    
    btn_start = Button(ax_start, 'START')
    btn_stop = Button(ax_stop, 'STOP')    
    
    def start_recording(event):
        start_recording_event.set()
        btn_start.color = '0.8'
        btn_start.hovercolor = '0.8'
        btn_start.label.set_text('STARTED')
        print("Recording STARTED")
    
    def stop_recording(event):
        manual_stop_event.set()
        print("Recording STOPPED")
    
    btn_start.on_clicked(start_recording)
    btn_stop.on_clicked(stop_recording)
    
    def animate(i):
        while not plot_queue.empty():
            try:
                ain0_data, ain1_data = plot_queue.get_nowait()
                local_ain0.extend(ain0_data)
                local_ain1.extend(ain1_data)
            except:
                break
        
        ax.cla()
        ax.plot(list(local_ain1), label="AIN1")
        ax.set_xlabel("Sample Index")
        ax.set_ylabel("ADC Value")
        
        if start_recording_event.is_set():
            ax.set_title("Real-Time Sensor Data from ESP32 BLE - RECORDING")
        else:
            ax.set_title("Real-Time Sensor Data from ESP32 BLE - NOT RECORDING")
        
        ax.legend()
        plt.tight_layout()
        
        if (stop_event.is_set() and plot_queue.empty()) or manual_stop_event.is_set():
            plt.close('all')
    
    ani = animation.FuncAnimation(fig, animate, interval=500, cache_frame_data=False)
    plt.show()

def main():
    # 1. Get participant name
    participant_name = get_file_name()
    
    # 2. Check metadata / get demographic info BEFORE starting BLE
    demographics = get_user_info(participant_name)
    
    # Create session timestamp and directories
    session_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    participant_dir = os.path.join("data", "HEPPSBP", participant_name)
    
    # Create shared queues and events
    plot_queue = mp.Queue(maxsize=1000)
    file_queue = mp.Queue(maxsize=5000)
    stop_event = mp.Event()
    start_recording_event = mp.Event()
    manual_stop_event = mp.Event()
    data_saved_event = mp.Event()
    
    # Create processes
    ble_proc = mp.Process(target=ble_process, args=(plot_queue, file_queue, stop_event, start_recording_event, manual_stop_event))
    plot_proc = mp.Process(target=plotting_process, args=(plot_queue, stop_event, start_recording_event, manual_stop_event))
    file_proc = mp.Process(target=file_writer_process, args=(file_queue, stop_event, participant_dir, session_timestamp, start_recording_event, data_saved_event))
    
    # 3. Start data collection processes
    ble_proc.start()
    plot_proc.start()
    file_proc.start()
    
    ble_proc.join()
    time.sleep(2)
    
    if plot_proc.is_alive():
        plot_proc.terminate()
        plot_proc.join()
    
    file_proc.join()
    
    # 4. Input blood pressure and update metadata AFTER collection
    print("\nData collection process finished.")
    if data_saved_event.is_set():
        sbp, dbp = get_blood_pressure_input()
        update_metadata_csv(participant_name, session_timestamp, demographics, sbp, dbp)
    else:
        print("No session file saved, skipping metadata update.")
    
    print("All processes completed.")

if __name__ == "__main__":
    mp.freeze_support()
    main()