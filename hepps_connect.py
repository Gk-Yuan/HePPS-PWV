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
METADATA_FILE = os.path.join("data", "HEPPSBP", "metadata_super.csv")

def get_file_name():
    """Get participant name from user input or use the last used name as default"""
    default_name = "data"
    
    # Try to load the last used name
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
    
    # Save the name for next time
    try:
        with open(LAST_NAME_FILE, 'w') as f:
            json.dump({'last_name': name}, f)
    except:
        pass
    
    return name

def get_blood_pressure_input():
    """Get blood pressure values from user input"""
    default_sbp = "120"
    default_dbp = "80"
    
    # Try to load the last used blood pressure values
    if os.path.exists(LAST_NAME_FILE):
        try:
            with open(LAST_NAME_FILE, 'r') as f:
                data = json.load(f)
                default_sbp = str(data.get('last_sbp', 120))
                default_dbp = str(data.get('last_dbp', 80))
        except:
            pass
    
    # Get SBP input
    sbp_input = input(f"Enter SBP (systolic blood pressure) (default: {default_sbp}): ").strip()
    if not sbp_input:
        sbp_input = default_sbp
    
    # Get DBP input
    dbp_input = input(f"Enter DBP (diastolic blood pressure) (default: {default_dbp}): ").strip()
    if not dbp_input:
        dbp_input = default_dbp
    
    try:
        sbp = int(sbp_input)
        dbp = int(dbp_input)
        
        # Save the values for next time
        try:
            # Load existing data
            existing_data = {}
            if os.path.exists(LAST_NAME_FILE):
                with open(LAST_NAME_FILE, 'r') as f:
                    existing_data = json.load(f)
            
            # Update with new blood pressure values
            existing_data['last_sbp'] = sbp
            existing_data['last_dbp'] = dbp
            
            # Save back to file
            with open(LAST_NAME_FILE, 'w') as f:
                json.dump(existing_data, f)
        except:
            pass
        
        return sbp, dbp
    except ValueError:
        print("Invalid input. Using default values.")
        return int(default_sbp), int(default_dbp)

def update_metadata_csv(name, session_timestamp, sbp, dbp):
    """Append one metadata row per recording session (Name + time)."""
    try:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(METADATA_FILE), exist_ok=True)
        
        # Try to read existing CSV file
        if os.path.exists(METADATA_FILE):
            df = pd.read_csv(METADATA_FILE)
        else:
            # Create new DataFrame with the expected columns
            columns = ['ID', 'Name', 'time', 'Age', 'Sex', 'Height', 'Weight', 'Arm Span', 'Leg Length',
                      'SBP', 'DBP', 'bpHR', 'vPTT', 'vPWV', 'vHR', 'PTT', 'HR', 
                      'SBP1', 'DBP1', 'HR1', 'SBP2', 'DBP2', 'HR2', 'SBP3', 'DBP3', 'HR3']
            df = pd.DataFrame(columns=columns)

        # Backfill the session key column if metadata existed without it
        if 'time' not in df.columns:
            df.insert(2, 'time', '')

        # Add new entry (append-only; no overwrite)
        new_id = df['ID'].max() + 1 if not df.empty and df['ID'].notna().any() else 1
        new_row = {
            'ID': new_id,
            'Name': name,
            'time': session_timestamp,
            'SBP1': sbp,
            'SBP2': sbp,
            'SBP3': sbp,
            'DBP1': dbp,
            'DBP2': dbp,
            'DBP3': dbp
        }
        # Convert new_row to DataFrame and concatenate
        new_row_df = pd.DataFrame([new_row])
        df = pd.concat([df, new_row_df], ignore_index=True)
        print(f"Appended metadata row for: {name}, time={session_timestamp}")
        
        # Save the updated DataFrame
        df.to_csv(METADATA_FILE, index=False)
        print(f"Metadata updated in: {METADATA_FILE}")
        
    except Exception as e:
        print(f"Error updating metadata CSV: {e}")

def file_writer_process(file_queue, stop_event, participant_dir, session_timestamp, start_recording_event, data_saved_event):
    """Process for writing data to file"""
    received_batches = []
    start_time = None
    
    while not stop_event.is_set() or not file_queue.empty():
        try:
            # Get data from queue with timeout
            batch_data = file_queue.get(timeout=1)
            timestamp, ain0_data, ain1_data, should_record = batch_data
            
            # Only record if start button has been clicked
            if should_record and start_recording_event.is_set():
                if start_time is None:
                    start_time = timestamp  # Set the reference time when recording starts
                received_batches.append((timestamp, ain0_data, ain1_data))
        except:
            continue
    
    # Only create directory and file if we have data
    if received_batches:
        # Create participant directory structure only when we have data to save
        # if the directory exists, do not create a new directory
        if not os.path.exists(participant_dir):
            os.makedirs(participant_dir, exist_ok=True)
        data_file_path = os.path.join(participant_dir, f"{session_timestamp}.csv")
        
        # Write all data to file in CSV format
        print("Writing data to file...")
        with open(data_file_path, "w") as f:
            # Write header
            f.write("timestamp,ain0,ain1\n")
            
            # Process each batch
            for i, (timestamp, ain0_data, ain1_data) in enumerate(received_batches):
                # Calculate timestamps for the 30 samples, starting from 0
                if i < len(received_batches) - 1:
                    next_timestamp = received_batches[i + 1][0]
                    time_interval = (next_timestamp - timestamp) / 30
                else:
                    # For the last batch, use the same interval as the previous batch
                    if i > 0:
                        prev_timestamp = received_batches[i - 1][0]
                        time_interval = (timestamp - prev_timestamp) / 30
                    else:
                        # If only one batch, use a default interval
                        time_interval = 0.001  # 1ms default
                
                # Write 30 rows for this batch with timestamp starting from 0
                for j in range(30):
                    # Calculate relative timestamp from start of recording
                    relative_timestamp = (timestamp - start_time) + (j * time_interval)
                    f.write(f"{relative_timestamp:.6f},{ain0_data[j]},{ain1_data[j]}\n")
        
        print(f"Data saved to {data_file_path}")
        data_saved_event.set()
    else:
        print("No data received - no file created")

def ble_process(plot_queue, file_queue, stop_event, start_recording_event, manual_stop_event):
    """Process for BLE data collection"""
    
    async def notification_handler(sender, data):
        # Data is bytes; decode to string
        line = data.decode("utf-8").strip()
        
        # Get current timestamp
        current_time = time.time()
        
        # Parse and send to both file and plotting processes
        samples = [int(x) for x in line.split(",") if x]
        if len(samples) == 60:  # 30 samples per channel * 2 channels
            # Split into two channels
            ain0_data = samples[0:30]     # 0-29 from ain0
            ain1_data = samples[30:60]    # 30-59 from ain1
            
            # Send structured data to file writer process (non-blocking)
            # Include recording status
            should_record = start_recording_event.is_set()
            try:
                file_queue.put_nowait((current_time, ain0_data, ain1_data, should_record))
            except:
                print("File queue full, skipping")
            
            try:
                plot_queue.put_nowait((ain0_data, ain1_data))
            except:
                print("Plot queue full, skipping")
        else:
            print(f"Warning: Expected 60 samples, got {len(samples)}")
    
    async def ble_task():
        # Scan for device
        print("Scanning for BLE devices...")
        try:
            devices = await BleakScanner.discover()
            print(f"Found {len(devices)} devices:")
            for d in devices:
                print(f"  - {d.name}: {d.address}")
            
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
                print("Collecting data. Use Start/Stop buttons to control recording...")
                
                # Run for maximum 2 minutes (120 seconds) or until manual stop
                start_time = time.time()
                while time.time() - start_time < 120:  # 2 minutes max
                    if manual_stop_event.is_set():
                        print("Manual stop triggered")
                        break
                    await asyncio.sleep(0.1)  # Check every 100ms
                
                await client.stop_notify(CHAR_UUID)

            print("BLE data collection completed.")
        except Exception as e:
            print(f"BLE connection error: {e}")
        finally:
            stop_event.set()
    
    # Run the BLE task
    asyncio.run(ble_task())

def plotting_process(plot_queue, stop_event, start_recording_event, manual_stop_event):
    """Process for real-time plotting with interactive buttons"""
    local_ain0 = deque(maxlen=1000)
    local_ain1 = deque(maxlen=1000)
    
    # Set up the plot
    fig, ax = plt.subplots(figsize=(12, 8))
    plt.subplots_adjust(bottom=0.15)  # Make room for buttons
    
    # Create button axes
    ax_start = plt.axes([0.3, 0.02, 0.15, 0.05])
    ax_stop = plt.axes([0.55, 0.02, 0.15, 0.05])
    
    # Create buttons
    btn_start = Button(ax_start, 'START')
    btn_stop = Button(ax_stop, 'STOP')    # Button callback functions
    def start_recording(event):
        start_recording_event.set()
        # Gray out the start button to show it's been activated
        btn_start.color = '0.8'  # Light gray
        btn_start.hovercolor = '0.8'  # Keep gray on hover
        btn_start.label.set_text('STARTED')  # Change button text
        print("Recording STARTED")
    
    def stop_recording(event):
        manual_stop_event.set()
        print("Recording STOPPED")
    
    # Connect button callbacks
    btn_start.on_clicked(start_recording)
    btn_stop.on_clicked(stop_recording)
    
    def animate(i):
        # Get new data from queue
        while not plot_queue.empty():
            try:
                ain0_data, ain1_data = plot_queue.get_nowait()
                local_ain0.extend(ain0_data)  # Add all 30 samples from ain0
                local_ain1.extend(ain1_data)  # Add all 30 samples from ain1
            except:
                break
        
        ax.cla()
        # ax.plot(list(local_ain0), label="AIN0")
        ax.plot(list(local_ain1), label="AIN1")
        ax.set_xlabel("Sample Index")
        ax.set_ylabel("ADC Value")
        
        # Update title based on recording status
        if start_recording_event.is_set():
            ax.set_title("Real-Time Sensor Data from ESP32 BLE - RECORDING")
        else:
            ax.set_title("Real-Time Sensor Data from ESP32 BLE - NOT RECORDING")
        
        ax.legend()
        plt.tight_layout()
        
        # Check if BLE process is done or manual stop
        if (stop_event.is_set() and plot_queue.empty()) or manual_stop_event.is_set():
            plt.close('all')
    
    # Create animation
    ani = animation.FuncAnimation(fig, animate, interval=500, cache_frame_data=False)
    
    # Show plot and keep it running until BLE process is done
    plt.show()

def main():
    # Get participant name from user
    participant_name = get_file_name()
    
    # Create session timestamp
    session_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Create participant directory structure path but don't create it yet
    participant_dir = os.path.join("data", "HEPPSBP", participant_name)
    
    # Create shared queues and event for inter-process communication
    plot_queue = mp.Queue(maxsize=1000)  # Limit queue size to prevent memory issues
    file_queue = mp.Queue(maxsize=5000)  # Larger queue for file writing
    stop_event = mp.Event()
    start_recording_event = mp.Event()  # Event to control recording start
    manual_stop_event = mp.Event()      # Event for manual stop button
    data_saved_event = mp.Event()       # Event set only when a session CSV is written
    
    # Create processes
    ble_proc = mp.Process(target=ble_process, args=(plot_queue, file_queue, stop_event, start_recording_event, manual_stop_event))
    plot_proc = mp.Process(target=plotting_process, args=(plot_queue, stop_event, start_recording_event, manual_stop_event))
    file_proc = mp.Process(target=file_writer_process, args=(file_queue, stop_event, participant_dir, session_timestamp, start_recording_event, data_saved_event))
    
    # Start processes
    ble_proc.start()
    plot_proc.start()
    file_proc.start()
    
    # Wait for BLE process to complete
    ble_proc.join()
    
    # Wait a bit for plotting process to finish displaying remaining data
    time.sleep(2)
    
    # Terminate plotting process if it's still running
    if plot_proc.is_alive():
        plot_proc.terminate()
        plot_proc.join()
    
    # Wait for file writer to finish
    file_proc.join()
    
    # After all processes complete, ask for blood pressure input
    print("\nData collection completed.")
    if data_saved_event.is_set():
        sbp, dbp = get_blood_pressure_input()
        update_metadata_csv(participant_name, session_timestamp, sbp, dbp)
    else:
        print("No session file saved, skipping metadata update.")
    
    print("All processes completed.")

if __name__ == "__main__":
    # Required for Windows multiprocessing
    mp.freeze_support()
    main()
