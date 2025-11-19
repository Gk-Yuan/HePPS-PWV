import asyncio
import multiprocessing as mp
from bleak import BleakClient, BleakScanner
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque
import time
from datetime import datetime
import os

DEVICE_NAME = "HePPS"  # Your ESP32 BLE device name
CHAR_UUID = "0000cf7a-0000-1000-8000-00805f9b34fb"  # Replace with your characteristic UUID

# Ensure the "data" directory exists
os.makedirs("data", exist_ok=True)

# add current time to the data file name, store in "data" folder
data_file = os.path.join("data", f"ble_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")

def file_writer_process(file_queue, stop_event):
    """Process for writing data to file"""
    received_lines = []
    
    while not stop_event.is_set() or not file_queue.empty():
        try:
            # Get data from queue with timeout
            line = file_queue.get(timeout=1)
            received_lines.append(line)
        except:
            continue
    
    # Write all data to file
    print("Writing data to file...")
    with open(data_file, "w") as f:
        for line in received_lines:
            f.write(line + "\n")
    print(f"Data saved to {data_file}")

def ble_process(plot_queue, file_queue, stop_event):
    """Process for BLE data collection"""
    
    async def notification_handler(sender, data):
        # Data is bytes; decode to string
        line = data.decode("utf-8").strip()
        print(line)
        
        # Send raw line to file writer process (non-blocking)
        try:
            file_queue.put_nowait(line)
        except:
            pass  # Queue full, skip this line for file writing
        
        # Parse and send to plotting process
        samples = [int(x) for x in line.split(",") if x]
        if len(samples) == 60:  # 30 samples per channel * 2 channels
            # Split into two channels
            ain0_data = samples[0:30]     # 0-29 from ain0
            ain1_data = samples[30:60]    # 30-59 from ain1
            
            try:
                plot_queue.put_nowait((ain0_data, ain1_data))
            except:
                pass  # Queue full, skip this sample for plotting
    
    async def ble_task():
        # Scan for device
        print("Scanning for BLE devices...")
        devices = await BleakScanner.discover()
        address = None
        for d in devices:
            if d.name == DEVICE_NAME:
                address = d.address
                break
        if not address:
            print("Device not found!")
            stop_event.set()
            return

        async with BleakClient(address) as client:
            print("Connected. Subscribing to notifications...")
            await client.start_notify(CHAR_UUID, notification_handler)
            print("Collecting data for 30 seconds...")
            await asyncio.sleep(30)  # Collect for 30 seconds
            await client.stop_notify(CHAR_UUID)

        print("BLE data collection completed.")
        stop_event.set()
    
    # Run the BLE task
    asyncio.run(ble_task())

def plotting_process(plot_queue, stop_event):
    """Process for real-time plotting"""
    local_ain0 = deque(maxlen=1000)
    local_ain1 = deque(maxlen=1000)
    
    def animate(i):
        # Get new data from queue
        while not plot_queue.empty():
            try:
                ain0_data, ain1_data = plot_queue.get_nowait()
                local_ain0.extend(ain0_data)  # Add all 30 samples from ain0
                local_ain1.extend(ain1_data)  # Add all 30 samples from ain1
            except:
                break
        
        plt.cla()
        plt.plot(list(local_ain0), label="AIN0")
        plt.plot(list(local_ain1), label="AIN1")
        plt.xlabel("Sample Index")
        plt.ylabel("ADC Value")
        plt.title("Real-Time Sensor Data from ESP32 BLE")
        plt.legend()
        plt.tight_layout()
        
        # Check if BLE process is done
        if stop_event.is_set() and plot_queue.empty():
            plt.close('all')
    
    # Set up the plot
    fig = plt.figure(figsize=(12, 6))
    ani = animation.FuncAnimation(fig, animate, interval=500, cache_frame_data=False)
    
    # Show plot and keep it running until BLE process is done
    plt.show()

def main():
    # Create shared queues and event for inter-process communication
    plot_queue = mp.Queue(maxsize=1000)  # Limit queue size to prevent memory issues
    file_queue = mp.Queue(maxsize=5000)  # Larger queue for file writing
    stop_event = mp.Event()
    
    # Create processes
    ble_proc = mp.Process(target=ble_process, args=(plot_queue, file_queue, stop_event))
    plot_proc = mp.Process(target=plotting_process, args=(plot_queue, stop_event))
    file_proc = mp.Process(target=file_writer_process, args=(file_queue, stop_event))
    
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
    
    print("All processes completed.")

if __name__ == "__main__":
    # Required for Windows multiprocessing
    mp.freeze_support()
    main()