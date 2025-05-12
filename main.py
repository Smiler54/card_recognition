import cv2
import numpy as np
import pyautogui
import time
import threading
from queue import Queue

def capture_screen(frame_queue, stop_event):
    while not stop_event.is_set():
        # Capture the screen
        screenshot = pyautogui.screenshot()
        
        # Convert the screenshot to a numpy array
        frame = np.array(screenshot)
        
        # Convert from RGB to BGR (OpenCV format)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        # Resize the frame to make it more manageable (optional)
        scale_percent = 50  # percent of original size
        width = int(frame.shape[1] * scale_percent / 100)
        height = int(frame.shape[0] * scale_percent / 100)
        frame = cv2.resize(frame, (width, height))
        
        # Add frame to queue
        frame_queue.put(frame)
        
        # Add a small delay to reduce CPU usage
        time.sleep(0.1)

def main():
    print("Screen capture started. Press 'q' to quit.")
    print("The captured window will be displayed in real-time.")
    
    # Create a queue for frames and stop event
    frame_queue = Queue(maxsize=10)
    stop_event = threading.Event()
    
    # Create and start capture thread
    capture_thread = threading.Thread(target=capture_screen, args=(frame_queue, stop_event))
    capture_thread.start()
    
    try:
        while True:
            if not frame_queue.empty():
                # Get and display the frame
                frame = frame_queue.get()
                cv2.imshow('Screen Capture', frame)
            
            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    finally:
        # Signal the capture thread to stop
        stop_event.set()
        capture_thread.join()
        
        # Release resources
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
