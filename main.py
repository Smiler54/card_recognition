import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import pygetwindow
import pyautogui

# Set default image dimensions
IMAGE_WIDTH = 800
IMAGE_HEIGHT = 600

def capture_screen():
    window = None
    titles = pygetwindow.getAllTitles()
    # for title in titles:
    #     if "Discord" in title:
    #         window = pygetwindow.
    #         # window = pygetwindow.getWindowsWithTitle(title)[0]
    #         break

    if not window is None:
        x, y = window
        width, height = window.getSize()
        screenshot = pyautogui.screenshot(region=(x, y, width, height))
    else:
        screenshot = pyautogui.screenshot()
    
    # Convert PIL Image to cv2 format
    screenshot = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
    return screenshot

def process_image(img):
    if img is None or img.size == 0:
        print("Error: Invalid image input")
        return None
    
    # Load background image
    bg = cv2.imread('bg.png')
    if bg is None:
        print("Error: Could not load background image")
        return img
        
    # Resize background to match input image dimensions if needed
    if bg.shape != img.shape:
        bg = cv2.resize(bg, (img.shape[1], img.shape[0]))
    
    # Subtract background from original image
    diff = cv2.absdiff(img, bg)
    
    clip = diff
    
    # Clip the image to the specified region
    height, width = clip.shape[:2]
    x_start = int(width / 2)
    x_end = width
    y_start = int(height * 3 / 4)
    y_end = int(height * 7 / 8)
    
    # Check if the region is valid
    if x_start >= x_end or y_start >= y_end or x_start < 0 or y_start < 0:
        print("Error: Invalid region for processing")
        return img
    
    clip = clip[y_start:y_end, x_start:x_end]
    # Convert to HSV color space for better color detection
    hsv = cv2.cvtColor(clip, cv2.COLOR_BGR2HSV)
    
    # Define color ranges (you can adjust these values)
    lower_red = np.array([0, 50, 50])
    upper_red = np.array([10, 255, 255])
    
    lower_green = np.array([40, 50, 50]) 
    upper_green = np.array([80, 255, 255])
    
    lower_blue = np.array([100, 50, 50])
    upper_blue = np.array([140, 255, 255])
    
    # Create masks for each color
    red_mask = cv2.inRange(hsv, lower_red, upper_red)
    green_mask = cv2.inRange(hsv, lower_green, upper_green)
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
    
    # Combine all color masks
    color_mask = cv2.bitwise_or(red_mask, green_mask)
    color_mask = cv2.bitwise_or(color_mask, blue_mask)
    
    # Create final binary image (white where colors were detected, black elsewhere)
    binary = np.zeros_like(color_mask)
    binary[color_mask > 0] = 255
    
    # Convert to 3-channel image to match expected output format
    result = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    
    # Apply morphological operations to reduce noise
    kernel = np.ones((3,3), np.uint8)
    # Opening operation (erosion followed by dilation)
    result = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel)
    
    # Find contours in the binary image
    contours, _ = cv2.findContours(cv2.cvtColor(result, cv2.COLOR_BGR2GRAY), 
                                  cv2.RETR_EXTERNAL, 
                                  cv2.CHAIN_APPROX_SIMPLE)
    
    # If contours found, find the largest one
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Get the minimum area rectangle
        rect = cv2.minAreaRect(largest_contour)
        box = cv2.boxPoints(rect)
        box = np.int32(box)  # Using int32 for OpenCV contour points
        
        for point in box:
            point[0] += x_start
            point[1] += y_start
        
        # Validate box points before drawing
        if len(box) > 0 and np.all(box >= 0):
            # Draw the rectangle on the result image
            result = img.copy()
            cv2.drawContours(result, [box], 0, (0,255,0), 2)    
    
    return result

def main():
    # Create the main window
    root = tk.Tk()
    root.title("Image Processor")
    
    # Initialize image variables and labels
    current_img = None
    image_frame = tk.Frame(root)
    image_frame.pack(pady=10)
    
    original_label = tk.Label(image_frame, text="Original Image")
    original_label.pack(side=tk.LEFT, padx=10)
    
    result_label = tk.Label(image_frame, text="Processed Image")
    result_label.pack(side=tk.LEFT, padx=10)
    
    def open_image():
        nonlocal current_img
        # Open file dialog
        file_path = filedialog.askopenfilename()
        if file_path:
            # Read the image
            current_img = cv2.imread(file_path)
            if current_img is not None:
                # Convert BGR to RGB
                rgb_img = cv2.cvtColor(current_img, cv2.COLOR_BGR2RGB)
                # Convert to PIL Image
                pil_img = Image.fromarray(rgb_img)
                # Resize if too large
                display_size = (IMAGE_WIDTH, IMAGE_HEIGHT)  # Smaller size for side-by-side display
                pil_img.thumbnail(display_size, Image.LANCZOS)
                # Convert to PhotoImage
                photo = ImageTk.PhotoImage(pil_img)
                # Update label
                original_label.configure(image=photo)
                original_label.image = photo  # Keep a reference!
    
    def process():
        nonlocal current_img
        if current_img is not None and current_img.size > 0:
            # Process the image
            processed = process_image(current_img)
            if processed is not None:
                # Convert BGR to RGB
                rgb_processed = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
                # Convert to PIL Image
                pil_processed = Image.fromarray(rgb_processed)
                # Resize if too large
                display_size = (IMAGE_WIDTH, IMAGE_HEIGHT)  # Smaller size for side-by-side display
                pil_processed.thumbnail(display_size, Image.LANCZOS)
                # Convert to PhotoImage
                photo = ImageTk.PhotoImage(pil_processed)
                # Update label
                result_label.configure(image=photo)
                result_label.image = photo  # Keep a reference!
        else:
            print("Please open an image first")

    def capture():
        nonlocal current_img
        screenshot = capture_screen()
        current_img = screenshot
        if current_img is not None:
            # Convert BGR to RGB
            rgb_img = cv2.cvtColor(current_img, cv2.COLOR_BGR2RGB)
            # Convert to PIL Image
            pil_img = Image.fromarray(rgb_img)
            # Resize if too large
            display_size = (IMAGE_WIDTH, IMAGE_HEIGHT)  # Smaller size for side-by-side display
            pil_img.thumbnail(display_size, Image.LANCZOS)
            # Convert to PhotoImage
            photo = ImageTk.PhotoImage(pil_img)
            # Update label
            original_label.configure(image=photo)
            original_label.image = photo  # Keep a reference!
        process()
    
    # Create buttons
    # Create button frame at top
    button_frame = tk.Frame(root)
    button_frame.pack(side=tk.TOP, pady=10)
    
    open_btn = tk.Button(button_frame, text="Open Image", command=open_image)
    open_btn.pack(side=tk.LEFT, padx=5)
    
    process_btn = tk.Button(button_frame, text="Process Image", command=process)
    process_btn.pack(side=tk.LEFT, padx=5)

    screenshot_btn = tk.Button(button_frame, text="Screenshot", command=capture)
    screenshot_btn.pack(side=tk.LEFT, padx=5)
    
    # Move image label to bottom
    image_frame.pack_forget()
    image_frame.pack(side=tk.BOTTOM, pady=10)

    # Start the GUI event loop
    root.mainloop()

if __name__ == "__main__":
    main()
