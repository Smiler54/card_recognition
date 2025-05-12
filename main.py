import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import pygetwindow
import time
import os
import pyautogui

def capture_screen():
    # get geometry
    x, y = pyautogui.size()
    # get screenshot
    screenshot = pyautogui.screenshot()
    

def process_image(img):
    # Convert to HSV color space for better color detection
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
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
    
    return result

def main():
    # Create the main window
    root = tk.Tk()
    root.title("Image Processor")
    
    # Initialize image variable and label
    current_img = None
    image_label = tk.Label(root)
    image_label.pack(pady=10)
    
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
                display_size = (800, 600)
                pil_img.thumbnail(display_size, Image.LANCZOS)
                # Convert to PhotoImage
                photo = ImageTk.PhotoImage(pil_img)
                # Update label
                image_label.configure(image=photo)
                image_label.image = photo  # Keep a reference!
    
    def process():
        nonlocal current_img
        if current_img is not None:
            # Process the image
            processed = process_image(current_img)
            # Convert BGR to RGB
            rgb_processed = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
            # Convert to PIL Image
            pil_processed = Image.fromarray(rgb_processed)
            # Resize if too large
            display_size = (800, 600)
            pil_processed.thumbnail(display_size, Image.LANCZOS)
            # Convert to PhotoImage
            photo = ImageTk.PhotoImage(pil_processed)
            # Update label
            image_label.configure(image=photo)
            image_label.image = photo  # Keep a reference!
        else:
            print("Please open an image first")
    
    # Create buttons
    # Create button frame at top
    button_frame = tk.Frame(root)
    button_frame.pack(side=tk.TOP, pady=10)
    
    open_btn = tk.Button(button_frame, text="Open Image", command=open_image)
    open_btn.pack(side=tk.LEFT, padx=5)
    
    process_btn = tk.Button(button_frame, text="Process Image", command=process)
    process_btn.pack(side=tk.LEFT, padx=5)

    # Move image label to bottom
    image_label.pack_forget()
    image_label.pack(side=tk.BOTTOM, pady=10)
    # Start the GUI event loop
    root.mainloop()

if __name__ == "__main__":
    main()
