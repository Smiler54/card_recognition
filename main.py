import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import pygetwindow
import pyautogui
import glob
# from scipy.signal import find_peaks

# Set default image dimensions
IMAGE_WIDTH = 800
IMAGE_HEIGHT = 800

# Define color ranges (you can adjust these values)
lower_red = np.array([0, 50, 50])
upper_red = np.array([10, 255, 255])

lower_green = np.array([40, 50, 50]) 
upper_green = np.array([80, 255, 255])

lower_blue = np.array([100, 50, 50])
upper_blue = np.array([140, 255, 255])

lower_gray = np.array([0, 0, 50])
upper_gray = np.array([180, 20, 120])

def seperate_images(image):
    # Seperate by table
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Sobel(blurred, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=3)
    column_sum = np.sum(np.abs(edges), axis=0)

    img_height, img_width = image.shape[:2]

    # Find valleys (local minima) in column sum
    valleys = []
    threshold = 0.1 * np.max(column_sum)
    for i in range(1, len(column_sum)-1):
        if column_sum[i] < column_sum[i-1] and column_sum[i] < column_sum[i+1]:
            if column_sum[i] < threshold:
                valleys.append([i, column_sum[i]])

    # Calculate center positions for each separated area
    clip_images = []
    if len(valleys) > 0:
        seperator_x = [s[0] for s in valleys]
        seperator_x.insert(0, 0)  # Add start of image
        seperator_x.append(img_width)  # Add end of image

        for i in range(len(seperator_x) - 1):
            start_x = seperator_x[i]
            end_x = seperator_x[i + 1]
            if end_x - start_x < 100:
                continue
            clip = image[0:img_height, start_x:end_x]
            clip_images.append(clip)
    
    return clip_images
    
def normalize(image):
    height, width = image.shape[:2]
    image = image[60:height, 0:width]
    height = height - 60

    ratio = height / 800
    width = int(width / ratio)
    height = 800
    # Resize the screenshot to a fixed height
    result = cv2.resize(np.array(image), (width, height))
    return result

def capture_screen():
    window = None
    titles = pygetwindow.getAllTitles()
    for title in titles:
        if "SupremaPoker" in title:
            window = pygetwindow.getWindowsWithTitle(title)[0]
            break

    if window is not None:
        x, y = window.left, window.top
        width, height = window.width, window.height
        screenshot = pyautogui.screenshot(region=(x, y, width, height))
    else:
        screenshot = pyautogui.screenshot()
    
    # Convert PIL Image to cv2 format
    screenshot = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
    result = normalize(screenshot)

    return result

def binary_mask(image):
    # Convert to HSV color space for better color detection
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Create masks for each color
    red_mask = cv2.inRange(hsv, lower_red, upper_red)
    green_mask = cv2.inRange(hsv, lower_green, upper_green)
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
    gray_mask = cv2.inRange(hsv, lower_gray, upper_gray)

    # Apply erosion to each mask to reduce noise
    kernel = np.ones((3,3), np.uint8)
    red_mask = cv2.erode(red_mask, kernel, iterations=2)
    green_mask = cv2.erode(green_mask, kernel, iterations=2) 
    blue_mask = cv2.erode(blue_mask, kernel, iterations=2)
    gray_mask = cv2.erode(gray_mask, kernel, iterations=2)

    # Apply dilation to each mask to enhance detected regions
    red_mask = cv2.dilate(red_mask, kernel, iterations=4)
    green_mask = cv2.dilate(green_mask, kernel, iterations=4)
    blue_mask = cv2.dilate(blue_mask, kernel, iterations=4)
    gray_mask = cv2.dilate(gray_mask, kernel, iterations=4)
    
    # Combine all color masks
    color_mask = cv2.bitwise_or(red_mask, green_mask)
    color_mask = cv2.bitwise_or(color_mask, blue_mask)
    color_mask = cv2.bitwise_or(color_mask, gray_mask)

    # Combine masks into a single color image: red, green, blue, gray
    color_masks = np.zeros((*red_mask.shape, 3), dtype=np.uint8)
    color_masks[red_mask > 0] = [0, 0, 255]      # Red in BGR
    color_masks[green_mask > 0] = [0, 255, 0]    # Green in BGR
    color_masks[blue_mask > 0] = [255, 0, 0]     # Blue in BGR
    color_masks[gray_mask > 0] = [128, 128, 128] # Gray

    # cv2.imshow('All Masks Combined by Color', color_masks)
    
    # Create final binary image (white where colors were detected, black elsewhere)
    binary = np.zeros_like(color_mask)
    binary[color_mask > 0] = 255
    
    # Convert to 3-channel image to match expected output format
    result = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    
    # Apply morphological operations to reduce noise
    kernel = np.ones((3,3), np.uint8)
    # Opening operation (erosion followed by dilation)
    result = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel)
    return result

def find_rectangle_contours(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Find contours
    contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    longest_contour = max(contours, key=cv2.contourArea) if contours else None
    
    return longest_contour

def draw_rectangle(image, x, y, w, h, clip_width, clip_height):
    if x <= 10 and y <= 10:
        return image
    if w < h:
        return image
    if w < clip_width / 3 and h < clip_height / 3:
        return image

    result = image.copy()
    cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return result

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

    # Clip the image to the specified region
    clip = diff
    height, width = clip.shape[:2]
    x_start = int(width / 2)
    x_end = width
    y_start = int(height * 3 / 4)
    y_end = int(height * 7 / 8)
    
    clip = clip[y_start:y_end, x_start:x_end]
    
    mask = binary_mask(clip)

    # Find contours in the binary image
    rectangle = find_rectangle_contours(mask)
    if rectangle is None or len(rectangle) < 4:
        print("Error: No valid rectangle found")
        return img
    
    x, y, w, h = cv2.boundingRect(rectangle)
    cv2.rectangle(mask, (x, y), (x + w, y + h), (0, 0, 255), 2)
    height, width = mask.shape[:2]
    mask = cv2.resize(mask, (width * 3, height * 3))
    # cv2.imshow('Combined Mask', mask)

    clip_height, clip_width = mask.shape[:2]
    x = int(x + x_start)
    y = int(y + y_start)
    result = draw_rectangle(img, x, y, w, h, clip_width, clip_height)
    
    # template = cv2.imread('templates/d10.png')
    # if template is not None and rectangle is not None:
    #     height, width = template.shape[:2]
    #     ratio = height / h
    #     width = int(width / ratio)
    #     height = int(h)
    #     # Resize template to match detected rectangle size
    #     template_resized = cv2.resize(template, (width, height))
    #     # Perform template matching on the clipped region
    #     match_result = cv2.matchTemplate(clip, template_resized, cv2.TM_CCOEFF_NORMED)
    #     min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(match_result)
    #     threshold = 0.5
    #     if max_val >= threshold:
    #         match_x, match_y = max_loc
    #         # Convert coordinates to original image space
    #         abs_x = x_start + match_x
    #         abs_y = y_start + match_y
    #         # Draw rectangle on result image
    #         cv2.rectangle(result, (abs_x, abs_y), (abs_x + width, abs_y + height), (0, 255, 255), -1)
    #         print(f"Template matched at position: ({abs_x}, {abs_y}) with score: {max_val:.2f}")
    
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
            current_img = normalize(current_img)
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
    
    live_capture = [False]  # Use a mutable type to allow modification in nested functions

    def update_live_capture():
        if live_capture[0]:
            screenshot = capture_screen()
            tables = seperate_images(screenshot)
            if len(tables) > 0:
                image = tables[0]
                if image is not None:
                    rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(rgb_img)
                    display_size = (IMAGE_WIDTH, IMAGE_HEIGHT)
                    pil_img.thumbnail(display_size, Image.LANCZOS)
                    photo = ImageTk.PhotoImage(pil_img)
                    original_label.configure(image=photo)
                    original_label.image = photo
                    # Optionally process and show result
                    processed = process_image(image)
                    if processed is not None:
                        rgb_processed = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
                        pil_processed = Image.fromarray(rgb_processed)
                        pil_processed.thumbnail(display_size, Image.LANCZOS)
                        photo2 = ImageTk.PhotoImage(pil_processed)
                        result_label.configure(image=photo2)
                        result_label.image = photo2
            root.after(30, update_live_capture)  # Update every 100 ms

    def toggle_live_capture():
        live_capture[0] = not live_capture[0]
        if live_capture[0]:
            live_btn.config(text="Stop Capture")
            update_live_capture()
        else:
            live_btn.config(text="Start Capture")

    def seperate():
        seperate_images(current_img)
    
    # Create buttons
    # Create button frame at top
    button_frame = tk.Frame(root)
    button_frame.pack(side=tk.TOP, pady=10)
    
    # open_btn = tk.Button(button_frame, text="Open", command=open_image)
    # open_btn.pack(side=tk.LEFT, padx=5)
    
    # process_btn = tk.Button(button_frame, text="Process", command=process)
    # process_btn.pack(side=tk.LEFT, padx=5)

    live_btn = tk.Button(button_frame, text="Start Capture", command=toggle_live_capture)
    live_btn.pack(side=tk.LEFT, padx=5)

    # capture_btn = tk.Button(button_frame, text="Capture", command=capture)
    # capture_btn.pack(side=tk.LEFT, padx=5)

    # seperate_btn = tk.Button(button_frame, text="Seperate", command=seperate)
    # seperate_btn.pack(side=tk.LEFT, padx=5)

    # Move image label to bottom
    image_frame.pack_forget()
    image_frame.pack(side=tk.BOTTOM, pady=10)

    # Start the GUI event loop
    root.mainloop()

if __name__ == "__main__":
    main()
