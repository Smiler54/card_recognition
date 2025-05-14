import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import pygetwindow
import pyautogui
from matcher import Matcher
import os

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

# Variables for Detect
SEPERATE_MIN = 0.05
SEPERATE_LIMIT = 250
RECT_MIN = 20
SCORE_MIN = 0.8
DELAY = 100

AREA_LEFT = 0.55
AREA_RIGHT = 0.9
AREA_TOP = 0.75
AREA_BOTTOM = 0.88

def seperate_images(image):
    # Seperate by table
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Sobel(blurred, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=3)
    column_sum = np.sum(np.abs(edges), axis=0)

    # Create a histogram image to visualize column_sum
    hist_height = 200
    hist_width = image.shape[1]
    hist_img = np.zeros((hist_height, hist_width, 3), dtype=np.uint8)

    # Normalize column_sum to fit in hist_height
    norm_col_sum = column_sum / (np.max(column_sum) + 1e-6) * (hist_height - 1)
    for x in range(hist_width):
        y = int(hist_height - norm_col_sum[x])
        cv2.line(hist_img, (x, hist_height - 1), (x, y), (255, 255, 255), 1)

    img_height, img_width = image.shape[:2]

    # Find valleys (local minima) in column sum
    valleys = []
    threshold = SEPERATE_MIN * np.max(column_sum)
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
            if end_x - start_x < SEPERATE_LIMIT:
                continue
            clip = image[0:img_height, start_x:end_x]
            clip_images.append(clip)
    
    return clip_images, hist_img
    
def normalize(image):
    height, width = image.shape[:2]
    image = image[60:height, 0:width]
    height = height - 60

    ratio = height / 2000
    width = int(width / ratio)
    height = 2000
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
    if x <= RECT_MIN and y <= RECT_MIN:
        return image
    if w < h:
        return image

    result = image.copy()
    cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 2)
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
    
    def update_image_label(label, image):
        if label is None or image is None:
            return
        rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_img)
        display_size = (IMAGE_WIDTH, IMAGE_HEIGHT)
        pil_img.thumbnail(display_size, Image.LANCZOS)
        photo = ImageTk.PhotoImage(pil_img)
        label.configure(image=photo)
        label.image = photo  # Keep a reference!

    def open_image():
        nonlocal current_img
        # Open file dialog
        file_path = filedialog.askopenfilename()
        if file_path:
            # Read the image
            current_img = cv2.imread(file_path)
            current_img = normalize(current_img)
            update_image_label(original_label, current_img)

    def process_image_with_names(img):
        matches = []
        if img is None or img.size == 0:
            print("Error: Invalid image input")
            return None, matches
        
        # Load background image
        bg = cv2.imread('bg.png')
        if bg is None:
            print("Error: Could not load background image")
            return img, matches
            
        # Resize background to match input image dimensions if needed
        if bg.shape != img.shape:
            bg = cv2.resize(bg, (img.shape[1], img.shape[0]))
        
        # Subtract background from original image
        diff = cv2.absdiff(img, bg)

        # Clip the image to the specified region
        clip = diff
        height, width = clip.shape[:2]
        x_start = int(width * 0.55)
        x_end = int(width * 0.9)
        y_start = int(height * 0.75)
        y_end = int(height * 0.88)
        
        clip = clip[y_start:y_end, x_start:x_end]
        mask = binary_mask(clip)

        # Find contours in the binary image
        rectangle = find_rectangle_contours(mask)
        if rectangle is None or len(rectangle) < 4:
            print("Error: No valid rectangle found")
            return img, matches
        
        x, y, w, h = cv2.boundingRect(rectangle)
        cv2.rectangle(mask, (x, y), (x + w, y + h), (0, 0, 255), 2)
        height, width = mask.shape[:2]
        mask = cv2.resize(mask, (width * 3, height * 3))
        
        result = img.copy()
        # Load the template image
        template_dir = './templates'
        templates = [f for f in os.listdir(template_dir) if f.lower().endswith(('.png'))]
        if len(templates) > 0:
            clip_image = img[y_start:y_end, x_start:x_end]

            for template in templates:
                t_img = cv2.imread(os.path.join(template_dir, template))

                m_height = clip_image.shape[0]
                t_height, t_width = t_img.shape[:2]

                ratio = t_height * 2 / m_height
                t_width = int(t_width / ratio)
                t_height = int(m_height // 2)
                t_img = cv2.resize(t_img, (t_width, t_height))

                matcher = Matcher(clip_image, t_img)
                position, score = matcher.match(method=cv2.TM_CCORR_NORMED)
                index = (position[0] - x) // (t_width + 10)
                if index < 0 or index >= 5:
                    continue
                if score < SCORE_MIN:
                    continue
                if position[1] < 30:
                    continue

                name_no_ext = os.path.splitext(template)[0]
                matches.append({
                    'name': name_no_ext,
                    'position': position,
                    'score': score
                })

        matches = sorted(matches, key=lambda m: m['score'], reverse=True)[:5]
        for i, item in enumerate(matches):
            colors = [
                (255, 255, 255), (0, 255, 0), (255, 255, 0),
                (255, 0, 255), (0, 255, 255), (0, 0, 255),
            ]
            pos = item['position']
            name = item['name']
            index = (pos[0] - x) // (t_width + 10)
            result = cv2.rectangle(clip_image,
                                (pos[0], pos[1]),
                                (pos[0] + t_img.shape[1], pos[1] + t_img.shape[0]),
                                colors[index], 1)
            cv2.putText(result, name,
                        (pos[0], pos[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[index], 1)
        return result, matches

    def process():
        nonlocal current_img
        if current_img is not None and current_img.size > 0:
            processed, matches = process_image_with_names(current_img)
            if processed is not None:
                update_image_label(result_label, processed)

            matched_names_var.set(", ".join([m['name'] for m in matches]))

    def capture():
        nonlocal current_img
        screenshot = capture_screen()
        tables, _ = seperate_images(screenshot)
        if len(tables) > 0:
            current_img = tables[0]
        else:
            current_img = screenshot

        update_image_label(original_label, current_img)
    
    live_capture = [False]  # Use a mutable type to allow modification in nested functions

    def update_live_capture():
        if live_capture[0]:
            screenshot = capture_screen()
            tables, _ = seperate_images(screenshot)
            if len(tables) > 0:
                image = tables[0]
                update_image_label(original_label, image)
                processed, matches = process_image_with_names(image)
                if processed is not None:
                    update_image_label(result_label, processed)
                matched_names_var.set(", ".join([m['name'] for m in matches]))
            root.after(DELAY, update_live_capture)  # Update every 100 ms

    def toggle_live_capture():
        live_capture[0] = not live_capture[0]
        if live_capture[0]:
            live_btn.config(text="Stop Capture")
            update_live_capture()
        else:
            live_btn.config(text="Start Capture")

    def seperate():
        nonlocal current_img
        if not current_img is None:
            tables, histogram = seperate_images(current_img)
            if len(tables) > 0:
                current_img = tables[0]
                update_image_label(original_label, current_img)
                update_image_label(result_label, histogram)
    
    # Create button frame at top
    button_frame = tk.Frame(root)
    button_frame.pack(side=tk.TOP, pady=10)
    
    open_btn = tk.Button(button_frame, text="Open", command=open_image)
    open_btn.pack(side=tk.LEFT, padx=5)
    
    process_btn = tk.Button(button_frame, text="Process", command=process)
    process_btn.pack(side=tk.LEFT, padx=5)

    capture_btn = tk.Button(button_frame, text="Capture", command=capture)
    capture_btn.pack(side=tk.LEFT, padx=5)

    seperate_btn = tk.Button(button_frame, text="Seperate", command=seperate)
    seperate_btn.pack(side=tk.LEFT, padx=5)

    live_btn = tk.Button(button_frame, text="Start Capture", command=toggle_live_capture)
    live_btn.pack(side=tk.LEFT, padx=5)

    # Add input box for matched template names
    matched_names_var = tk.StringVar()
    matched_names_entry = tk.Entry(
        root, 
        textvariable=matched_names_var, 
        width=40, 
        state='readonly', 
        justify='center'  # Center align the text
    )
    matched_names_entry.pack(pady=5)

    # Move image label to bottom
    image_frame.pack_forget()
    image_frame.pack(side=tk.BOTTOM, pady=10)

    # Start the GUI event loop
    root.mainloop()

if __name__ == "__main__":
    main()
