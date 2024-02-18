# Airdrop location detector
import cv2
import numpy as np
import matplotlib.pyplot as plt

import pdb 


def find_drop_locations(image_path):
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Gaussian blur to reduce noise (more helpful with real data hopefully)
    blurred_image = cv2.GaussianBlur(gray_image, (3, 3), 0)

    # Thresholding img creates binary image & extract contours
    binary_image = cv2.adaptiveThreshold(blurred_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #pdb.set_trace() 
    
    min_area = 200   # 100
    max_area = 2500  # 50000
    filtered_contours = [cnt for cnt in contours if min_area < cv2.contourArea(cnt) < max_area]
    filtered_contours = filtered_contours[:5]
    drop_locations = []
    for contour in filtered_contours:
        M = cv2.moments(contour)
        if M["m00"] != 0:
            centroid_x = int(M["m10"] / M["m00"])
            centroid_y = int(M["m01"] / M["m00"])
            drop_locations.append((centroid_x, centroid_y))

    return drop_locations




def highlight_drop_locations(image_path, drop_locations):
    image = cv2.imread(image_path)

    for location in drop_locations:
        square_half_len = 35
        st = (location[0] - square_half_len, location[1] - square_half_len)
        ed = (location[0] + square_half_len, location[1] + square_half_len)
        cv2.rectangle(image, st, ed, (220, 220, 220), 2)

    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("Drop Locations")
    plt.show()

if __name__ == "__main__":
    #file_name = "exmple-airdrop-config-5-smaller.png"
    file_name = ".data/full_drop_zone/images/config_0.png"

    image_path = file_name

    # Find drop locations
    drop_locations = find_drop_locations(image_path)
    print("Drop Locations:")
    for i, location in enumerate(drop_locations):
        print(f"Drop {i+1}: ({location[0]}, {location[1]})")

    # Overlay/highlight drop locations 
    highlight_drop_locations(image_path, drop_locations)



