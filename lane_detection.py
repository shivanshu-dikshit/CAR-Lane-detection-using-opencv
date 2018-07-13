# -*- coding: utf-8 -*-
"""
Created on Tue Jul 10 15:22:14 2018

@author: shivanshu_dikshit
"""

import cv2
import numpy as np
import math


def grayscale(image):
    """
    converts the image into grayscale for further
    processing
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def canny(image, low_thresh, high_thresh):
    """
    detects the gradients into the images
    and show them that is the areas where
    there is change in the color
    """
    return cv2.Canny(image,low_thresh, high_thresh)

def gaussian(image, ksize):
    """
    averages the pixels in the image
    that is helps in reducing the noise
    """
    return cv2.GaussianBlur(image, (ksize, ksize), 0)

def roi(image, vertices):
    """
    applies a mask over the image so as to help
    in working with only required portion 
    of the image
    """
    mask = np.zeros_like(image)
    match_mask_color = 255
    cv2.fillPoly(mask, vertices, match_mask_color)
    image_mask = cv2.bitwise_and(image, mask)
    
    return image_mask


def draw_lines(image, lines, color = [0, 0, 255], thickness = 3):
    """
    used in order to create the lines over the image
    
    """
    line_img = np.zeros((image.shape[0], image.shape[1], 3), dtype = np.uint8)
    img = np.copy(image)
    
    if lines is None:
        return
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(line_img, (x1, y1), (x2, y2), color, thickness)
    img= cv2.addWeighted(image, 0.8, line_img, 1.0, 0.0)
    
    return img

video = cv2.VideoCapture("input1.mp4")
 
while True:
    ret, orig_frame = video.read()
    if not ret:
        video = cv2.VideoCapture("input1.mp4")
        continue
    height, width = orig_frame.shape[:2]
    roi_vertices =[(0,height), (width/2, height/2), (width, height)]

    gray_image = grayscale(orig_frame)
    blurry = gaussian(gray_image, ksize = 5)
    edges = canny(blurry, 100, 200)

    cropped_image = roi(edges, np.array([roi_vertices],np.int32),)

    lines = cv2.HoughLinesP(cropped_image, rho = 6, theta = np.pi/60,threshold = 160,lines= np.array([]) , minLineLength = 40, maxLineGap = 25)

"""
calculating the slope of the lines detected through hough lines , then determining which
of them are left and right and then averaging them in order to get single line 
for all the lines

"""
    left_x = [] 
    left_y = []
    right_x = []
    right_y = []

    for line in lines:
        for x1, y1, x2, y2 in line:
            slope = (y2 - y1) / (x2 - x1)
            if math.fabs(slope) < 0.5:
                continue
            if slope <=0:
                left_x.extend([x1, x2])
                left_y.extend([y1, y2])
            elif slope>0:
                right_x.extend([x1, x2])
                right_y.extend([y1, y2])
                    
    print(left_x)
    print(left_y)
    min_y = orig_frame.shape[0] * (3/5)
    min_y = int(min_y)
    max_y = orig_frame.shape[0]
    max_y = int(max_y)

    poly_left = np.poly1d(np.polyfit(left_y, left_x, deg = 1))
 
    left_x_start = int(poly_left(max_y))
    left_x_end = int(poly_left(min_y))
    
    poly_right = np.poly1d(np.polyfit(right_y, right_x, deg = 1))
    
    right_x_start = int(poly_right(max_y))
    right_x_end = int(poly_right(min_y))

    line_image = draw_lines(
        orig_frame,
        [[
            [left_x_start, max_y, left_x_end, min_y],
            [right_x_start, max_y, right_x_end, min_y],
        ]],
        thickness=5,
    )
    cv2.imshow('orig_image', line_image)
    key = cv2.waitKey(25)
    if key == 27:
        break
video.release()
cv2.destroyAllWindows()
    

 
 