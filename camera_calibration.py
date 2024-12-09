#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import json
from scipy.optimize import least_squares
from typing import List, Dict


# In[ ]:


def detect_edges(image, low_threshold=10, high_threshold=150):
    # Convert to grayscale if the image is not already
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # Apply binary thresholding to obtain a binary image
    _, binary = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
    
    # # Define a kernel for erosion and dilation
    # kernel = np.ones((17, 17), np.uint8)

    # # Apply erosion and then dilation (closing)
    # binary = cv2.erode(binary, kernel, iterations=1)
    # binary = cv2.dilate(binary, kernel, iterations=1)

    # Detect edges using Canny edge detector on the binary image
    edges = cv2.Canny(binary, low_threshold, high_threshold)

    # Find contours from the edges
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    large_closed_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 0 and cv2.arcLength(cnt, True) > 0]
    
    # Create a blank image to draw contours
    contour_image = np.zeros_like(image)

    # Draw contours on the blank image
    cv2.drawContours(contour_image, large_closed_contours, -1, (255, 255, 255), 1)  # Draw in white

    # Convert contour image to grayscale
    contour_image_gray = cv2.cvtColor(contour_image, cv2.COLOR_BGR2GRAY)

    return edges


# In[ ]:


def hough_lines(edges, rho=2, theta=np.pi/180, threshold=70, min_line_length=5, max_line_gap=20):
    # Using Probabilistic Hough Transform for line detection
    lines = cv2.HoughLinesP(edges, rho, theta, threshold, minLineLength=min_line_length, maxLineGap=max_line_gap)
    return lines


# In[ ]:


def draw_lines(image, lines, color):
    # Draw each Hough line on the image
    for i, line in enumerate(lines):
        x1, y1, x2, y2 = line[0]
        cv2.line(image, (x1, y1), (x2, y2), color, 2)


# In[ ]:


def line_to_homogeneous(x1, y1, x2, y2):
    # Convert a line defined by two points to its homogeneous form (a, b, c)
    a = y1 - y2
    b = x2 - x1
    c = x1 * y2 - y1 * x2
    return np.array([a, b, c])


# In[ ]:


def angle_between_lines(line1, line2):
    """Calculate the angle between two lines defined by their endpoints."""
    # Extract coordinates
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2

    # Calculate direction vectors
    dir1 = np.array([x2 - x1, y2 - y1], dtype=np.float64)
    dir2 = np.array([x4 - x3, y4 - y3], dtype=np.float64)

    # Check for zero-length direction vectors
    norm1 = np.linalg.norm(dir1)
    norm2 = np.linalg.norm(dir2)

    if norm1 == 0 or norm2 == 0:
        # If either line is degenerate, return an angle of 0 (or any consistent value)
        return 0.0

    # Normalize vectors
    dir1 /= norm1
    dir2 /= norm2

    # Calculate angle
    angle = np.arccos(np.clip(np.dot(dir1, dir2), -1.0, 1.0))
    return angle


# In[ ]:


def point_to_line_distance(px, py, x1, y1, x2, y2):
    """Calculate the distance from a point (px, py) to a line segment defined by (x1, y1) and (x2, y2)."""
    line_vec = np.array([x2 - x1, y2 - y1])
    line_len = np.linalg.norm(line_vec)
    
    if line_len == 0:  # Line segment is a point
        return np.sqrt((px - x1) ** 2 + (py - y1) ** 2)

    line_unit_vec = line_vec / line_len
    point_vec = np.array([px - x1, py - y1])
    t = np.dot(point_vec, line_unit_vec)

    if t < 0:  # Projection falls before the line segment
        return np.sqrt((px - x1) ** 2 + (py - y1) ** 2)
    elif t > line_len:  # Projection falls after the line segment
        return np.sqrt((px - x2) ** 2 + (py - y2) ** 2)

    projection = line_unit_vec * t
    closest_point = np.array([x1, y1]) + projection
    return np.sqrt((px - closest_point[0]) ** 2 + (py - closest_point[1]) ** 2)


# In[ ]:


def perpendicular_distance(point, line):
    """Calculate the perpendicular distance from a point to a line defined by two points."""
    x1, y1 = point
    x3, y3, x4, y4 = line[0]  # Unpack line's endpoints

    # Line coefficients A, B, C for the line defined by (x3, y3) to (x4, y4)
    A = y4 - y3
    B = x3 - x4
    C = (x4 * y3) - (x3 * y4)

    # Calculate the perpendicular distance
    distance = abs(A * x1 + B * y1 + C) / np.sqrt(A**2 + B**2)
    return distance


# In[ ]:


def are_lines_coincident(line1, line2, distance_threshold, angle_threshold):
    """Check if two lines can be extended to meet."""
    x1, y1, x2, y2 = line1[0]

    # Check if the angle between lines is less than the threshold or the lines are opposite (180 degrees)
    angle = angle_between_lines(line1[0], line2[0])
    if angle < angle_threshold or abs(angle - np.pi) < angle_threshold:
        # Calculate the perpendicular distance from both endpoints of line1 to line2
        distance_to_line2_start = perpendicular_distance((x1, y1), line2)
        distance_to_line2_end = perpendicular_distance((x2, y2), line2)
        # Lines are considered coincident if the distance to either endpoint is below the threshold
        return distance_to_line2_start < distance_threshold or distance_to_line2_end < distance_threshold

    return False


# In[ ]:


def group_lines(lines, img_width, img_height, angle_threshold=np.pi / 36, distance_threshold=20):
    groups = []
    for line in lines:
        found_group = False
        for group in groups:
            # Check if the group is not empty before comparing angles
            if len(group) > 0:
                if are_lines_coincident(line, group[0], distance_threshold, angle_threshold):
                    found_group = True
                    group.append(line)  # Maintain the nested structure
                    break
        
        if not found_group:
            # If no group was found, create a new group with the current line
            groups.append([line])  # Maintain a nested structure

    # Merge lines in each group by fitting a single line across all points
    merged_lines = []
    filtered_groups = []
    for group in groups:
        # Only proceed if the group has more than one line
        if len(group) > 1:
            all_points = []
            
            # Collect all points from each line in the group
            for line in group:
                x1, y1, x2, y2 = line[0]
                all_points.extend([(x1, y1), (x2, y2)])
            
            # Separate x and y coordinates
            x_coords, y_coords = zip(*all_points)
            x_coords, y_coords = np.array(x_coords), np.array(y_coords)
            
            # Perform linear regression to fit a line (y = mx + b)
            A = np.vstack([x_coords, np.ones(len(x_coords))]).T
            m, b = np.linalg.lstsq(A, y_coords, rcond=None)[0]
            
            # Calculate line endpoints based on image edges
            start_point = (0, int(b))
            end_point = (img_width, int(m * img_width + b))
            
            # Append the fitted line for this group
            merged_lines.append([[start_point[0], start_point[1], end_point[0], end_point[1]]])
            filtered_groups.append(group)

    return np.array(merged_lines), filtered_groups


# In[ ]:


def homogeneous_intersection(L1, L2):
    # Compute the intersection of two lines in homogeneous form
    intersection = np.cross(L1, L2)
    if intersection[2] == 0:
        return None  # Lines are parallel
    # Convert from homogeneous to Cartesian coordinates
    x = intersection[0] / intersection[2]
    y = intersection[1] / intersection[2]
    return int(x), int(y)


# In[ ]:


def detect_corners(lines):
    intersections = []
    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            x1, y1, x2, y2 = lines[i][0]
            x3, y3, x4, y4 = lines[j][0]
            L1 = line_to_homogeneous(x1, y1, x2, y2)
            L2 = line_to_homogeneous(x3, y3, x4, y4)
            intersect = homogeneous_intersection(L1, L2)
            if intersect:
                intersections.append(intersect)
    return intersections


# In[ ]:


def label_corners(image, corners):
    for idx, (x, y) in enumerate(corners):
        cv2.circle(image, (x, y), 2, (0, 0, 255), -1)
        cv2.putText(image, str(idx + 1), (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)


# In[ ]:


def find_intersections(merged_lines, img, img_height, img_width):
    # Filter out lines starting from (0, 0)
    filtered_lines = [line for line in merged_lines if not (line[0][0] == 0 and line[0][1] == 0)]
    
    slope_threshold = 1  # Adjust this threshold to control "mostly horizontal" classification

    # Filter out lines starting from (0, y) where y > 0 and that are more horizontal
    filtered_horizontal_lines = []
    for line in filtered_lines:
        x1, y1, x2, y2 = line[0]
        dx = x2 - x1
        dy = y2 - y1
        # Check if line starts near (0, y) with y > 0 and is mostly horizontal
        if abs(dy) < slope_threshold * abs(dx):
            filtered_horizontal_lines.append(line)
    filtered_horizontal_lines.sort(key=lambda line: line[0][1])
    # print(filtered_horizontal_lines)
    # Store intersections and their labels
    labeled_intersections = {}
    current_label = 1
    tolerance = 1e-5  # Tolerance for checking duplicate points

    # Iterate through each line starting from (0, y)
    for line1 in filtered_horizontal_lines:
        intersections = []

        # Extract coordinates for the current line
        x1, y1, x2, y2 = line1[0]

        # Iterate through all filtered lines to find intersections
        for line2 in filtered_lines:
            if np.array_equal(line1, line2):  # Use np.array_equal to compare arrays directly
                continue  # Skip the same line
            
            x3, y3, x4, y4 = line2[0]

            # Calculate the intersection point using line equations
            denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
            if denom == 0:
                continue  # Lines are parallel, no intersection

            # Calculate intersection point (x, y)
            intersect_x = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom
            intersect_y = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom

            # Check if the intersection is within image bounds
            if 0 <= intersect_x <= img.shape[1] and 0 <= intersect_y <= img_height:
                # Check if this intersection is already labeled
                is_duplicate = any(
                    np.linalg.norm(np.array((intersect_x, intersect_y)) - np.array(pt)) < tolerance
                    for pt in labeled_intersections
                )
                if not is_duplicate:
                    intersections.append((intersect_x, intersect_y))

        # Sort the intersection points based on x-coordinate
        intersections.sort(key=lambda point: point[0])

        # Label the sorted intersection points
        for point in intersections:
            labeled_intersections[point] = current_label
            
            # Draw the intersection point and label on the image
            x, y = int(point[0]), int(point[1])
            cv2.circle(img, (x, y), 5, (0, 0, 255), -1)  # Draw circle at intersection
            cv2.putText(img, str(current_label), (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)  # Label
            
            current_label += 1  # Increment label for the next intersection

    labeled_intersections_dict = {label: (float(point[0]), float(point[1])) for point, label in labeled_intersections.items()}
    return labeled_intersections_dict, img


# In[ ]:


def process_images_in_directory(directory_path, output_directory):
    os.makedirs(output_directory, exist_ok=True)
    edges_directory = os.path.join(output_directory, 'Edges')
    os.makedirs(edges_directory, exist_ok=True)
    
    files = os.listdir(directory_path)
    files.sort()
    
    # Iterate over all files in the specified directory
    for filename in os.listdir(directory_path):
        # Check if the file is a valid image file
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(directory_path, filename)
            img = cv2.imread(img_path)
            
            # Perform edge detection
            edges = detect_edges(img, low_threshold=25, high_threshold=100)
            
            # Save the edge image to the Edges folder
            edge_image_path = os.path.join(edges_directory, filename)
            cv2.imwrite(edge_image_path, edges)
            
            # Perform Hough line detection
            lines = hough_lines(edges, rho=1, threshold=30, max_line_gap=10)
            if lines is not None:
                # Format the lines for grouping
                formatted_lines = [[line[0]] for line in lines]

                # Group lines
                grouped_lines, groups = group_lines(formatted_lines, img.shape[1], img.shape[0], angle_threshold=np.pi/40, distance_threshold=15)
                
                # Draw grouped lines on the original image
                draw_lines(img, grouped_lines, (255, 0, 0))  # Draw in red
                
                # Find intersections and label on image
                intersections, labeled_img = find_intersections(grouped_lines, img, img.shape[0], img.shape[1])

                # Save the processed image to the output directory
                output_image_path = os.path.join(output_directory, filename)
                cv2.imwrite(output_image_path, labeled_img)

                # Save intersections to a JSON file
                intersections_path = os.path.join(output_directory, f"{os.path.splitext(filename)[0]}_intersections.json")
                with open(intersections_path, 'w') as f:
                    json.dump(intersections, f, indent=4)
                
                # Optionally display results
                plt.figure(figsize=(10, 10))
                plt.subplot(1, 2, 1)
                plt.title('Edges')
                plt.imshow(edges, cmap='gray')
                
                plt.subplot(1, 2, 2)
                plt.title(f"Detected Lines in {filename}")
                plt.imshow(cv2.cvtColor(labeled_img, cv2.COLOR_BGR2RGB))
                plt.show()

    print(f"Processed images and intersections have been saved to {output_directory}")


# In[ ]:


output_dir= 'HW8/output'


# In[ ]:


world_points_file= 'HW8/world.json' # choose json file containing world coordinates


# In[ ]:


image_coords_files= output_dir # choose directory containing json files with corner information for each picture


# In[ ]:


# process_images_in_directory('HW8/dataset_own3rr', 'HW8/dataset_own3rr_o')


# In[ ]:


def compute_homography(src_pts, dst_pts):
    """
    Computes the homography matrix using Direct Linear Transformation (DLT).
    
    Parameters:
    src_pts (ndarray): Source points in the first image (Nx2).
    dst_pts (ndarray): Corresponding destination points in the second image (Nx2).
    
    Returns:
    H (ndarray): The 3x3 homography matrix.
    """
    num_points = src_pts.shape[0]
    A = []

    # Construct the matrix A from the point correspondences
    for i in range(num_points):
        x_src, y_src = src_pts[i]
        x_dst, y_dst = dst_pts[i]
        A.append([-x_src, -y_src, -1, 0, 0, 0, x_dst * x_src, x_dst * y_src, x_dst])
        A.append([0, 0, 0, -x_src, -y_src, -1, y_dst * x_src, y_dst * y_src, y_dst])

    A = np.array(A)

    # Solve for the homography matrix using SVD
    U, S, Vt = np.linalg.svd(A)
    H = Vt[-1].reshape((3, 3))

    # Normalize the homography matrix so that H[2, 2] = 1
    H /= H[2, 2]

    return H


# In[ ]:


def compute_homography_lm(src_pts, dst_pts, initial_H):   

    def reprojection_error(h, src_pts, dst_pts):
        # Convert the 8-parameter vector into a full 3x3 homography matrix
        H = np.array([
            [h[0], h[1], h[2]],
            [h[3], h[4], h[5]],
            [h[6], h[7], 1.0]  # We fix H33 = 1
        ])
        
        # Apply the homography to src_pts
        src_pts_h = np.hstack((src_pts, np.ones((src_pts.shape[0], 1))))  # Make them homogeneous
        projected_pts_h = (H @ src_pts_h.T).T  # Transform the points
        
        # Convert back to Cartesian coordinates
        projected_pts = projected_pts_h[:, :2] / projected_pts_h[:, 2:3]
        
        # Compute the reprojection error
        error = dst_pts[:, :2] - projected_pts
        
        return error.flatten()

    def refine_homography_lm(src_pts, dst_pts, initial_H):
        """Library implementation of LM algorithm"""
        # Flatten the initial homography matrix to a vector of 8 parameters
        h0 = initial_H.flatten()[:8]  # We exclude the last element (H33)
        
        # Use LM optimization to minimize reprojection error
        result = least_squares(reprojection_error, h0, args=(src_pts, dst_pts))
        
        # Rebuild the refined homography matrix
        h_optimized = result.x
        refined_H = np.array([
            [h_optimized[0], h_optimized[1], h_optimized[2]],
            [h_optimized[3], h_optimized[4], h_optimized[5]],
            [h_optimized[6], h_optimized[7], 1.0]  # Set H33 to 1
        ])
        
        return refined_H
    
    new_H= refine_homography_lm(src_pts, dst_pts, initial_H)
    return new_H


# In[ ]:


def compute_homography_ransac(src_pts, dst_pts, num_iterations=1000, threshold=3.0):
    """
    Computes the homography matrix with RANSAC to handle outliers.
    """
    max_inliers = 0
    best_H = None
    best_inliers = None
    
    for _ in range(num_iterations):
        # Randomly select 4 pairs of points
        indices = np.random.choice(len(src_pts), 4, replace=False)
        src_sample = src_pts[indices]
        dst_sample = dst_pts[indices]
        
        # Compute the homography from these 4 points
        H = compute_homography(src_sample, dst_sample)
        
        # Apply homography to all source points
        src_pts_h = np.hstack((src_pts, np.ones((len(src_pts), 1))))
        projected_pts_h = (H @ src_pts_h.T).T
        projected_pts = projected_pts_h[:, :2] / projected_pts_h[:, 2:3]
        
        # Calculate the Euclidean distance to destination points
        distances = np.linalg.norm(dst_pts - projected_pts, axis=1)
        
        # Find inliers based on the threshold
        inliers = distances < threshold
        num_inliers = np.sum(inliers)
        
        # Update the best homography if more inliers are found
        if num_inliers > max_inliers:
            max_inliers = num_inliers
            best_H = H
            best_inliers = inliers
    
    # Recompute homography with inliers only
    if best_inliers is not None:
        refined_H = compute_homography(src_pts[best_inliers], dst_pts[best_inliers])
        # Further refine with LM optimization
        refined_H = compute_homography_lm(src_pts[best_inliers], dst_pts[best_inliers], refined_H)
    else:
        refined_H = None  # Fallback if no inliers found
    
    return refined_H, best_inliers


# In[ ]:


def load_json(filename):
    with open(filename, 'r') as f:
        return json.load(f)


# In[ ]:


def find_homographies(world_points_file, images_intersections_dir):
    # Load world points
    world_points = load_json(world_points_file)
    # Initialize a dictionary to hold homographies for each image
    homographies = {}

    files= os.listdir(images_intersections_dir)
    files.sort()
    
    # Iterate through the files in the intersections directory
    for filename in files:
        if filename.endswith('.json'):  # Assuming intersection points are stored in JSON format
            intersections_path = os.path.join(images_intersections_dir, filename)
            intersections = load_json(intersections_path)

            # Prepare source and destination points
            src_pts = []
            dst_pts = []

            for label, point in intersections.items():
                # Convert label to integer to access world points
                label_int = int(label)
                if str(label_int) in world_points:
                    # Append the corresponding world point
                    src_pts.append(world_points[str(label_int)])
                    # Append the intersection point from the image
                    dst_pts.append(point)
            if len(src_pts) >= 4 and len(dst_pts) >= 4:  # Need at least 4 points to compute homography
                src_pts = np.array(src_pts)
                dst_pts = np.array(dst_pts)

                # Compute homography
                H_refined, _= compute_homography_ransac(src_pts, dst_pts)
                
                # Save the homography for the current image
                homographies[filename] = H_refined.tolist()  

    return homographies


# In[ ]:


def draw_world_points_on_images(world_points_file, images_intersections_dir, homographies, output_dir):
    # Load world points
    world_points = load_json(world_points_file)
    os.makedirs(output_dir, exist_ok=True)

    # Iterate through homographies
    for json_filename, H in homographies.items():
        # Find the corresponding image file
        image_filename = json_filename.replace('_intersections.json', '.jpeg')
        image_path = os.path.join(images_intersections_dir, image_filename)
        
        # Load the image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Image {image_filename} not found.")
            continue

        # Transform each world point and draw on the image
        for label, world_point in world_points.items():
            # Convert world point to homogeneous coordinates
            world_point_h = np.array([*world_point, 1])

            # Project the world point to the image plane using the homography
            img_point_h = H @ world_point_h
            img_point = img_point_h[:2] / img_point_h[2]  # Normalize

            # Draw a green circle of radius 2 at the computed image point
            img_point_int = tuple(np.round(img_point).astype(int))
            cv2.circle(image, img_point_int, radius=5, color=(0, 255, 0), thickness=-1)

        output_path= os.path.join(output_dir, f"reprojected_{image_filename}")
        # Save or display the modified image with the circles drawn
        cv2.imwrite(output_path, image)
        print(f"Image saved at {output_path}")


# In[ ]:


def compute_intrinsic_params(homographies):
    """ Estimate intrinsic camera parameters from homographies.

    Parameters:
        homographies (dict): A dictionary of homography matrices indexed by some keys.

    Returns:
        numpy.ndarray: The intrinsic camera matrix K.
    """
    # Step 1: Construct the V matrix from homographies
    V = _construct_v_matrix(homographies)

    # Step 2: Perform Singular Value Decomposition (SVD)
    b = _perform_svd(V)

    # Step 3: Compute the intrinsic parameters K from the vector b
    K = _calculate_intrinsic_parameters(b)

    return K

def _construct_v_matrix(homographies):
    """Construct the V matrix from homographies."""
    V = []
    
    for H in homographies.values():
        H = np.array(H)
        h11, h12, h13 = (H[0, 0], H[1, 0], H[2, 0])
        h21, h22, h23 = (H[0, 1], H[1, 1], H[2, 1])
        # Construct V for the current homography
        V.append([h11 * h21, h11 * h22 + h12 * h21, h12 * h22, h13 * h21 + h11 * h23, h13 * h22 + h12 * h23, h13 * h23])
        V.append([h11**2 - h21**2, 2 * h11 * h12 - 2 * h21 * h22, h12**2 - h22**2, 2 * h11 * h13 - 2 * h21 * h23, 2 * h12 * h13 - 2 * h22 * h23, h13**2 - h23**2])
    
    
    return np.array(V)

def _perform_svd(V):
    """Perform Singular Value Decomposition (SVD) and return the solution vector."""
    _, _, v = np.linalg.svd(V.T @ V)
    b = v[-1]  # The last row of Vh corresponds to the solution of the zero vector
    return b

def _calculate_intrinsic_parameters(b):
    """Compute the intrinsic camera matrix K from the vector b."""
    w = np.array([[b[0], b[1], b[3]],
                  [b[1], b[2], b[4]],
                  [b[3], b[4], b[5]]])

    # Calculate intrinsic parameters K from w
    det = w[0, 0] * w[1, 1] - w[0, 1]**2
    if det == 0:
        raise ValueError("Cannot compute intrinsic parameters.")

    # Corrected computation of x0 and lambda based on first approach
    x0 = (w[0, 1] * w[0, 2] - w[0, 0] * w[1, 2]) / det
    lamda = w[2, 2] - (w[0, 2]**2 + x0 * (w[0, 1] * w[0, 2] - w[0, 0] * w[1, 2])) / w[0, 0]
    alpha_x = np.sqrt(lamda / w[0, 0])
    alpha_y = np.sqrt(lamda * w[0, 0] / det)
    s = -w[0, 1] * alpha_x**2 * alpha_y / lamda
    y0 = s * x0 / alpha_y - w[0, 2] * alpha_x**2 / lamda

    # Construct the intrinsic matrix K
    K = np.array([[alpha_x, s, x0],
                  [0, alpha_y, y0],
                  [0, 0, 1]])

    return K


# In[ ]:


def compute_extrinsic_params(homographies, K):
    """ Estimate extrinsic parameters (rotation and translation) from homographies. """
    extrinsics = []
    K_inv = np.linalg.inv(K)

    for H in homographies:
        H_matrix = np.array(homographies[H])
        normalized_H = K_inv @ H_matrix
        

        # Extract rotation and translation
        r1 = normalized_H[:, 0]
        scale_factor= np.linalg.norm(r1)
        r1 = r1/scale_factor
        
        r2 = normalized_H[:, 1]
        r2 = r2/scale_factor
        r3 = np.cross(r1, r2)  # Ensure r3 is orthogonal to r1 and r2

        # Stack r1, r2, r3 to form a matrix
        R = np.column_stack((r1, r2, r3))

        # Condition R using SVD to ensure orthogonality
        U, _, Vt = np.linalg.svd(R)
        R_conditioned = U @ Vt  # Ensure orthogonality

        # # Check the determinant
        # if np.linalg.det(R_conditioned) < 0:
        #     R_conditioned[:, 2] *= -1  # Flip the last column if determinant is negative

        # Extract translation
        t = normalized_H[:, 2]/scale_factor

        # Store rotation and translation
        extrinsics.append({
            'rotation': R_conditioned,
            'translation': t
        })

    return extrinsics


# In[ ]:


homographies= find_homographies(world_points_file, image_coords_files)


# In[ ]:


K= compute_intrinsic_params(homographies)


# In[ ]:


draw_world_points_on_images(world_points_file, image_coords_files, homographies, os.path.join(output_dir, 'homography_reprojected_own')) # to check if the computed homographies are correct or not


# In[ ]:


extrinsic_params= compute_extrinsic_params(homographies, K)


# In[ ]:


def load_points(world_path: str, image_coords_dir):
    """Loads world and image coordinates from JSON files."""
    world_data = load_json(world_path)
    points_3d = np.array([[coord[0], coord[1], 0] for coord in world_data.values()], dtype=np.float32)
    coords_files= os.listdir(image_coords_dir)
    coords_files.sort()
    points_2d = []
    for file in coords_files:
        if file.endswith('.json'):
            full_file= os.path.join(image_coords_dir, file)
            image_data = load_json(full_file)
            points_2d.append(np.array([image_data[str(key)] for key in world_data.keys()], dtype=np.float32))

    return points_3d, points_2d


# In[ ]:


def project_points(points_3d, intrinsic_matrix, rvec, tvec, distCoeffs=None):
    """Projects 3D points to 2D using intrinsic and extrinsic parameters."""
    if distCoeffs is not None:
        distCoeffs = np.asarray(distCoeffs)
        if distCoeffs.size == 3:  
            distCoeffs = np.array([*distCoeffs, 0, 0]) 

    points_2d, _ = cv2.projectPoints(points_3d, rvec, tvec, intrinsic_matrix, distCoeffs)
    return points_2d.reshape(-1, 2)


# In[ ]:


def reprojection_error(params, points_3d, points_2d, num_images, distCoeffs=None):
    """Computes reprojection error for all images."""
    intrinsic_matrix = params[:9].reshape(3, 3)
    errors = []
    offset = 9
    for i in range(num_images):
        rvec = params[offset:offset+3].reshape(3, 1)
        tvec = params[offset+3:offset+6].reshape(3, 1)
        projected_points = project_points(points_3d, intrinsic_matrix, rvec, tvec, distCoeffs)
        errors.append((projected_points - points_2d[i]).ravel())
        offset += 6
    return np.hstack(errors)


# In[ ]:


def rotation_matrix_to_rodrigues(rotation_matrix):
    """Convert a 3x3 rotation matrix to a Rodrigues vector."""
    angle = np.arccos((np.trace(rotation_matrix) - 1) / 2)
    if angle < 1e-6:
        # Angle is close to zero, return zero vector
        return np.zeros(3)
    
    # Calculate the rotation axis normalized
    rx = (rotation_matrix[2, 1] - rotation_matrix[1, 2]) / (2 * np.sin(angle))
    ry = (rotation_matrix[0, 2] - rotation_matrix[2, 0]) / (2 * np.sin(angle))
    rz = (rotation_matrix[1, 0] - rotation_matrix[0, 1]) / (2 * np.sin(angle))
    
    axis = np.array([rx, ry, rz])
    
    # Return the Rodrigues vector (axis * angle)
    return axis * angle


# In[ ]:


def rodrigues_to_rotation_matrix(rvec):
    """Convert a Rodrigues vector to a 3x3 rotation matrix."""
    # Ensure that rvec is a numpy array and is of shape (3,)
    rvec = np.asarray(rvec).flatten()

    theta = np.linalg.norm(rvec)
    if theta < 1e-6:
        # If the angle is too small, return the identity matrix
        return np.eye(3)
    
    # Normalize the rotation vector to get the rotation axis
    k = rvec / theta
    
    # Compute the skew-symmetric cross-product matrix K
    K = np.array([
        [0, -k[2], k[1]],
        [k[2], 0, -k[0]],
        [-k[1], k[0], 0]
    ])
    
    # Compute the rotation matrix using Rodrigues' rotation formula
    rotation_matrix = (
        np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * np.dot(K, K)
    )
    return rotation_matrix


# In[ ]:


def generate_params(intrinsic_matrix: np.ndarray, extrinsics: List[Dict]) -> np.ndarray:
    """Generates the parameter vector for optimization."""
    # Flatten intrinsic matrix
    initial_params = intrinsic_matrix.ravel().tolist()
    
    # Flatten extrinsic parameters (convert rotation matrices to Rodrigues vectors)
    for extrinsic in extrinsics:
        rotation_matrix = extrinsic['rotation']
        translation_vector = extrinsic['translation']
        
        # Convert rotation matrix to Rodrigues rotation vector
        # rvec, _ = cv2.Rodrigues(rotation_matrix)
        rvec = rotation_matrix_to_rodrigues(rotation_matrix) 
        initial_params.extend(rvec.ravel())
        initial_params.extend(translation_vector)
    
    return np.array(initial_params)


# In[ ]:


def convert_to_extrinsics_dict(rvecs, tvecs):
    """Converts rotation and translation vectors to the extrinsics dictionary format."""
    extrinsics = []
    for rvec, tvec in zip(rvecs, tvecs):
        # Convert Rodrigues rotation vector to rotation matrix
        # rotation_matrix, _ = cv2.Rodrigues(rvec)
        rotation_matrix = rodrigues_to_rotation_matrix(rvec)
        translation_vector = tvec.flatten()
        
        # Add to extrinsics dictionary list
        extrinsics.append({
            'rotation': rotation_matrix,
            'translation': translation_vector
        })
    return extrinsics


# In[ ]:


def optimize_parameters(points_3d, points_2d, num_images, initial_params):
    """Optimizes intrinsic and extrinsic parameters using Levenberg-Marquardt."""
    # Optimize parameters using LM algorithm
    result = least_squares(
        reprojection_error,
        initial_params,
        args=(points_3d, points_2d, num_images),
        method='lm'
    )

    # Retrieve optimized intrinsic matrix and extrinsic parameters
    optimized_intrinsic_matrix = result.x[:9].reshape(3, 3)
    optimized_extrinsics = result.x[9:].reshape(num_images, 6)  # Each row has rvec and tvec for an image
    optimized_rvecs = [optimized_extrinsics[i, :3].reshape(3, 1) for i in range(num_images)]
    optimized_tvecs = [optimized_extrinsics[i, 3:].reshape(3, 1) for i in range(num_images)]

    # Convert optimized rvecs and tvecs to the extrinsics dictionary format
    optimized_extrinsics_dict = convert_to_extrinsics_dict(optimized_rvecs, optimized_tvecs)

    return optimized_intrinsic_matrix, optimized_extrinsics_dict, result


# In[ ]:


def calculate_mean_reprojection_error(intrinsic_matrix, extrinsics, points_3d, points_2d, num_images, distCoeffs=None):
    """Calculates mean reprojection error for given intrinsic and extrinsic parameters."""
 
    if distCoeffs is None:
        # Generate parameter vector from intrinsic and extrinsic parameters
        params = generate_params(intrinsic_matrix, extrinsics)
    # Calculate reprojection error using the generated parameter vector
        final_error = reprojection_error(params, points_3d, points_2d, num_images, distCoeffs)
    
    else:
        params = generate_params_rad(intrinsic_matrix, extrinsics, distCoeffs)
        final_error = reprojection_error_rad(params, points_2d, points_3d, num_images)
    
    # Compute and return the mean reprojection error
    mean_reprojection_error = np.mean(np.sqrt(final_error**2))
    return mean_reprojection_error


# In[ ]:


def radial_distortion(K, distortion_coeffs, points_3d, extrinsics):
    """
    Apply radial distortion to the 3D points and return the 2D points.
    
    Parameters:
        K (numpy.ndarray): Intrinsic matrix.
        distortion_coeffs (tuple): Radial distortion coefficients (k1, k2, k3).
        points_3d (list): List of 3D points.
        extrinsics (list): List of extrinsic parameters (rotation and translation) for each image.
        
    Returns:
        list: List of corrected 2D points for each image.
    """
    corrected_points = []
    if not isinstance(extrinsics, list):
        extrinsics= [extrinsics]

    for i, extrinsic in enumerate(extrinsics):

        R = extrinsic['rotation']  # Rotation matrix for the current image
        T = extrinsic['translation'].reshape(3, 1)  # Translation vector for the current image
        
        # Project points to image
        projected_points = []
        for point_3d in points_3d:
            point_3d_h = np.array([*point_3d, 1])  # Homogeneous coordinates
            # Project to camera space
            point_cam = R @ point_3d_h[:3] + T.flatten()
            # Project to image plane
            point_proj = K @ point_cam
            point_proj = (point_proj / point_proj[2])[:2]  # Normalize by z
            projected_points.append(point_proj)
        
        # Convert to numpy array for distortion correction
        projected_points = np.array(projected_points)
        x = projected_points[:, 0]
        y = projected_points[:, 1]
        r_squared = x**2 + y**2

        k1, k2, = distortion_coeffs
        # Apply radial distortion
        x_distorted = x + (x-K[0][2])*(k1*r_squared + k2*r_squared**2)
        y_distorted = y + (y-K[1][2])*(k1*r_squared + k2*r_squared**2)

        # Store the corrected points for the current image
        corrected_points.append(np.column_stack((x_distorted, y_distorted)))

    return corrected_points

def generate_params_rad(intrinsic_matrix: np.ndarray, extrinsics: list, distortion_coeffs=(0, 0)) -> np.ndarray:
    """Generates the parameter vector for optimization."""
    initial_params = intrinsic_matrix.ravel().tolist()  # Flatten intrinsic matrix

    for extrinsic in extrinsics:
        rotation_matrix = extrinsic['rotation']
        translation_vector = extrinsic['translation']
        rvec = rotation_matrix_to_rodrigues(rotation_matrix)  # Convert to Rodrigues vector
        initial_params.extend(rvec.ravel())
        initial_params.extend(translation_vector)
    initial_params.extend(distortion_coeffs)  # Add distortion coefficients

    return np.array(initial_params)

def unpack_params(params, num_images):
    """Unpacks the parameter vector into intrinsic matrix, distortion coeffs, and extrinsics."""
    K = params[:9].reshape(3, 3)  # Intrinsic matrix

    extrinsics = []
    index = 9
    for _ in range(num_images):
        rvec = params[index:index+3].reshape(3, 1)
        rotation_matrix = rodrigues_to_rotation_matrix(rvec)
        translation_vector = params[index+3:index+6]
        extrinsics.append({'rotation': rotation_matrix, 'translation': translation_vector.reshape(3,)})
        index += 6
    distortion_coeffs = params[-2:]  # Radial distortion coefficients

    return K, distortion_coeffs, extrinsics

def reprojection_error_rad(params, points_2d, points_3d, num_images):
    """Calculate reprojection error for optimization of K, distortion, and extrinsics."""
    K, distortion_coeffs, extrinsics = unpack_params(params, num_images)
    projected_points = radial_distortion(K, distortion_coeffs, points_3d, extrinsics)
    projected_points_flat = np.vstack(projected_points)
    points_2d_flat = np.vstack(points_2d)

    return (projected_points_flat - points_2d_flat).ravel()

def estimate_all_params_distortion(points_2d, K, points_3d, extrinsics):
    """Optimize intrinsic matrix, distortion coefficients, and extrinsics."""
    initial_params = generate_params_rad(K, extrinsics)  # Generate initial params
    result = least_squares(reprojection_error_rad, initial_params, args=(points_2d, points_3d, len(extrinsics)))
        # Retrieve optimized intrinsic matrix and extrinsic parameters
    optimized_intrinsic_matrix, optimized_dist_coeffs, optimized_extrinsics = unpack_params(result.x, len(extrinsics))

    return optimized_intrinsic_matrix, optimized_extrinsics, optimized_dist_coeffs


# In[ ]:


def reproject_points(points_3d, K, extrinsic, correct_rad_distortion=False, distortion_coeffs=None):
    """Reprojects 3D world points to 2D image points using given intrinsic and extrinsic parameters.
    
    Parameters:
        points_3d (list): List of 3D points.
        K (numpy.ndarray): Intrinsic matrix.
        extrinsic (dict): Extrinsic parameters containing rotation and translation.
        
    Returns:
        list: List of reprojected 2D points.
    """
    # Extract rotation and translation
    R = extrinsic['rotation']
    T = extrinsic['translation'].reshape(3, 1)
    
    # Construct the camera projection matrix P = K * [R | T]
    extrinsic_matrix = np.hstack((R, T))  # Combine R and T
    P = K @ extrinsic_matrix              # Camera projection matrix

    # Apply radial distortion correction if specified
    if correct_rad_distortion and distortion_coeffs is not None:
        points_2d = np.array(radial_distortion(K, distortion_coeffs, points_3d, extrinsic)).squeeze()
    
        
    else:
        # Convert 3D points to homogeneous coordinates
        points_3d_h = np.hstack((points_3d, np.ones((len(points_3d), 1))))
        
        # Project points using the camera projection matrix
        projected_points_h = (P @ points_3d_h.T).T  # Projected points in homogeneous coordinates
        
        # Normalize by the last (z) coordinate to get 2D image points
        points_2d = projected_points_h[:, :2] / projected_points_h[:, 2, np.newaxis]
    
    
    return points_2d

def visualize_reprojection(images_folder, K, extrinsics, points_3d, output_folder, suffix, correct_rad_distortion=False, distortion_coefficients=None):
    """Visualizes reprojection on images for given intrinsic and extrinsic parameters and saves the images."""
    files = os.listdir(images_folder)
    os.makedirs(output_folder, exist_ok=True)
    files.sort()
    json_files = [file for file in files if file.endswith('json')]
    
    for json_file, extrinsic in zip(json_files, extrinsics):  # Pair each JSON file with an extrinsic parameter
        # Load corresponding image
        image_name = os.path.basename(json_file).replace('_intersections.json', '.jpeg')
        image_path = os.path.join(images_folder, image_name)
        image = cv2.imread(image_path)

        if image is None:
            image_name = os.path.basename(json_file).replace('_intersections.json', '.jpg')
            image_path = os.path.join(images_folder, image_name)
            image = cv2.imread(image_path)
        
        # Reproject points with the current extrinsic parameters
        reprojected_points = reproject_points(points_3d, K, extrinsic, correct_rad_distortion, distortion_coefficients)

        # Plot points on the image
        for pt in reprojected_points:
            # Draw points in green
            cv2.circle(image, tuple(int(x) for x in pt), 2, (0, 255, 0), -1)

        # Save the resulting image
        output_image_path = os.path.join(output_folder, f"{os.path.splitext(image_name)[0]}_{suffix}.jpg")
        cv2.imwrite(output_image_path, image)
        print(f"Saved reprojection image: {output_image_path}")
       
    return None 


# In[ ]:


points_world, points_image= load_points(world_points_file, image_coords_files)


# In[ ]:


initial_params= generate_params(K, extrinsic_params)


# In[ ]:


optimized_K, optimized_extrinsic, result = optimize_parameters(
        points_world, points_image, len(points_image), initial_params
    )


# In[ ]:


rad_optimized_K, rad_optimized_extrinsic, rad_optimized_dist_coeffs = estimate_all_params_distortion(points_image, K, points_world, extrinsic_params)


# In[ ]:


calculate_mean_reprojection_error(K, extrinsic_params, points_world, points_image, len(points_image))


# In[ ]:


calculate_mean_reprojection_error(optimized_K, optimized_extrinsic, points_world, points_image, len(points_image)) # mean projection error for optimized parameters


# In[ ]:


calculate_mean_reprojection_error(rad_optimized_K, rad_optimized_extrinsic, points_world, points_image, len(points_image), rad_optimized_dist_coeffs)


# In[ ]:


reprojected_points=visualize_reprojection(image_coords_files, K, extrinsic_params, points_world, os.path.join(output_dir, 'ReprojectionU'), 'u')


# In[ ]:


visualize_reprojection(image_coords_files, optimized_K, optimized_extrinsic, points_world, os.path.join(output_dir, 'ReprojectionO'), 'o')


# In[ ]:


reprojected_points=visualize_reprojection(image_coords_files, rad_optimized_K, rad_optimized_extrinsic, points_world, os.path.join(output_dir, 'ReprojectionR'), 'rad', correct_rad_distortion=True, distortion_coefficients=rad_optimized_dist_coeffs)


# ### Plot and Visualize the camera axis

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import json

# Load camera calibration pattern from world.json
with open(world_points_file, "r") as file:
    pattern = json.load(file)
    
# Convert pattern to a NumPy array for plotting
pattern_points = np.array([coord + [0] for coord in pattern.values()])


# In[ ]:


#select 4 camera poses to plot
selected_extrinsic= []
for i in [38,3,9,27]:
    selected_extrinsic.append(optimized_extrinsic[i])


# In[ ]:


plane_scale = 2.0  # Larger plane size
arrow_scale = 2.0  # Longer arrows

# Set up the 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the camera poses
for i, cam_pose in enumerate(selected_extrinsic):
    R = cam_pose['rotation']
    t = cam_pose['translation'].reshape(-1, 1)
    C = -R.T @ t  # Camera center in world coordinates

    # Define camera frame axes in the world frame
    X_cam = R.T @ np.array([1, 0, 0]) * arrow_scale + C.flatten()
    Y_cam = R.T @ np.array([0, 1, 0]) * arrow_scale + C.flatten()
    Z_cam = R.T @ np.array([0, 0, 1]) * arrow_scale + C.flatten()

    # Plot camera axes
    ax.quiver(*C.flatten(), *(X_cam - C.flatten()), color='r', label='Xcam' if i == 0 else "")
    ax.quiver(*C.flatten(), *(Y_cam - C.flatten()), color='g', label='Ycam' if i == 0 else "")
    ax.quiver(*C.flatten(), *(Z_cam - C.flatten()), color='b', label='Zcam' if i == 0 else "")

    # Plot the camera principal plane as a larger, semi-transparent rectangle
    plane_corners = np.array([[1, 1, 0], [-1, 1, 0], [-1, -1, 0], [1, -1, 0]]) * plane_scale
    plane_corners_world = R.T @ plane_corners.T + C
    ax.plot_trisurf(plane_corners_world[0], plane_corners_world[1], plane_corners_world[2], color=np.random.rand(3,), alpha=0.5)

# Plot calibration pattern on Z=0 plane
ax.scatter(pattern_points[:, 0], pattern_points[:, 1], pattern_points[:, 2], c='k', marker='s')

# Label and show the plot
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.legend()
ax.view_init(elev=20, azim=20) 
plt.show()


# In[ ]:




