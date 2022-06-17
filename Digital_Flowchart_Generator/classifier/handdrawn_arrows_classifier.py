import os
import cv2
import numpy as np
import math
import matplotlib.pyplot as plt


class arrows_classifier:
    """
    This is a class that takes in an image of the arrows only and isola

    Attributes:
        model_path (str): The path to the hand-drawn flowchart shapes classifier
    """
    def __init__(self, img_arrows_only):
        """
        The constructor for arrows_classifier class
  
        Parameters:
            model_path (str): The path to the hand-drawn flowchart shapes classifier
        """
        height, width = img_arrows_only.shape
        self.img_area = width * height
        self.img_gray = img_arrows_only
        self.img_rgb = cv2.cvtColor(img_arrows_only, cv2.COLOR_GRAY2RGB)

    def find_connected_components_info(self):
        """
        Finds all the connected components in the image and generates stats on them
  
        Returns:
            stats_filtered: list containing stats of all connected components in the image
            centroids_filtered: list containing the centroids of all connected components in the image
            num_components_filtered (int): number of connected components in the image
        """
        #find all your connected components (white blobs in your image)
        num_components, labels, stats, centroids = cv2.connectedComponentsWithStats(self.img_gray, connectivity=8)
        #remove background from connected component lists
        stats_filtered = stats[1:]
        centroids_filtered = centroids[1:]
        num_components_filtered = num_components - 1
        return stats_filtered, centroids_filtered, num_components_filtered
    
    def calculate_distance(self,x1,y1,x2,y2):
        """
        Finds the distance between two points
  
        Parameters:
            x1: x-coordinate of pt1
            y1: y-coordinate of pt1
            x2: x-coordinate of pt2
            y2: y-coordinate of pt2
        
        Returns:
            dist: distance between the two points
        """
        dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        return dist

    def plot_images(self, images, is_gray, figure_number, num_rows, num_cols, title_str):
        """
        Takes in a set of images and parameters and creates a figure and a grid of subplots
  
        Parameters:
            images: array containing all the images to be included in the subplots
            is_gray (bool): flag to set whether the images should be plotted as gray-scale or RGB
            figure_number (int): the unique figure number
            num_rows (int): number of rows in the subplot
            num_cols (int): number of columns in the subplot
            title_str (str): string containing the title of the figure
        """
        plt.figure(figure_number)
        for i in range(0,len(images)):
            plt.subplot(num_rows, num_cols, i+1)
            if is_gray:
                plt.imshow(images[i], cmap='gray')
            else:
                plt.imshow(images[i])
            plt.title(title_str + '#{}'.format(str(i)))

    def find_arrows(self):
        """
        This entry point function return a dict containing all the arrows information
        such as the coordinate of every arrows head and tail

        Returns:
            arrows_info (dict): dictionary containing all the arows and their information 
        """
        arrows_info = []
        stats_filtered, centroids_filtered, num_components_filtered = self.find_connected_components_info()
        for i in range(0, num_components_filtered):
            #bounding box top left corner stats for the connected component
            bb_x = stats_filtered[i][0]
            bb_y = stats_filtered[i][1]
            bb_width = stats_filtered[i][2]
            bb_height = stats_filtered[i][3]
            connected_component_aspect_ratio = bb_width / bb_height
            connected_component_area = stats_filtered[i][4]
            centroid_x = int(centroids_filtered[i][0])
            centroid_y = int(centroids_filtered[i][1])
            if connected_component_area < 400:
                continue

            cv2.rectangle(self.img_rgb,(bb_x, bb_y), (bb_x + bb_width, bb_y + bb_height) ,(0,0,255), 2)
            cv2.circle(self.img_rgb, (bb_x + int(bb_width/2), bb_y + int(bb_height/2)), 5, (255, 255, 0), -1)
            cv2.circle(self.img_rgb, (centroid_x, centroid_y), 5, (255, 0, 0), -1)

            #Top left, top right, bottom left, bottom right
            bb_pts = [(bb_x, bb_y), (bb_x + bb_width, bb_y), (bb_x, bb_y + bb_height), (bb_x + bb_width, bb_y + bb_height)]

            bb_box_corners_distances_to_centroid = []
            for bb_pt in bb_pts:
                dist = self.calculate_distance(centroid_x, centroid_y, bb_pt[0], bb_pt[1])
                bb_box_corners_distances_to_centroid.append(dist)

            #find closest bbox corner point to the centroid of the arrow
            closest_pt = bb_pts[bb_box_corners_distances_to_centroid.index(min(bb_box_corners_distances_to_centroid))]
            furthest_pt = bb_pts[bb_box_corners_distances_to_centroid.index(max(bb_box_corners_distances_to_centroid))]
            arrow_info = {
                "arrow_number": i,
                "tail_coordinate": furthest_pt,
                "head_coordinate": closest_pt,
                "centroid": (centroid_x, centroid_y) 
            }
            arrows_info.append(arrow_info)

            cv2.circle(self.img_rgb, closest_pt, 5, (0,255,0), -1)
            cv2.circle(self.img_rgb, furthest_pt, 5, (0,255,0), -1)
        self.plot_images([self.img_rgb], False, 15, 1, 1, "img")
        return arrows_info
