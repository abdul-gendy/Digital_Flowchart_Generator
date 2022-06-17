
import math
import cv2
import matplotlib.pyplot as plt


class reconstruct_flowchart:
    """
    This is a class that takes in detailed information about the shapes
    in the flowchart and the arrows in the flowchart and then connects
    the shapes and arrows to have a defined layout for the flowchart

    Attributes:
        arrows_info (dict): dictionary containing all the arrows and their information 
        shapes_info (dict): dictionary containing all the shapes and their information 
        full_image_path (str): The path to the original image file
    """
    def __init__(self, arrows_info, shapes_info, full_image_path):
        """
        The constructor for train_shapes_classifier class.
  
        Parameters:
            arrows_info (dict): dictionary containing all the arrows and their information 
            shapes_info (dict): dictionary containing all the shapes and their information 
            full_image_path (str): The path to the original image file
        """
        self.arrows_info = arrows_info
        self.shapes_info = shapes_info
        img_bgr = cv2.imread(full_image_path)
        self.img_full_img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

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

    def establish_connections_between_shapes(self):
        """
        This entry point function  connects the shapes and arrows to 
        have a defined layout for the flowchart

        Returns:
            shapes_info (dict): dictionary containing all the shapes and their information with
                                added information regarding which shapes are linked to each other
        """
        for arrow_info in self.arrows_info:
            tx,ty = arrow_info["tail_coordinate"]
            hx,hy = arrow_info["head_coordinate"]
            tc_distance_to_shapes = []
            hc_distance_to_shapes = []
            #For every arrow find shape closest to the head and shape closest to tail and save info in dict
            for shape_info in self.shapes_info:
                cx, cy = shape_info["centroid"]
                tc_distance = self.calculate_distance(tx,ty,cx,cy)
                hc_distance = self.calculate_distance(hx,hy,cx,cy)
                tc_distance_to_shapes.append(tc_distance)
                hc_distance_to_shapes.append(hc_distance)
            tc_closest_shape = self.shapes_info[tc_distance_to_shapes.index(min(tc_distance_to_shapes))]
            hc_closest_shape = self.shapes_info[hc_distance_to_shapes.index(min(hc_distance_to_shapes))]
            tc_closest_shape["arrow_to"].append(hc_distance_to_shapes.index(min(hc_distance_to_shapes)))

            cv2.line(self.img_full_img_rgb, (tx,ty), tc_closest_shape["centroid"], (255,0,0), 2)
            cv2.line(self.img_full_img_rgb, (hx,hy), hc_closest_shape["centroid"], (0,0,255), 2)

        self.plot_images([self.img_full_img_rgb], False, 19, 1, 1, "img")
        return self.shapes_info
