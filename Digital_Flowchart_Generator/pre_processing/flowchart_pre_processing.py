import cv2
import matplotlib.pyplot as plt
import numpy as np


class decompose_flowchart:
    """
    This is a class that takes in a camera image of a hand-drawn flowchart and decomposes
    the flowchart into two seperate images, one containing the shapes and the other 
    containing the arrows

    Attributes:
        flowchart_img_path (str): The path to the image file
    """
    def __init__(self, flowchart_img_path):
        """
        The constructor for decompose_flowchart class.
  
        Parameters:
           flowchart_img_path (str): The path to the image file
        """
        #read image
        img_bgr = cv2.imread(flowchart_img_path)
        self.img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        #convert to Gray
        self.img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        self.plot_images([self.img_gray], True, 1, 1, 1, "img")
    
    def binarize_flowchart(self, img_gray):
        """
        Takes in a gray-scale image of teh flowchart and turns it into a binary image
  
        Parameters: 
            img_gray: Gray-scale image of the flowchart
          
        Returns:
            inverted_img_thresholded: A binary image of the flowchart
        """
        #gaussian blur to blur image
        blurred_gray_img = cv2.GaussianBlur(img_gray, (13, 13), 0)        
        #thresholding
        img_thresholded = cv2.adaptiveThreshold(blurred_gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5, 1)
        #filter to remove salt and pepper noise
        filtered_gray_img = cv2.medianBlur(img_thresholded, 5)
        #invert pixel values
        inverted_img_thresholded = cv2.bitwise_not(filtered_gray_img)
        self.plot_images([blurred_gray_img, img_thresholded, filtered_gray_img, inverted_img_thresholded], True, 2, 2, 2, "img")
        return inverted_img_thresholded

    def find_flowchart_components(self, img_thresholded):
        """
        Takes in a binary image of the flowchart of the flowchart and finds all 
        the flowchart components (shapes and arrows)
  
        Parameters: 
            img_thresholded: A binary image of the flowchart
          
        Returns:
            img_contours_filled: A binary image of the flowchart where shapes and arrows are filled with white
                                 to have a clear separation from the background which is black
        """
        #Canny edge detection
        canny_lower_thres = 125
        canny_upper_thres = 250
        canny_img_edges = cv2.Canny(img_thresholded, canny_lower_thres, canny_upper_thres)
        #Find contours
        contours, _ = cv2.findContours(canny_img_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        #Fill outer contours with white
        img_contours_filled = cv2.drawContours(canny_img_edges.copy(), contours, -1, (255,255,255), thickness=cv2.FILLED)
        self.plot_images([canny_img_edges,img_contours_filled], True, 3, 1, 2, "img")
        return img_contours_filled

    def split_flowchart_components(self, img_contours_filled):
        """
        Takes in the binary image where flowchart components are filled with white and 
        seperates the arrows and shapes into seperate images
  
        Parameters: 
            img_contours_filled: A binary image of the flowchart where shapes and arrows are filled with white
                                 to have a clear separation from the background which is black
          
        Returns:
            img_shapes_only: Binary image containing the flowchart shapes only
            dilated_arrows: Binary image containing the flowchart arrows only
        """
        #Remove arrows(connecting bridges) from shapes using morph opening
        kernel = np.ones((3,3), np.uint8)                               
        img_shapes_only = cv2.morphologyEx(img_contours_filled, cv2.MORPH_OPEN, kernel,iterations=4) 
        #get arrows only
        img_arrows_only = cv2.subtract(img_contours_filled, img_shapes_only)
        img_arrows_only = cv2.morphologyEx(img_arrows_only, cv2.MORPH_OPEN, kernel,iterations=1) 
        #dilate to strengthen arrows and fill holes
        kernel = np.ones((3,3), np.uint8)                               
        dilated_arrows = cv2.dilate(img_arrows_only, kernel, iterations=2)
        self.plot_images([img_shapes_only, img_arrows_only, dilated_arrows], True, 4, 1, 3, "img")
        return img_shapes_only, dilated_arrows       

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

    def start_decomposition(self):
        """
        This entry point function uses the other utility methods in the decompose_flowchart
        class and decomposes the flowchart into shapes and arrows

        Returns:
            img_shapes_only: Binary image containing the flowchart shapes only
            img_arrows_only: Binary image containing the flowchart arrows only
        """
        img_thresholded = self.binarize_flowchart(self.img_gray)
        img_contours_filled = self.find_flowchart_components(img_thresholded)
        img_shapes_only, img_arrows_only = self.split_flowchart_components(img_contours_filled)
        return img_shapes_only, img_arrows_only
