import os
import cv2
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
from tqdm import tqdm


class handdrawn_shapes_model_inference:
    """
    This is a class that takes in a trained  hand-drawn flowchart shapes classifier
    and tests the model either on a single image or on a test-set and generates
    model metrics

    Attributes:
        model_path (str): The path to the hand-drawn flowchart shapes classifier
    """
    def __init__(self, model_path):
        """
        The constructor for handdrawn_shapes_model_inference class.
  
        Parameters:
            model_path (str): The path to the hand-drawn flowchart shapes classifier
        """
        self.model_path = model_path
        #possible classifier outputs
        self.classes = {"circle": 1,"rectangle": 2, "diamond": 3, "triangle" : 4}
        self.classes_list = ["circle", "rectangle", "diamond", "triangle"]

    def setup_HoG(self):
        """
        Returns a HoG descriptor extractor that will be used for feature extraction
  
        Returns:
            hog: HoG descriptor extractor
        """
        winSize = (10,10)
        blockSize = (10,10)
        blockStride = (5,5)
        cellSize = (10,10)
        nbins = 9
        derivAperture = 1
        winSigma = -1.
        histogramNormType = 0
        L2HysThreshold = 0.2
        gammaCorrection = 1
        nlevels = 64
        signedGradients = True
        hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,
                winSigma,histogramNormType,L2HysThreshold, gammaCorrection,nlevels, signedGradients)
        return hog

    def extract_feature_vector(self, img_path, hog_computer):
        """
        Takes in an image of a flowchart shape and returns a feature vector
        of all the relevant shapes features that will be used for model training
        
        Parameters:
            img_path (str): The path to the hand-drawn flowchart shapes classifier
            hog_computer: HoG descriptor extractor

        Returns:
            feature_vector: The feature vector for a hand-drawn flowchart shape
                            and this will be used for model training 
        """
        img_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img_gray_resized = cv2.resize(img_gray, (250,200))
        contours, _  = cv2.findContours(img_gray_resized, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        #sort the contours by area
        contours = sorted(contours, key=cv2.contourArea)
        #if there are any contours, continue
        if len(contours)>0:
            #Find are contour / area of minimum enclosing circle
            (x,y), radius = cv2.minEnclosingCircle(contours[-1])
            area_min_enlosing_circle = 3.14 * radius * radius 
            contour_area = cv2.contourArea(contours[-1])
            enclosed_circle_ratio = contour_area/area_min_enlosing_circle
        else:
            raise ValueError("testing data doesnt have contour")
        feature_vector = hog_computer.compute(img_gray_resized)
        np.append(feature_vector, enclosed_circle_ratio) 
        return feature_vector

    def generate_metrics(self, y_pred, y_true):
        """
        Takes in model prediction and the ground truths of a test set and generates
        model metrics
        
        Parameters:
            y_pred: array containing the model predictions on the entire test set
            y_true: array containing the ground truth classifications of the entire test set

        Returns:
            cm: confusion matrix 
            cr: classification report 
        """
        labels = ["circle","rectangle","diamond","triangle"]
        cm = confusion_matrix(y_true, y_pred)
        cr = classification_report(y_true, y_pred)
        print(cm)
        print(cr)
        cmd = ConfusionMatrixDisplay(cm, display_labels=labels)
        cmd.plot()
        plt.show()
        return cm, cr

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

    def test_entire_dataset(self, test_dataset_path):
        """
        This entry point function takes in the path to a directory containing
        test images of hand-drawn shapes and uses the other utility methods 
        in this class to generate model metrics

        Parameters:
            test_dataset_path (str): The path to the directory containing all 
                                     hand-drawn shapes test images
        
        Returns:
            cm: confusion matrix 
            cr: classification report 
        """
        #load model
        loaded_model = pickle.load(open(self.model_path, 'rb'))
        print("model loaded")
        #setup HoG extractor
        y_true = []
        y_pred = []
        hog_computer = self.setup_HoG()
        for shape, idx in tqdm(self.classes.items()):
            class_folder = os.path.join(test_dataset_path, shape)
            label = idx
            for img_name in tqdm(os.listdir(class_folder)):
                img_path = os.path.join(class_folder, img_name)
                feature_vector = self.extract_feature_vector(img_path, hog_computer) 
                Ypredict = loaded_model.predict([feature_vector])
                y_pred.append(Ypredict)
                y_true.append(label)
        cm, cr = self.generate_metrics(y_pred, y_true)
        return cm, cr
    
    def test_single_image(self, img_gray):
        """
        This entry point function takes in the path to a single test image
        of a hand-drawn shape and uses the other utility methods in this class
        to visualize the model output

        Parameters:
            img_gray (str): The path to the image file
        
        Returns:
            shapes_info (dict): dictionary containing all the shapes and their information 
        """
        img_color = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
        #load model
        loaded_model = pickle.load(open(self.model_path, 'rb'))
        print("model loaded")
        #setup HoG extractor
        hog_computer = self.setup_HoG()

        #find all your connected components (white blobs in your image)
        num_components, labels, stats, centroids = cv2.connectedComponentsWithStats(img_gray, connectivity=8)
        #remove background from connected component lists
        stats_filtered = stats[1:]
        centroids_filtered = centroids[1:]
        labels_filtered = labels[1:]
        num_components_filtered = num_components - 1
        shapes_info = []
        #create list to save connected components shape
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
            height_pad = int(0.25*bb_height)
            width_pad = int(0.25*bb_width)
            ROI = img_gray[bb_y:bb_y+bb_height, bb_x:bb_width+bb_x]
            ROI = cv2.copyMakeBorder(
                 ROI, 
                 height_pad, 
                 height_pad, 
                 width_pad, 
                 width_pad, 
                 cv2.BORDER_CONSTANT, 
                 value=(0,0,0)
              )
            ROI = cv2.resize(ROI, (250, 200))
            contours, _  = cv2.findContours(ROI, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            #sort the contours by area
            contours = sorted(contours, key=cv2.contourArea)
            #if there are any contours, continue
            if len(contours)>0:
                (x,y), radius = cv2.minEnclosingCircle(contours[-1])
                area_min_enlosing_circle = 3.14 * radius * radius 
                contour_area = cv2.contourArea(contours[-1])
                enclosed_circle_ratio = contour_area/area_min_enlosing_circle
            else:
                raise ValueError("training data doesnt have contour")

            hog_features = hog_computer.compute(ROI)
            #finalize feature vector
            np.append(hog_features, enclosed_circle_ratio) 
            Ypredict = loaded_model.predict([hog_features])
            shape = self.classes_list[Ypredict[0]-1]
            shape_info = {
                "shape_number": i,
                "shape": shape,
                "centroid": (centroid_x, centroid_y),
                "arrow_to": [] 
            }
            shapes_info.append(shape_info)
            cv2.putText(img_color, str(shape), (centroid_x, centroid_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        self.plot_images([img_color], False, 5, 1, 1, "img")
        return shapes_info
