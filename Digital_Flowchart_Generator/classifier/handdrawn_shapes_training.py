import os
import cv2
import numpy as np
import pickle
from sklearn import metrics
from sklearn.svm import SVC as SVM
from sklearn.model_selection import train_test_split
from tqdm import tqdm


class train_shapes_classifier:
    """
    This is a class that trains a hand-drawn shapes classifier

    Attributes:
        dataset_path (str): The path to the hand-drawn flowchart shapes training set
    """
    def __init__(self, dataset_path):
        """
        The constructor for train_shapes_classifier class.
  
        Parameters:
            dataset_path (str): The path to the hand-drawn flowchart shapes training set
        """
        self.dataset_path = dataset_path
        self.classes = {"circle":1, "rectangle":2, "diamond":3, "triangle":4}
   
    def setup_HoG(self):
        """
        Returns a HoG descriptor extractor that will be used for feature extraction
  
        Returns:
            hog: HoG descriptor extractor
        """
        #Hog arguments
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
            (x,y), radius = cv2.minEnclosingCircle(contours[-1])
            area_min_enlosing_circle = 3.14 * radius * radius 
            contour_area = cv2.contourArea(contours[-1])
            enclosed_circle_ratio = contour_area/area_min_enlosing_circle
        else:
            raise ValueError("testing data doesnt have contour")

        feature_vector = hog_computer.compute(img_gray_resized)
        np.append(feature_vector, enclosed_circle_ratio) 
        return feature_vector

    def setup_data(self):
        """
        Extracts the feature vectors from all images in the training set
        and assigns labels to each image in the training set

        Returns:
            data: list containing the training data and their respective labels
        """
        data = []
        hog_computer = self.setup_HoG()
        for shape, idx in self.classes.items():
            class_folder = os.path.join(self.dataset_path, shape)
            label = idx
            for img_name in os.listdir(class_folder):
                img_path = os.path.join(class_folder, img_name)
                feature_vector = self.extract_feature_vector(img_path, hog_computer) 
                data.append([feature_vector, label])
        print("Dataset setup Done, There contains {} images".format(len(data)))
        return data
    
    def train_classifier(self, data, model_name, model_path, rounds=10, debug=False):
        """
        This entry point function uses the training set to train a 
        SVM model to classify hand-drawn flowchart shapes

        Parameters:
            data: list containing training dataset
            rounds (int): number of training rounds
            debug (bool): bool to display validation set results right after training
            model_name (str): Name given to the trained model
            model_path (str): path where model will be saved  

        Returns:
            classifier: trained SVM model
            scores: training metrics
        """
        features = []
        labels = []
        scores = []
        for feature, label in data:
            features.append(feature)
            labels.append(label)
        
        print("Training Started...")
        for i in tqdm(range(rounds)):
            X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.1, shuffle=True)
            classifier = SVM(kernel="linear")
            #train SVM model
            classifier.fit(X_train, y_train)
            score = classifier.score(X_test, y_test)         
            scores.append(score)
            tqdm.write("Training score is {} ".format(score))

        print("Training done!")
        #extract the predictions of the model
        test_predictions = classifier.predict(X_test)
        #print the classification report
        print(metrics.classification_report(y_test, test_predictions))
        # save the model to disk
        filepath = os.path.join(model_path, model_name + '.sav')
        pickle.dump(classifier, open(filepath, 'wb'))

        if debug:
            for i in range(0,len(X_test)):
                test_prediction = classifier.predict([X_test[i]])
                print("{} predicted as {}".format(y_test[i], test_prediction))
        return classifier, scores

    def summary(self, scores, title="Summary"):
        """
        Display Training metrics summary

        Parameters:
            scores: training metrics
        """
        size = 70
        separator = "-"
        
        print(separator*size)
        print("SUMARY: {}".format(title))
        print(separator*size)
        print("CLASSIF\t\tMEAN\tMEDIAN\tMINV\tMAXV\tSTD")
        print(separator*size)
        
        m = round(np.mean(scores)*100, 2)
        med = round(np.median(scores)*100, 2)
        minv = round(np.min(scores)*100, 2)
        maxv = round(np.max(scores)*100, 2)
        std = round(np.std(scores)*100, 2)    
        print("{:<16}{}\t{}\t{}\t{}\t{}".format("SVM Linear", m, med, minv, maxv, std))
        print(separator*size)
