import argparse
import matplotlib.pyplot as plt
from Digital_Flowchart_Generator import *


def launch_training(dataset_path, model_name, model_path):
    """
    Train a SVM model to classify hand-drawn flowchart shapes

    Parameters: 
        test_dataset_path: Path to the training set 
        model_name: Name given to the trained model
        model_path: path where model will be saved
    """
    training_tools = train_shapes_classifier(dataset_path)
    data = training_tools.setup_data()
    classifier, scores = training_tools.train_classifier(data, model_name, model_path, rounds=10, debug=False)
    training_tools.summary(scores)

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Arguments needed to start training a SVM classifier\
                                                    to classify hand-drawn flowchart shapes')
    parser.add_argument('-d', '--data_path', required=True, type=str, help='Path to the training set')
    parser.add_argument('-m', '--model_name', required=True, type=str, help='Name given to the trained model')
    parser.add_argument('-p', '--model_path', required=True, type=str, help='path where model will be saved')
    args = parser.parse_args()

    dataset_path = str(args.data_path)    
    model_name = str(args.model_name)  
    model_path = str(args.model_path)    
    launch_training(dataset_path, model_name, model_path)
