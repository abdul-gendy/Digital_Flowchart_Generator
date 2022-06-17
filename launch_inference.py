import argparse
import matplotlib.pyplot as plt
from Digital_Flowchart_Generator import *


def launch_inference(test_dataset_path, shape_classifier_model):
    """
    run inference on the hand-drawn flowchart shapes classifier

    Parameters: 
        test_dataset_path: The path to the test set 
        shape_classifier_model: path to the shape classifier that will be used
    """
    shapes_finder = handdrawn_shapes_model_inference(shape_classifier_model)
    cm, cr = shapes_finder.test_entire_dataset(test_dataset_path)
    plt.show()

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Arguments needed to run inference on the hand-drawn \
                                                 flowchart shapes classifier')
    parser.add_argument('-m', '--model_path', required=True, type=str, help='path to the model that will be used')
    parser.add_argument('-d', '--dataset_path', required=True, type=str, help='The path to the test set')
    args = parser.parse_args()

    dataset_path = str(args.dataset_path)    
    model_path = str(args.model_path)  
    launch_inference(dataset_path, model_path)
