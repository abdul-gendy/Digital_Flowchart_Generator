import os
import cv2
import argparse
import matplotlib.pyplot as plt
from Digital_Flowchart_Generator import *


def convert_to_digital_flowchart(img_path, shape_classifier_model):    
    """
   convert an image of a hand-drawn flowchart to a digital version

    Parameters: 
        img_path: The path to the image of a hand-drawn flowchart
        shape_classifier_model: path to the shape classifier that will be used
    """
    pre_processor = decompose_flowchart(img_path)
    img_shapes_only, img_arrows_only = pre_processor.start_decomposition()

    shapes_finder = handdrawn_shapes_model_inference(shape_classifier_model)
    shapes_info = shapes_finder.test_single_image(img_shapes_only)

    arrows_finder = arrows_classifier(img_arrows_only)
    arrows_info = arrows_finder.find_arrows()

    reconstructor = reconstruct_flowchart(arrows_info, shapes_info, img_path)
    shapes_info = reconstructor.establish_connections_between_shapes()

    graphics = flowchart_graphics_simple(shapes_info)
    graphics.draw_flowchart()
    plt.show()


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Arguments needed to convert hand-drawn flowchart to digital version')
    parser.add_argument('-m', '--model_path', required=True, type=str, help='path to the model that will be used for shape classification')
    parser.add_argument('-i', '--img_path', required=True, type=str, help='The path to the image of a hand-drawn flowchart')
    args = parser.parse_args()

    img_path = str(args.img_path) 
    model_path = str(args.model_path)  
    convert_to_digital_flowchart(img_path, model_path)  
