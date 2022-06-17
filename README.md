# Digital_Flowchart_Generator
The aim of this project is to allow the user to quickly draw a flowchart template made up of shapes and arrows in their required order on a piece of paper and then
convert an image of that into a digital template that the user can then start working on right away. This application could be useful in many settings, especially in brainstorming sessions where individuals don’t want to spend a lot of time creating a flowchart template on a software tool.

This repo allows you to do the following:
  - Directly convert a picture of a hand-drawn flowchart to a digital version using one of the pretrained model saved on the repo
  - Train a new hand-drawn flowchart shapes classifier 
  - Test a trained hand-drawn flowchart shapes classifier


### Setup
Install all required dependencies in the requirements.txt file.
```
pip install -r requirements.txt
```

### Usage
##### converting a picture of a hand-drawn flowchart to digital
```
python convert_to_digital_flowchart.py -m MODEL_PATH -i IMAGE_PATH
```
##### Conversion Sample Output
```
python convert_to_digital_flowchart.py -m ".\Digital_Flowchart_Generator\classifier\models\flowchart_shape_classifier_v2.sav" -i ".\sample_inputs\sample.jpg"
```
<img src="Documentation\pictures\conversion.PNG" alt="alt text" width="450" height="350" class="center">


##### Training a hand-drawn flowchart shapes classifier
To train a SVM model that classifies hand-drawn shapes, you will need access to a dataset of hand-drawn shapes. The dataset should follow the directory structure of "sample_shapes_dataset" in this repo. This directory also contains some sample images from the dataset. Instructions on how to create your own dataset for this application is discussed in the video and paper found in the link below. 

  - https://abdulelgendy.com/portfolio/flowchart-generator/
```
python launch_training.py -d TRAINING_DATA_DIRECTORY_PATH -m MODEL_NAME -p MODEL_SAVE_PATH
```
##### sample
```
python launch_training.py -d ".\sample_shapes_dataset\training_set" -m "new_model" -p ".\Digital_Flowchart_Generator\classifier\models"
```


##### Testing a hand-drawn flowchart shapes classifier
To test and generate metrics for a trained hand-drawn flowchart shapes classifier, run the following command: 
```
python launch_inference.py -m MODEL_PATH  -d TEST_DATA_DIRECTORY_PATH
```
##### Testing Sample Output
```
python launch_inference.py -m ".\Digital_Flowchart_Generator\classifier\models\flowchart_shape_classifier_v2.sav"  -d ".\Digital_Flowchart_Generator\sample_shapes_dataset\test_set"
```
<img src="Documentation\pictures\confusion.PNG" alt="alt text" width="500" height="500">
