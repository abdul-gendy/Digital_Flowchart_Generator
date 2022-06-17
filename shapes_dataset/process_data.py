import cv2, numpy as np, os
import glob
from tqdm import tqdm





#chromakey values for green
h,s,v,h1,s1,v1 = 163,2,150,179,255,255
#h,s,v,h1,s1,v1 = 163,2,20,179,255,255



#takes image and range returns parts of image in range
def only_color(self, img, h,s,v,h1,s1,v1, pad):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower, upper = np.array([h,s,v]), np.array([h1,s1,v1])
    mask = cv2.inRange(hsv, lower, upper)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))  
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask

#paths to source video
data_paths = [r"C:\Users\Dell\Desktop\5th year\4TN4\Project\hand_written_flowchart_to_digital\classifier\shapes_dataset\test_set\videos\t3.mp4"]
#paths to folders where training data will be stored
folder_names = [r'C:\Users\Dell\Desktop\5th year\4TN4\Project\hand_written_flowchart_to_digital\classifier\shapes_dataset\test_set\circle\ttt']

'''
for data_path, folder_name in zip(data_paths, folder_names):
    frame_num = 0
    cap = cv2.VideoCapture(data_path)
    while True:
        _, img= cap.read()
        try:
            height, width, _ = img.shape
        except:
            print("an error occured")
            break
        #get a mask of the image to remove the green background
        mask = only_color(img,h,s,v,h1,s1,v1, pad)
        #find the contours in the image
        contours, _  = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        #sort the contours by area
        contours = sorted(contours, key=cv2.contourArea)
        #if there are any contours, continue
        if len(contours)>0:
            x,y,w,h = cv2.boundingRect(contours[-1])
            height_pad = int(0.25*h)
            width_pad = int(0.25*w)
            ROI = mask[y:y+h, x:x+w]
            ROI = cv2.copyMakeBorder(
                 ROI, 
                 height_pad, 
                 height_pad, 
                 width_pad, 
                 width_pad, 
                 cv2.BORDER_CONSTANT, 
                 value=(0,0,0))
            print(ROI.shape)
            #if the shape is not too long and thin, continue
            if np.prod(ROI.shape)!=0:
                #crop = cv2.resize(crop, (250, 200))
                #crop = 255-crop
                cv2.imshow('img', ROI)
                cv2.imwrite(folder_name+str(frame_num)+'.png', ROI)
                frame_num += 1
        cv2.imshow('img1', cv2.resize(mask, (640,480)))
        cv2.waitKey(10)
        if frame_num%250==0: print (frame_num,'----------------------------')               
    cap.release()
cv2.destroyAllWindows()
'''
'''
for data_path, folder_name in zip(data_paths, folder_names):
    frame_num = 0
    cap = cv2.VideoCapture(data_path)
    while True:
        _, img= cap.read()
        try:
            height, width, _ = img.shape
        except:
            print("finished going through video")
            break
        #get a mask of the image to remove the green background
        mask = only_color(img,h,s,v,h1,s1,v1, pad)
        contours, _  = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        #sort the contours by area
        contours = sorted(contours, key=cv2.contourArea)
        #if there are any contours, continue
        if len(contours)>0:
            print("countour found")
            x,y,w,h = cv2.boundingRect(contours[-1])
            height_pad = int(0.25*h)
            width_pad = int(0.25*w)
            #ROI = mask[y:y+h, x:x+w]
            #print("ROI found")
            #cv2.imshow('img', cv2.resize(mask, (640,480)))
        cv2.imshow('img1', cv2.resize(mask, (640,480)))
        cv2.waitKey(30)
    cap.release()
cv2.destroyAllWindows()
'''
image_files = glob.glob(r"C:\Users\Dell\Desktop\5th year\4TN4\Project\shapes_dataset\test_set\rectangle\*")
for image_path in tqdm(image_files):
    frame = cv2.imread(image_path)
    #convert to Gray
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    contours, _  = cv2.findContours(frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #sort the contours by area
    contours = sorted(contours, key=cv2.contourArea)
    #if there are any contours, continue
    if len(contours)>0:
        x,y,w,h = cv2.boundingRect(contours[-1])
        height_pad = int(0.25*h)
        width_pad = int(0.25*w)
        ROI = frame[y:y+h, x:x+w]
        ROI = cv2.copyMakeBorder(
                ROI, 
                height_pad, 
                height_pad, 
                width_pad, 
                width_pad, 
                cv2.BORDER_CONSTANT, 
                value=(0,0,0)
              )
    #frame = cv2.bitwise_not(frame)
    cv2.imwrite(image_path, ROI)
