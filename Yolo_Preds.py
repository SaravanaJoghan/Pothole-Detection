#!/usr/bin/env python
# coding: utf-8

import yaml
import cv2 
import numpy as np
import os
from yaml.loader import SafeLoader

class YOLO_Preds():
    def __init__(self,onnx_model, data_yaml):
        #load YAML
        with open('./models/data.yaml', mode = 'r' ) as f:
            data_yaml = yaml.load(f, Loader = SafeLoader)

        self.labels = data_yaml["names"]
        self.nc = data_yaml["nc"]
         #load yolo model
        self.yolo = cv2.dnn.readNetFromONNX('./models/150epochs.onnx')
        # yolo = cv2.dnn.readNetFromONNX('./Model/MyModel14/weights/best.onnx')
        self.yolo.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.yolo.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    
    def Predictions(self, image):
        row,col,d = image.shape

        #convert image to square array
        max_rc = max(row,col)
        input_image = np.zeros((max_rc,max_rc,3), dtype = np.uint8)
        input_image[0:row, 0:col] = image

        #get prediction in square array
        inp_wh = 640
        blob = cv2.dnn.blobFromImage(input_image, 1/255, (inp_wh,inp_wh), swapRB = True, crop = False)
        self.yolo.setInput(blob)
        preds = self.yolo.forward() #prediction from yolo

        #Non maximum supression
        #filter detections base on confidence scores (0.4) and probabilitty score (0.25)
        detections = preds[0]
        boxes = []
        confidences = []
        classes = []

        #width and height of the image
        image_w, image_h = input_image.shape[:2]
        x_factor = image_w/inp_wh
        y_factor = image_w/inp_wh

        for i in range(len(detections)):
            row = detections[i]
            confidence = row[4]
            if confidence > 0.4:
                class_score = row[5:].max()    #take the maximum probability from 8 class
                class_id = row[5:].argmax()    # index position of the maximum probablity

                if class_score > 0.25:
                    cx, cy, w, h = row[0:4]
                    # construct bounding box
                    # left, top, width and height
                    left = int((cx - 0.5*w) * x_factor)
                    top = int((cy - 0.5*h) * y_factor)
                    width = int(w * x_factor)
                    height = int(h * y_factor)

                    box = np.array([left,top,width,height])
                    #append all values
                    confidences.append(confidence)
                    boxes.append(box)
                    classes.append(class_id)

        # Get the best objects
        boxes_np = np.array(boxes).tolist()
        confidences_np = np.array(confidences).tolist()
        index  = np.array(cv2.dnn.NMSBoxes(boxes_np, confidences_np, 0.25, 0.45)).flatten()

        #Draw the bounding box

        for i in index:
            x,y,w,h = boxes_np[i]
            bbox_confidence = int(confidences_np[i] * 100)
            classes_id = classes[i]
            class_name = self.labels[classes_id]
            #generate colors
            colors = self.generate_colors(classes_id)
            #text to display
            text = f"{class_name}: {bbox_confidence}%"
            print(text)

            #Draw rectangle
            cv2.rectangle(image, (x,y), (x+w, y+h), colors, 2)
            cv2.rectangle(image, (x, y-30), (x+w, y), colors, -1)
            cv2.putText(image, text, (x,y-10), cv2.FONT_HERSHEY_PLAIN, 0.7, (255,255,255), 1)
        
        return image
    
    def generate_colors(self,ID):
        np.random.seed(10)
        colors = np.random.randint(100,255,size = (self.nc, 3)).tolist()
        return tuple(colors[ID])
        
        




