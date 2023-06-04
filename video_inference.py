import cv2
import torch
from model import model_utils
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import draw_bounding_boxes

video_path = 'demo_vid\pexels-tom-fisk-2955576-3840x2160-30fps (1).mp4'
device = torch.device('cpu')
model = model_utils.getDroneObjectDetectionInstance()
model = model_utils.load_model_state('model\model_file.pt',device)

vid = cv2.VideoCapture(video_path)

labels =   {
        1:'person',
        2:'tree',
        3:'car',
        4:'motorcycle'
    }

for ii in range(40):
    ret, frame = vid.read()
    orig_size = frame.shape
    frame = cv2.resize(frame, (512, 512))
    frame = torch.from_numpy(frame)
    frame = frame.permute(2, 0, 1)

    
    with torch.no_grad():
        prediction = model([(frame/255).to(device)])
        
    predictions = []
    for i in prediction[0]['labels'].tolist():
        predictions.append(labels[i])

    scores = prediction[0]['scores'].tolist()
    idx = next(x for x, val in enumerate(scores) if val < 0.1)
    output = draw_bounding_boxes(frame.to(torch.uint8), prediction[0]['boxes'][:idx], predictions[:idx])
    out = torchvision.transforms.ToPILImage()(output) 
    out = cv2.resize(np.array(out), (orig_size[1]-100, orig_size[0]-100))
    cv2.imshow('', out)
    cv2.waitKey(1)