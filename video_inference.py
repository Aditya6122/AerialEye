import cv2
import torch
from model import model_utils
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import to_tensor

video_path = 'demo_vid\\video_preview_h264.mp4'
device = torch.device('cpu')
model = model_utils.getDroneObjectDetectionInstance()
model = model_utils.load_model_state(
    'model\model_mobilenet_large_fpn.pt', device)

class_labels = {
    1: 'person',
    2: 'tree',
    3: 'car',
    4: 'motorcycle'
}

video = cv2.VideoCapture(video_path)
frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = video.get(cv2.CAP_PROP_FPS)
output_video = cv2.VideoWriter('output\output_video.mp4', cv2.VideoWriter_fourcc(
    *'mp4v'), 30, (frame_width, frame_height))

cv2.namedWindow("output", cv2.WINDOW_NORMAL) 
cv2.resizeWindow('output',960,540)

model.eval()
with torch.no_grad():
    while(True):
        ret, frame = video.read()
        if not ret:
            break

        resized_frame = cv2.resize(frame, (512, 512))
        input_tensor = to_tensor(resized_frame)
        input_tensor.to(device)
        prediction = model([input_tensor])

        boxes = prediction[0]['boxes'].numpy()
        labels = prediction[0]['labels'].numpy()
        scores = prediction[0]['scores'].numpy()

        final_boxes = []
        final_labels = []
        final_scores = []
        for i in range(len(labels)):
            if(scores[i] > 0.7):
                xmin, ymin, xmax, ymax = boxes[i]
                xmin = int(xmin * frame_width / 512)
                ymin = int(ymin * frame_height / 512)
                xmax = int(xmax * frame_width / 512)
                ymax = int(ymax * frame_height / 512)
                final_boxes.append((xmin, ymin, xmax, ymax))
                final_scores.append(scores[i])
                final_labels.append(labels[i])
        
        for box, label, score in zip(final_boxes, final_labels, final_scores):
            xmin, ymin, xmax, ymax = box
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255, 0, 0), 1)
            cv2.putText(frame, f'{class_labels[label]}: {score:.2f}', (xmin, ymin - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)
        
        # cv2.imshow('output', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()