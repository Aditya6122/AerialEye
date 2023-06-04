import torch
from PIL import Image
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import PILToTensor
from torchvision.transforms import ToPILImage
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn
from torchvision.utils import draw_bounding_boxes


def getDroneObjectDetectionInstance():
    model = fasterrcnn_mobilenet_v3_large_fpn(weights='DEFAULT')
    num_classes=5
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features,num_classes)
    return model

def load_model_state(model_path,device):
    model = getDroneObjectDetectionInstance()
    model.load_state_dict(torch.load(model_path,map_location = device))
    return model

def get_inference(img_path,threshold,model,device):
    img = Image.open(img_path).convert("RGB")
    img = PILToTensor()(img)

    model.eval()
    with torch.no_grad():
        prediction = model([(img/255).to(device)])
        
    labels =   {
            1:'person',
            2:'tree',
            3:'car',
            4:'motorcycle'
        }

    predictions = []
    for i in prediction[0]['labels'].tolist():
        predictions.append(labels[i])

    scores = prediction[0]['scores'].tolist()
    idx = next(x for x, val in enumerate(scores) if val < threshold)
    output = draw_bounding_boxes(img.to(torch.uint8), prediction[0]['boxes'][:idx], predictions[:idx])
    out = ToPILImage()(output) 
    return out