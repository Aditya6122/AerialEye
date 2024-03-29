import torch
from model import model_utils
import sys

img_path = sys.argv[1]
model_path = sys.argv[2]
threshold = float(sys.argv[3])

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = model_utils.getDroneObjectDetectionInstance()
model = model_utils.load_model_state(model_path,device)
time_taken,predicted_output = model_utils.get_inference(img_path,threshold,model,device)
print(time_taken)
predicted_output.show() 