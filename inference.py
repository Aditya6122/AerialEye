import torch
from model import model_utils
import sys

img_path = sys.argv[1]
model_path = sys.argv[2]
threshold = float(sys.argv[3])

device = torch.device('cpu') # if torch.cuda.is_available() else torch.device('cpu')
model = model_utils.getDroneObjectDetectionInstance()
model = model_utils.load_model_state(model_path,device)
predicted_output = model_utils.get_inference(img_path,threshold,model,device)
predicted_output.show()