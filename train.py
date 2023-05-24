import torch
from integration_utils.engine import train_one_epoch, evaluate
from integration_utils.utils import collate_fn
from torch.utils.data import DataLoader
from model import model_utils
from data.dataset import CityDataset
from integration_utils.transforms import get_transform

model1 = model_utils.getDroneObjectDetectionInstance()
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
num_classes = 5

dataset = CityDataset('/content/data/train', get_transform(train=True))
dataset_test = CityDataset('/content/data/test', get_transform(train=False))

data_loader = DataLoader(
    dataset, batch_size=4, shuffle=True, num_workers=4,
    collate_fn= collate_fn)

data_loader_test = DataLoader(
    dataset_test, batch_size=2, shuffle=False, num_workers=4,
    collate_fn= collate_fn)

model1.to(device)

params = [p for p in model1.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=3, gamma=0.1)

num_epochs = 15

for epoch in range(num_epochs):
    train_one_epoch(model1, optimizer, data_loader, device, epoch, print_freq=10)
    lr_scheduler.step()
    evaluate(model1, data_loader_test, device=device)

torch.save(model1.state_dict(), '/content/model_file.pt')