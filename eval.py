import matplotlib.pyplot as plt
from model import HDense, HDense_LSTM
import torchmetrics
import torch
from data import ERA_Val_Loader
from torch.utils.data import DataLoader


model = TemporalNeXt(25, 7)
model.eval()

acc_metric = torchmetrics.Accuracy()
acc_metric.reset()

data_loader_val = DataLoader(ERA_Val_Loader("./test_labels.txt"),batch_size=1, shuffle=False, pin_memory=True, num_workers=2)

if torch.cuda.is_available():
    model = model.cuda()
    acc_metric = acc_metric.cuda()


if __name__ == "__main__":
    model.load_state_dict(torch.load("../input/tnext-363-era-e45e53/TNeXt_363-ERA__epoch45"))
    
    for inputs, labels in data_loader_val:
        if torch.cuda.is_available():
            inputs, labels = inputs.cuda(), labels.cuda()
        
        class_output = model(inputs)
        acc = acc_metric(class_output, labels)
        
        print(acc, "Labels", labels, "  Preds",torch.max(class_output, dim=1)[1])
        
    total_acc = acc_metric.compute()
    
    print("Total accuracy: ", total_acc)
