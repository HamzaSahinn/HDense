import sys

import kornia
from model import HDense, HDense_LSTM
from data import ERA_Test_Image_Loader, ERA_Train_Image_Loader, ERA_Test_Video_Loader, ERA_Train_Video_Loader
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
import torchmetrics
import torchsummary
from tqdm import tqdm


class_criterion = torch.nn.CrossEntropyLoss()

train_class_metric = torchmetrics.Accuracy()
train_class_metric.reset()


val_class_criterion = torch.nn.CrossEntropyLoss()

val_class_metric = torchmetrics.Accuracy()
val_class_metric.reset()



epoch = 100
model = HDense_LSTM()
swa_model = torch.optim.swa_utils.AveragedModel(model)
model.load_state_dict(torch.load("weights\DenseLargeLSTM-SWA-ERA__10-1280_64.99", map_location="cpu"))

if torch.cuda.is_available():
    model = model.cuda()
    train_class_metric = train_class_metric.cuda()
    val_class_metric = val_class_metric.cuda()
    class_criterion = class_criterion.cuda()

optimizer = optim.SGD(model.parameters(), lr=0.01)

data_loader_train = DataLoader(ERA_Train_Image_Loader("./train_labels.txt"), batch_size=16, shuffle=True, pin_memory=True, num_workers = 4, drop_last=True)
data_loader_val = DataLoader(ERA_Test_Image_Loader("./val_labels.txt"), batch_size=8, shuffle=True, pin_memory=True, num_workers=4)

writer = SummaryWriter('runs/HDense_trial_1')


def Calculate_Val_Dataset(data_loader, epoch):
    model.eval()
    with torch.no_grad():
        val_class_metric.reset()
        
        val_class_loss = 0
        
        for inputs, class_labels in data_loader:
            if torch.cuda.is_available():
                inputs, labels = inputs.cuda(), class_labels.cuda()
            
            class_output = model(inputs)
            
            val_class_loss += val_class_criterion(class_output, labels)
            
            val_class_metric(class_output, labels)
            
            del inputs; del labels; del class_output
            torch.cuda.empty_cache()
            
        writer.add_scalar('val_class_loss', val_class_loss.item()/len(data_loader), epoch)
        writer.add_scalar('val_class_metric', val_class_metric.compute(), epoch)
        


if __name__ == "__main__":
    #torchsummary.summary(model, (3,32,448,448))

    for epoch in range(0, epoch):
        model.train()
        with tqdm(data_loader_train, unit="batch") as tepoch:
            index = 0
            for inputs, labels in tepoch:
                tepoch.set_description(f"Epoch {epoch}")
                
                if torch.cuda.is_available():
                    inputs, labels = inputs.cuda(),  labels.cuda()
                
                optimizer.zero_grad(set_to_none=True)

                class_outputs = model(inputs)
                
                loss = class_criterion(class_outputs, labels)
                
                loss.backward()
                optimizer.step()
                
                swa_model.update_parameters(model)
                
                train_class_metric(class_outputs, labels)
                
                del inputs; del labels
                
                tepoch.set_postfix(loss=loss.item())
                index += 1
                writer.add_scalar('training loss', loss.item(), epoch * len(data_loader_train) + index)
                
        torch.save(model.state_dict(), "HDense"+str(epoch+1))
        writer.add_scalar('train_class_acc', train_class_metric.compute(), epoch)

        train_class_metric.reset()
        Calculate_Val_Dataset(data_loader_val, epoch)
    
    torch.optim.swa_utils.update_bn(data_loader_train, swa_model)
    torch.save(swa_model.state_dict(), "HDense_swa"+str(epoch+1))

    print('Finished Training')