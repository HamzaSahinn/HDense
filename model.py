import torch.nn as nn
import torchvision
import torch

class HDense_LSTM(nn.Module):
  def __init__(self):
    super().__init__()

    densenet = torchvision.models.densenet201(pretrained=True)

    self.dense_1 = densenet.features[0:5]
    self.transition_1 = densenet.features[5] #192
    self.dense_2 = densenet.features.denseblock2
    self.transition_2 = densenet.features.transition2 #384
    self.dense_3 = densenet.features.denseblock3
    self.transition_3 = densenet.features.transition3 #1056
    self.dense_4 = densenet.features.denseblock4

    
    self.down1 = nn.MaxPool2d(4,4)
    self.down2 = nn.MaxPool2d(2,2)
    
    self.conv = nn.Sequential(nn.BatchNorm2d(3200),nn.ReLU(),nn.Conv2d(3200,512,1,1,0))
    
    self.last_norm = nn.BatchNorm2d(512)

    self.last_pool = nn.AdaptiveAvgPool2d((1,1))
    self.rnn = nn.LSTM(512,128,1, batch_first=True)
    self.classifier = nn.Sequential(nn.Linear(1280,25))

    del densenet
    
  def forward(self, x):
    seq_list = []
   
    for i in range(x.shape[2]):
      inp = x[:,:,i].squeeze(2)
      feature_1 = self.transition_1(self.dense_1(inp))
      feature_2 = self.transition_2(self.dense_2(feature_1))
      feature_3 = self.transition_3(self.dense_3(feature_2))
      feature_4 = self.dense_4(feature_3)

      feature_1_downed = self.down1(feature_1)
      feature_2_downed = self.down2(feature_2)
    
      concated_features = torch.cat((feature_1_downed, feature_2_downed, feature_3, feature_4),1)

      features = self.last_pool(self.last_norm(self.conv(concated_features)))
      features = torch.flatten(features,1)
      seq_list.append(features)
    stack_inp = torch.stack(seq_list, 1)
    features= self.rnn(stack_inp)[0]
    

    features = torch.flatten(features,1)
    return self.classifier(features)


class HDense(nn.Module):
  def __init__(self):
    super().__init__()

    densenet = torchvision.models.densenet201(pretrained=True)

    self.dense_1 = densenet.features[0:5]
    self.transition_1 = densenet.features[5] #192
    self.dense_2 = densenet.features.denseblock2
    self.transition_2 = densenet.features.transition2 #384
    self.dense_3 = densenet.features.denseblock3
    self.transition_3 = densenet.features.transition3 #1056
    self.dense_4 = densenet.features.denseblock4

    
    self.down1 = nn.MaxPool2d(4,4)
    self.down2 = nn.MaxPool2d(2,2)
    
    self.conv = nn.Sequential(nn.BatchNorm2d(3200),nn.ReLU(),nn.Conv2d(3200,512,1,1,0))
    
    self.last_norm = nn.BatchNorm2d(512)

    self.last_pool = nn.AdaptiveAvgPool2d((1,1))
    self.classifier = nn.Sequential(nn.Linear(512,25))

    del densenet
    
  def forward(self, x):
  
    feature_1 = self.transition_1(self.dense_1(x))
    feature_2 = self.transition_2(self.dense_2(feature_1))
    feature_3 = self.transition_3(self.dense_3(feature_2))
    feature_4 = self.dense_4(feature_3)

    feature_1_downed = self.down1(feature_1)
    feature_2_downed = self.down2(feature_2)
    
    concated_features = torch.cat((feature_1_downed, feature_2_downed, feature_3, feature_4),1)

    features = self.last_pool(self.last_norm(self.conv(concated_features)))
    features = torch.flatten(features,1)

    return self.classifier(features)