from matplotlib import image
from torch.utils.data import Dataset
import torchvision
from pytorchvideo.transforms import ShortSideScale
from torchvision.transforms import Compose, Lambda, Resize, Normalize, AutoAugment
from pytorchvideo.transforms.functional import uniform_temporal_subsample
import os
from torchvision.transforms._transforms_video import NormalizeVideo


def create_labels_text_file(dir, label_file_path):
  class_index = 0
  class_dict = {}
  label_file = open(label_file_path, "w")

  for activity_name in sorted(os.listdir(dir)):
    class_dict[activity_name] = class_index

    path_root = os.path.join(dir, activity_name)
    videos = os.listdir(path_root)

    for video in videos:
        line = os.path.join(path_root, video) + "," + str(class_index) + "\n"
        label_file.write(line)
    class_index += 1
  label_file.close()
#create_labels_text_file("data\\SingleFrames\\Tra", "Image_Train_Labels.txt")
#create_labels_text_file("data\\Videos\\Tra", "Video_Train_Labels.txt")

def _normalize_image(x):
  return x/255.0

def crate_validation_labels_text_file(dir, label_file_path):
  class_index = 0
  class_dict = {}
  label_file = open(label_file_path, "w")

  for activity_name in sorted(os.listdir(dir)):
      class_dict[activity_name] = class_index
      path_root = os.path.join(dir, activity_name)
      videos = os.listdir(path_root)
      counter = 0
      for video in videos:
          if counter == 12:
              break
          line = os.path.join(path_root, video) + "," + str(class_index) + "\n"
          label_file.write(line)
          counter += 1
      class_index += 1
  label_file.close()
        
def apply_transition_to_frames(video_data):
  video_data[:,10]


class ERA_Train_Video_Loader(Dataset):
  def __init__(self, train_file_path):
    train_data = open(train_file_path, "r")
    self.train_data = list(train_data)
    
    self.train_transform = Compose(
    [Lambda(_normalize_image),
    ShortSideScale(size=224),
    NormalizeVideo((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
  def __len__(self):
    return len(self.train_data)
    
  def __getitem__(self,idx):
    video_path, label = self.train_data[idx].split(",")
    
    video_data = torchvision.io.read_video(video_path, start_pts=0, end_pts=float('inf'), pts_unit="sec")[0].permute(3,0,1,2)
    video_data = uniform_temporal_subsample(video_data, num_samples=10, temporal_dim=1)
    video_data = self.train_transform(video_data)
    
    return video_data, int(label)
  
  
class ERA_Test_Video_Loader(Dataset):
  def __init__(self, test_file_path):
    test_data = open(test_file_path, "r")
    self.test_data = list(test_data)
    
    self.train_transform = Compose(
    [Lambda(_normalize_image),
    ShortSideScale(size=224),
    NormalizeVideo((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
  def __len__(self):
    return len(self.test_data)
    
  def __getitem__(self,idx):
    video_path, label = self.test_data[idx].split(",")
    
    video_data = torchvision.io.read_video(video_path, start_pts=0, end_pts=float('inf'), pts_unit="sec")[0].permute(3,0,1,2)
    video_data = uniform_temporal_subsample(video_data, num_samples=10, temporal_dim=1)
    video_data = self.train_transform(video_data)
    
    return video_data, int(label)

 
class ERA_Train_Image_Loader:
  def __init__(self, train_file_path):
      train_data = open(train_file_path, "r")
      self.train_data = list(train_data)

      self.transform = Compose(
      [
        AutoAugment(),
        Lambda(_normalize_image),
        Resize((224,224)),
        Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
      ])

  def __len__(self):
      return len(self.train_data)

  def __getitem__(self, idx): 
      image_path, label = self.train_data[idx].split(",")
      
      image_data = torchvision.io.read_image(image_path)
      image_data = self.transform(image_data)

      return image_data, int(label)
      
         
class ERA_Test_Image_Loader:
    def __init__(self, test_file_path):
        test_data = open(test_file_path, "r")
        self.test_data = list(test_data)

        self.transform = Compose(
        [
         Lambda(_normalize_image),
         Resize((224,224)),
         Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    def __len__(self):
        return len(self.test_data)
  
    def __getitem__(self, idx): 
        image_path, label = self.test_data[idx].split(",")
        
        image_data = torchvision.io.read_image(image_path)
        image_data = self.transform(image_data)

        return image_data, int(label)
      
    
      

