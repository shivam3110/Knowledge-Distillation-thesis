
"""
-------------------------------------------------------------------------------
1. model_name == 'resnet18':
2. model_name == 'resnet50':
3. model_name == 'teacher_student':
4. model_name == 'student_teacher':
5. model_name = 'bidirectional_kd':     
6. model_name == 'convnext_tiny':
7. model_name == 'efficientnet_v2_s':
8. model_name =='maxvit_t':
9. model_name == 'densenet121':
10 model_name == 'inception_v3':
11 model_name == 'mobilenet_v2':
--------------------------------------------------------------------------------------



python code/kfold_testing_script.py \
 --test_csv "/home/shsingh/knowledge_distillation/dataset/scratch/dataframes/train_test_splits/test_df.csv" \
 --mod "flair"\
 --model_name "mobilenet_v2"\
 --fold 0 \
 --save_dir "flair/baseline/" \
 --pre_trained_weights "True" \
 --class_column_name 'label_flair' \
 --subject_id_column_name 'subject_id' \
 --class_healthy_name 'healthy' \
 --device_num=4
"""

import matplotlib.pyplot as plt
# import nibabel as nib
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd 
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import f1_score, roc_auc_score, roc_curve, auc
import torch 
import csv
import os
import glob
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import models
from tqdm import tqdm
from torchvision import transforms
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from torchvision.transforms import Resize
import numpy as np

import os
import sys
import argparse
import glob
import random
from pathlib import Path

# import imageio 
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import models
from torch.utils.tensorboard import SummaryWriter
# from kfold_evaluation import get_confusion_matrix, calculate_metrics
from models import get_model_by_name
from paths import DATASETS_DIR, MICCAI_BraTS2020_Data, OUTPUT_DIR, DATAFRAME_DIR


np.random.seed(31101995)
torch.manual_seed(31101995)


def parse_args():
    """Parse th earguments and return args"""
    parser = argparse.ArgumentParser(description="Create dataset script.")
    parser.add_argument(
        "--test_csv",
        type=str,
        default=None,
        required=True,
        help="Path to mapping_csv",
    )
    parser.add_argument(
        "--mod",
        type=str,
        default=None,
        required=True,
        help="modularity to be used for classification.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        required=True,
        help="model name:",
    )    
    parser.add_argument(
        "--fold",
        type=int,
        default=None,
        required=True,
        help="model: best fold",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default=None,
        required=True,
        help="Path to saving dir",
    )
    parser.add_argument(
        "--pre_trained_weights",
        type=str,
        default=False,
        required=True,
        help="If to use ImageNet initial weights",
    )
    parser.add_argument(
        "--class_column_name",
        type=str,
        default='label_flair',
        required=True,
        help="class column name in df",
    )
    parser.add_argument(
        "--subject_id_column_name",
        type=str,
        default='subject_id',
        required=True,
        help="subject_id column name in df",
    )
    parser.add_argument(
        "--class_healthy_name",
        type=str,
        default='healthy',
        required=True,
        help="class_healthy? class 'healthy' name in df",
    )
    parser.add_argument(
        "--device_num",
        type=int,
        default=None,
        required=True,
        help="GPU device index",
    )

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    #Sanity Checks
    if args.test_csv is None:
        raise ValueError("Need a mapping-csv to load.")
    if args.model_name is None:
        raise ValueError("Please provide name the model to use.")
    if args.fold is None:
        raise ValueError("Please provide fold of the model to use.")
    if args.save_dir is None:
        raise ValueError("Need a Saving Folder.")
    if args.pre_trained_weights is None:
        raise  ValueError("to use pretrained weights, True/False.")

    if args.class_column_name is None:
        raise  ValueError("required class_column name.")
    if args.subject_id_column_name is None:
        raise  ValueError("required subject_id column name.")
    if args.class_healthy_name is None:
        raise  ValueError("required class_healthy_name  as in the class_column in df.")
    if args.device_num is None:
        raise ValueError("device number(cuda).")
    return args

def class_to_idx(label):
    idx = 0
    if label == 'healthy':idx =0
    if label == 'HGG':idx =1
    if label=='LGG': idx = 2        
    return idx

class CustomDataset_from_csv(Dataset):
    def __init__(self, data_df , mod, transform = None , label_transform= None):
        self.dataframe = data_df #pd.read_csv(annotations_file)
        self.mod = mod
        self.transform = transform
        self.label_transform = label_transform
        
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self , idx):
        img_path =  self.dataframe.loc[idx,'image_path'] 
        img_path_mod = img_path. replace('seg',f'{self.mod}')
        img = np.load(img_path_mod)
        grayscale_image = np.resize(img, (224,224))
        image = np.repeat(grayscale_image[..., np.newaxis], 3, -1)
        # print(np.shape(image))
        # Convert numpy array to torch.Tensor and set the data type to torch.float32
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float()

        label_col = 'label_' + str(self.mod)     
        label_mod = self.dataframe.loc[idx, label_col]
        class_label = class_to_idx(label_mod)
        label = torch.tensor([class_label])
                             
        if self.transform is not None:
            image_tensor = self.transform(image_tensor)
        if self.label_transform:
            label = self.target_transform(label)
        return(image_tensor, label )

def get_CustomDataset_from_csv(df, mod, split):
    if split == 'test' or split == 'val':
        data = CustomDataset_from_csv(df , mod,
                transform=transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),                
            ])
            )
    return data

def get_test_dataloader( test_df, mod, batch_size):
    test_data =get_CustomDataset_from_csv(test_df, mod, 'test')
    print("Shape of test data :",test_data.dataframe.shape)
    test_dataloader = DataLoader(test_data, batch_size = batch_size , shuffle = False)
    return test_dataloader

def load_model_checkpoint(model_name,num_classes,  mod,device_index, to_use_pre_trained_weights, saving_dir, fold): 

    model = get_model_by_name(model_name,num_classes,  mod,device_index, to_use_pre_trained_weights )

    # PATH = Path('/home/shsingh/knowledge_distillation/kfold_result/flair/baseline')   
    PATH = Path(saving_dir)
    model_path = PATH /  model_name / f'fold{fold}' /  f'{model_name}_base_fold_{fold}.pt'
    model.load_state_dict(torch.load(model_path))
    return model





def get_normalized_confusion_matix(y_true_list, y_pred_list, save_dir,model_name,  fold):

    labels = ['healthy', 'HGG','LGG']
    label_dict ={'0':'healthy', '1':'HGG','2':'LGG'}
    cm = confusion_matrix(y_true_list, y_pred_list,  normalize='all')
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(cm)
    # Add numbers to the confusion matrix heatmap.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
    #         ax.text(x=j, y=i, s=cm[i, j], va='center', ha='center', color='white')
            ax.text(x=j, y=i, s=f'{cm[i, j]:.2f}', va='center', ha='center', color='white')
    plt.title('Confusion matrix of the classifier')
    fig.colorbar(cax)
    ax.set_xticklabels([''] + labels)
    ax.set_yticklabels([''] + labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(save_dir / f'confusion_matrix_normalized_fold{fold}.png')
    plt.show()



def get_confusion_matrix(model, dataloader, saving_dir, device_index, model_name, fold):  
    device = torch.device(device_index) if torch.cuda.is_available() else torch.device('cpu') 
    np.random.seed(31101995)
    torch.manual_seed(31101995)
    labels = ['healthy', 'HGG','LGG']
    label_dict ={'0':'healthy', '1':'HGG','2':'LGG'}
    y_pred_list = []
    y_true_list = []    
    with torch.no_grad():
        model.eval()
        for x_batch, y_batch in dataloader:
            x_batch, y_batch = x_batch.to(device), y_batch.squeeze_().to(device)
            y_test_pred = model(x_batch)
            _, y_pred_tag = torch.max(y_test_pred, dim = 1)
                        
            y_pred_list.extend(y_pred_tag.cpu().numpy())
            y_true_list.extend(y_batch.cpu().numpy())        

    print(classification_report(y_true_list, y_pred_list))
    classification_report_df = pd.DataFrame(classification_report(y_true_list, y_pred_list, output_dict=True))
    classification_report_df.rename(columns=label_dict, inplace=True)
    cm = confusion_matrix(y_true_list, y_pred_list)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(cm)
    # Add numbers to the confusion matrix heatmap.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(x=j, y=i, s=cm[i, j], va='center', ha='center', color='white')

    save_dir = saving_dir / model_name / f'fold{fold}' 
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    plt.title('Confusion matrix of the classifier')
    fig.colorbar(cax)
    ax.set_xticklabels([''] + labels)
    ax.set_yticklabels([''] + labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(save_dir / f'confusion_matrix_fold{fold}.png')
    plt.show()
    confusion_matrix_df = pd.DataFrame(confusion_matrix(y_true_list, y_pred_list),index=labels)
    confusion_matrix_df.to_csv(save_dir / f'confusion_matrix_df_fold{fold}.csv',index=False )
    classification_report_df.to_csv( save_dir / f'classification_report_df_fold{fold}.csv',index=False )

    #Normalized ConfusionMatrix
    get_normalized_confusion_matix(y_true_list, y_pred_list, save_dir,model_name,  fold)
    return confusion_matrix_df, classification_report_df


def main():


    args = parse_args()
    test_csv_path = args.test_csv    
    model_name = args.model_name
    fold = args.fold

    print('fold:',fold)

    output_dir = OUTPUT_DIR / str(args.save_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok = True)

    device_num = args.device_num
    mod = args.mod
    to_use_pre_trained_weights = args.pre_trained_weights

    class_column_name = args.class_column_name
    subject_id_column_name = args.subject_id_column_name
    class_healthy_name = args.class_healthy_name
    # saving_dir = args.saving_dir #'/home/shsingh/knowledge_distillation/kfold_result/flair/baseline'
    saving_dir = output_dir

    device_index = device_num
    batch_size = 64
    # mod = 'flair'
    num_classes = 3
    # to_use_pre_trained_weights = "True"
    device = torch.device(device_index) if torch.cuda.is_available() else torch.device('cpu')

    test_df = pd.read_csv(test_csv_path,index_col =0)
    test_df = test_df[test_df[class_column_name] != 'discard'].reset_index(drop=True)
    test_df.fillna('healthy', inplace=True)

    test_dataloader =  get_test_dataloader(test_df, mod,batch_size)

    model = load_model_checkpoint(model_name,num_classes,  mod,device_index, to_use_pre_trained_weights, saving_dir, fold)

    confusion_matrix_df, classification_report_df=  get_confusion_matrix(model.to(device), test_dataloader, saving_dir, device_index, model_name, fold)


    return


if __name__ == "__main__":
    # device_index = f'cuda:{device_num}'
    # device = torch.device('cuda:7') if torch.cuda.is_available() else torch.device('cpu')
    # device = torch.device(device_index) if torch.cuda.is_available() else torch.device('cpu')
    main()
    # torch.cuda.empty_cache()