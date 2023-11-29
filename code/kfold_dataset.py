import glob
import copy
import os
import random
from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GroupKFold,  StratifiedKFold, StratifiedGroupKFold

import torch
import torchvision
from torchvision import *
from torch.utils.data import Dataset, DataLoader

import time
import copy
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
from torchvision import *
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import make_grid
from torchvision.io import read_image
from torchvision.transforms import Resize

from paths import DATAFRAME_DIR

np.random.seed(31101995)
torch.manual_seed(31101995)

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
        # print(np.shape(image_tensor))
        # image = image.astype('float32')

        label_col = 'label_' + str(self.mod)     
        label_mod = self.dataframe.loc[idx, label_col]
        class_label = class_to_idx(label_mod)
        label = torch.tensor([class_label])
                             
        if self.transform is not None:
            image_tensor = self.transform(image_tensor)
        if self.label_transform:
            label = self.target_transform(label)
        return(image_tensor, label )


class CustomDataset_multimodal_from_csv(Dataset):
    def __init__(self, data_df , mod, transform = None , label_transform= None):
        self.dataframe = data_df #pd.read_csv(annotations_file)
        self.mod = mod
        self.transform = transform
        self.label_transform = label_transform
        
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self , idx):
        img_path =  self.dataframe.loc[idx,'image_path'] 
        img_path_flair = img_path.replace('seg','flair')
        img_path_t1ce = img_path.replace('seg','t1ce')
        img_path_t2 = img_path.replace('seg','t2')
        img_flair = np.load(img_path_flair)
        img_t1ce = np.load(img_path_t1ce)
        img_t2 = np.load(img_path_t2)

        img_flair_resize = np.repeat(np.resize(img_flair, (224,224))[..., np.newaxis], 1, -1 )
        img_t1ce_resize =np.repeat(np.resize(img_t1ce, (224,224))[..., np.newaxis], 1, -1 )
        img_t2_resize = np.repeat(np.resize(img_t2, (224,224))[..., np.newaxis], 1, -1 )

        img_multimodal =  np.concatenate((img_flair_resize, img_t1ce_resize, img_t2_resize), axis=-1)
        image_tensor = torch.from_numpy(img_multimodal).permute(2, 0, 1).float()

        # image = np.repeat(grayscale_image[..., np.newaxis], 3, -1)
        # img_multimodal = img_multimodal.astype('float32')
        label_col = 'label_' + str(self.mod)   
        label_mod = self.dataframe.loc[idx, label_col]
        class_label = class_to_idx(label_mod)
        label = torch.tensor([class_label])
                             
        if self.transform is not None:
            image_tensor = self.transform(image_tensor)
        if self.label_transform:
            label = self.target_transform(label)
        return(image_tensor, label )


def get_distribution_ratio(split_dict):
    total = sum(split_dict.values())
    result = {key: '{:.2f}'.format(value / total) for key, value in split_dict.items()}
    print(result)
    return result
    

def get_class_distribution(train_df, test_df, val_df, mod, output_dir):
    """Create dataframe for class-distribution for each split(train, val, test).        
    """
    label_col = f'label_{mod}'
    train_dict = train_df[label_col].value_counts().to_dict()
    val_dict = val_df[label_col].value_counts().to_dict()
    test_dict = test_df[label_col].value_counts().to_dict()
    index_labels=['train','val','test']
    class_distribution_df = pd.DataFrame([train_dict, val_dict,test_dict], 
                                                   index=index_labels,
                                                  )
    class_distribution_ratio_df = pd.DataFrame([get_distribution_ratio(train_dict), 
                                                 get_distribution_ratio(val_dict),
                                                 get_distribution_ratio(test_dict)], 
                                                     index=index_labels,
                                                    )   
    class_distribution_df.to_csv(output_dir / 'class_distribution_df.csv', index =False)
    class_distribution_ratio_df.to_csv(output_dir / 'class_distribution_ratio_df.csv', index =False)
    return class_distribution_df, class_distribution_ratio_df


def get_split_ratios(subject_list, training_subjects, val_subjects, test_subject  ):
    train_ratio = len(training_subjects)/len(subject_list)
    test_ratio = len(test_subject)/len(subject_list)
    val_ratio = len(val_subjects)/len(subject_list)
    print("train_split :", '{:.3f}'.format(train_ratio),"test_split :",  '{:.3f}'.format(test_ratio), "val_split :", '{:.3f}'.format(val_ratio))
    print("train_samples :",len(training_subjects), "test_samples :",len(test_subject), "val_samples :",len(val_subjects))
    return train_ratio, val_ratio, test_ratio


def preprocess_metadata_csv(csv_path):
    """Load meta csv. Filter samples to be discarded."""
    meta_data = pd.read_csv(csv_path)
    meta_data = meta_data[meta_data.label_flair != 'discard']
    meta_data = meta_data[meta_data.label_t1 != 'discard']
    meta_data = meta_data[meta_data.label_t1ce != 'discard']
    meta_data = meta_data[meta_data.label_t2 != 'discard']
    meta_data_df  = meta_data.reset_index(drop =True)
    return meta_data_df


def get_CustomDataset_from_csv(df, mod, split):
    if split == 'train':
        data = CustomDataset_from_csv(df , mod,
            transform=transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomRotation(degrees=30),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),    
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),        
                transforms.ToTensor(),
                # transforms.Normalize(mean=[0.0214, 0.0228, 0.0341], std=[0.2257, 0.2486, 0.2636]),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                # transforms.Normalize(mean=[0.0355, 0.0164, 0.0047], std=[0.2799, 0.1805, 0.1009]),(subset)            
            ])
            )
            #TODO: Calculate mean and stdD for .npy dataset to use for normalizing
    if split == 'test' or split == 'val':
        data = CustomDataset_from_csv(df , mod,
                transform=transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                    # transforms.Normalize(mean=[0.0214, 0.0228, 0.0341], std=[0.2257, 0.2486, 0.2636]),
            ])
            )
    return data


def get_dataloader(train_df, test_df, val_df, mod, batch_size):
    train_data = get_CustomDataset_from_csv(train_df, mod, 'train')
    test_data =get_CustomDataset_from_csv(test_df, mod, 'test')
    val_data = get_CustomDataset_from_csv(val_df, mod, 'val')
    print("Shape of trainig data :",train_data.dataframe.shape)
    print("Shape of validation data :",val_data.dataframe.shape)
    print("Shape of test data :",test_data.dataframe.shape)
    
    train_dataloader = DataLoader(train_data, batch_size = batch_size , shuffle = True)
    val_dataloader = DataLoader(val_data, batch_size = batch_size , shuffle = False)
    test_dataloader = DataLoader(test_data, batch_size = batch_size , shuffle = False)
    return  train_dataloader, val_dataloader, test_dataloader


def get_multimodal_CustomDataset_from_csv(df, mod):
    data = CustomDataset_multimodal_from_csv(df , mod,
           transform=transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomRotation(degrees=30),
            transforms.RandomRotation(degrees=270),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),    
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.0214, 0.0228, 0.0341], std=[0.2257, 0.2486, 0.2636]),
            # transforms.Normalize(mean=[0.0355, 0.0164, 0.0047], std=[0.2799, 0.1805, 0.1009]),(subset)            
           ])
           )
    return data


def get_multimodal_dataloader(train_df, test_df, val_df, batch_size): 
    train_data = get_multimodal_CustomDataset_from_csv(train_df, 'flair')
    test_data =get_multimodal_CustomDataset_from_csv(test_df, 'flair')
    val_data = get_multimodal_CustomDataset_from_csv(val_df, 'flair')
    print("Shape of trainig data :",train_data.dataframe.shape)
    print("Shape of validation data :",val_data.dataframe.shape)
    print("Shape of test data :",test_data.dataframe.shape)
    
    train_dataloader = DataLoader(train_data, batch_size = batch_size , shuffle = True)
    val_dataloader = DataLoader(val_data, batch_size = batch_size , shuffle = True)
    test_dataloader = DataLoader(test_data, batch_size = batch_size , shuffle = True)
    return  train_dataloader, val_dataloader, test_dataloader


def preprocess_train_df(train_df, subject_id_column_name):
    train_df = train_df.fillna(0)
    train_df = get_metadata_csv_label_encoded(train_df)
    train_df.set_index(subject_id_column_name, inplace=True, drop=False)
    return train_df


def prepare_train_data_split(meta_csv_path, train_df, subject_id_column_name):
    #read meta_data_csv with 'discard' as well
    df_meta = pd.read_csv(meta_csv_path)
    #get the correct indexes (subject_id) from split train_df(as train & test splits also accounts the class distribution)
    # use the only indexes from train_df to be considered from meta_data_csv
    training_subjects_list =  np.unique(train_df[subject_id_column_name]).tolist()

    train_df_indexed = df_meta[df_meta[subject_id_column_name].isin(training_subjects_list)].reset_index(drop=True)
    return train_df_indexed


def create_groupKFold_splits(df_indexed,class_column_name, subject_id_column_name,  k):
    
    kfold_dict = {f"train_{i}": [] for i in range(k)}
    kfold_dict.update({f"val_{i}": [] for i in range(k)})
    
    df = df_indexed.copy()

    X = df
    y = df[class_column_name]
    
    groups = df[subject_id_column_name].values
    group_kfold = GroupKFold(n_splits=3)
    group_kfold.get_n_splits(X, y, groups)
    
    for i, (train_index, val_index) in enumerate(group_kfold.split(X, y,groups)):        
        print(f"Fold {i}:")
        kfold_dict[f'train_{i}'].extend(groups[train_index])
        kfold_dict[f'val_{i}'].extend(groups[val_index])
        
    return kfold_dict


def get_df_using_kfold_indexes(df_indexed, 
                                class_column_name,
                                subject_id_column_name,
                                train_index, 
                                val_index,
                                splits_csv_path, 
                                fold):
    df = df_indexed.copy()
    # df.set_index('subject_id', inplace=True, drop=False)
    # train_df = df.iloc[train_index]
    # val_df = df.iloc[val_index]
    
    train_df = df[df[subject_id_column_name].isin(train_index)].reset_index(drop=True)
    val_df = df[df[subject_id_column_name].isin(val_index)].reset_index(drop=True)

    train_df[train_df[class_column_name]!='discard']
    val_df[val_df[class_column_name]!='discard']

    fold_save_path = splits_csv_path / f'fold_{fold}'

    if not os.path.exists(fold_save_path):
        os.makedirs(fold_save_path)

    train_df_fold_path = fold_save_path / f'train_df_fold_{fold}.csv'
    val_df_fold_path = fold_save_path / f'val_df_fold_{fold}.csv'

    if not os.path.isfile(train_df_fold_path):
        train_df.to_csv(fold_save_path / f'train_df_fold_{fold}.csv', index =False )
    else:
        train_df = pd.read_csv(train_df_fold_path)

    if not os.path.isfile(val_df_fold_path):
        val_df.to_csv(fold_save_path / f'val_df_fold_{fold}.csv', index =False )
    else:
        val_df = pd.read_csv(val_df_fold_path)
    
        
    # train_df.to_csv(fold_save_path / f'train_df_fold_{fold}.csv', index =False )
    # val_df.to_csv(fold_save_path / f'val_df_fold_{fold}.csv', index =False )
    train_df = train_df.sample(frac=1).reset_index(drop=True)
    val_df = val_df.sample(frac=1).reset_index(drop=True)
    print('len val_df:',len(val_df['subject_id']))
    return train_df, val_df


# def train_split_df_kfold_dict(df_indexed, class_column_name, subject_id_column_name, kfold_dict, k, test_df, splits_csv_path):
#     fold =0
#     if fold < k:        
#         train_index =  kfold_dict[f'train_{i}']
#         val_index = kfold_dict[f'val_{i}']
#         print(len(train_index), len(val_index))
#         train_df, val_df = get_df_using_kfold_indexes(df_indexed, 
#                                                       class_column_name, 
#                                                       subject_id_column_name,
#                                                       train_index,
#                                                       val_index,
#                                                       splits_csv_path,
#                                                       fold)
#         # dataloader loop and continue to trainig
#         if mod == 'flair_t1ce_t2':
#             train_dataloader, val_dataloader, test_dataloader =  get_multimodal_dataloader(train_df,
#                                                                                            test_df, 
#                                                                                            val_df,
#                                                                                            batch_size)
#         else:
#             train_dataloader, val_dataloader, test_dataloader =  get_dataloader(train_df,
#                                                                                 test_df,
#                                                                                 val_df,
#                                                                                 mod,
#                                                                                 batch_size)
#         trained_model = start_train(model.to(device_index), train_dataloader, 
#                                     val_dataloader,test_dataloader,
#                                     n_epochs , output_dir, 
#                                     device_index, model_name, fold)
#         fold=+1
#     return


def main():
    return


if __name__=='__main__':

    k =3
    train_df_indexed = prepare_train_data_split(meta_csv_path, train_df, subject_id_column_name)
    kfold_dict =  create_groupKFold_splits(train_df_indexed,class_column_name, subject_id_column_name,  k)



    # train_df_indexed = prepare_train_data_split(meta_csv_path, train_df, subject_id_column_name)
    # kfold_dict =  create_groupKFold_splits(train_df_indexed,class_column_name, subject_id_column_name,  k)