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
from torchvision import transforms

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


def shuffle_df(df):    
    df_shuffled = df.sample(frac=1).reset_index(drop=True)
    return df_shuffled

def class_to_idx(label):
    idx = 0
    if label == 'discard':idx =4
    if label == 'healthy':idx =0
    if label == 'HGG':idx =1
    if label=='LGG': idx = 2        
    return idx

def idx_to_class(idx):    
    if idx == 4:label='discard'
    if idx == 0:label='healthy'
    if idx == 1:label='HGG'
    if idx == 2:label='LGG'      
    return label


def get_metadata_csv_label_encoded(meta_data_df):
    """Load meta csv. Filter samples to be discarded and label encoding."""
    meta_data = meta_data_df.copy()
    meta_data['label_flair'] = meta_data['label_flair'].apply(lambda x: class_to_idx(x))
    meta_data['label_t1'] = meta_data['label_t1'].apply(lambda x: class_to_idx(x))
    meta_data['label_t1ce'] = meta_data['label_t1ce'].apply(lambda x: class_to_idx(x))
    meta_data['label_t2'] = meta_data['label_t2'].apply(lambda x: class_to_idx(x))
    meta_data_df  = meta_data.reset_index(drop =True)
    return meta_data_df


def create_StratifiedGroupKFold_splits(k, train_df,
                                        subject_id_column_name,
                                        class_column_name): 
    
    train_df= train_df.fillna('healthy')
    train_df_encoded = get_metadata_csv_label_encoded(train_df)
    train_df_encoded['patient_id'] = train_df_encoded[subject_id_column_name].str[-3:].astype(int)
    
    kfold_dict = {f"train_{i}": [] for i in range(k)}
    kfold_dict.update({f"val_{i}": [] for i in range(k)})
    
    kfold_dist_dict = {f"train_{i}": [] for i in range(k)}
    kfold_dist_dict.update({f"val_{i}": [] for i in range(k)})

    n_splits = k
    # Initialize StratifiedGroupKFold
    sgkf = StratifiedGroupKFold(n_splits=n_splits)
    # Extract features and target
    X = train_df_encoded
    y = train_df_encoded[class_column_name]
    # Define the groups (subject_id in your case)
    groups = train_df_encoded['patient_id']
    
    
    # Iterate through the splits
    for i, (train_index, val_idx) in enumerate(sgkf.split(X, y,groups)):
        print(f"Fold {i}:")
        # Ensure that the validation set is balanced
        unique_classes, class_counts = np.unique(y[val_idx], return_counts=True)
        min_class_count = min(class_counts)

        balanced_val_indices = []
        for cls in unique_classes:
            cls_indices = val_idx[y[val_idx] == cls]
            balanced_val_indices.extend(cls_indices[:min_class_count])

        kfold_dict[f'val_{i}'].extend(balanced_val_indices)
        kfold_dict[f'train_{i}'].extend(train_index)

        kfold_dist_dict[f'val_{i}'] = dict(y[balanced_val_indices].value_counts())
        kfold_dist_dict[f'train_{i}'] = dict(y[train_index].value_counts())
        # If you want to see the class distribution in each fold
        print(f'Fold {len(train_index)} Class Distribution:')
        print(y[balanced_val_indices].value_counts())
    return kfold_dict, kfold_dist_dict


def get_df_using_kfold_indexes(df_indexed, 
                                class_column_name,
                                subject_id_column_name,
                                train_index, 
                                val_index,
                                splits_csv_path, 
                                fold):
    df = df_indexed.copy()
    
    # train_df = df[df[subject_id_column_name].isin(train_index)].reset_index(drop=True)
    # val_df = df[df[subject_id_column_name].isin(val_index)].reset_index(drop=True)

    #extract subset_df using kfold indexes
    train_df = df.loc[train_index].reset_index(drop=True)
    val_df = df.loc[val_index].reset_index(drop=True)

    train_df =  train_df[train_df[class_column_name]!='discard'].reset_index(drop=True)
    val_df = val_df[val_df[class_column_name]!='discard'].reset_index(drop=True)

    train_df = shuffle_df(train_df)
    val_df = shuffle_df(val_df)

    fold_save_path = splits_csv_path / f'fold_{fold}'

    if not os.path.exists(fold_save_path):
        os.makedirs(fold_save_path)

    train_df_fold_path = fold_save_path / f'train_df_fold_{fold}.csv'
    val_df_fold_path = fold_save_path / f'val_df_fold_{fold}.csv'

    if not os.path.isfile(train_df_fold_path):
        train_df.to_csv(fold_save_path / f'train_df_fold_{fold}.csv', index =False )
    else:
        train_df = pd.read_csv(train_df_fold_path)
        train_df = shuffle_df(train_df)

    if not os.path.isfile(val_df_fold_path):
        val_df.to_csv(fold_save_path / f'val_df_fold_{fold}.csv', index =False )
    else:
        val_df = pd.read_csv(val_df_fold_path)
        val_df = shuffle_df(val_df)

    return train_df, val_df


def main():
    return


if __name__=='__main__':

    class_column_name = 'label_flair'
    subject_id_column_name = 'subject_id'

    k =3
    train_df_indexed = prepare_train_data_split(meta_csv_path, train_df, subject_id_column_name)
    kfold_dict =  create_groupKFold_splits(train_df_indexed,class_column_name, subject_id_column_name,  k)



    # train_df_indexed = prepare_train_data_split(meta_csv_path, train_df, subject_id_column_name)
    # kfold_dict =  create_groupKFold_splits(train_df_indexed,class_column_name, subject_id_column_name,  k)