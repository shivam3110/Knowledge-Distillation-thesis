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

import torch
import torchvision
from torchvision import *
from torch.utils.data import Dataset, DataLoader

print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)

np.random.seed(31101995)
torch.manual_seed(31101995)

# class create_splits:
#     def __init__(self,df, class_column, subject_id_column_name):
#         self.df = df
#         self.class_column = class_column
#         self.subject_id_column_name = subject_id_column_name

def get_class_label_dict(df, class_column):
    class_dict = df[class_column].value_counts().to_dict()
    return class_dict


def get_csv_after_discard(csv_path):
    """Load meta csv. Filter samples to be discarded."""
    meta_data = pd.read_csv(csv_path)
    meta_data = meta_data[meta_data.label_flair != 'discard']
    meta_data = meta_data[meta_data.label_t1 != 'discard']
    meta_data = meta_data[meta_data.label_t1ce != 'discard']
    meta_data = meta_data[meta_data.label_t2 != 'discard']
    meta_data_df  = meta_data.reset_index(drop =True)
    return meta_data_df


def generate_subject_id_splits(df, 
                             class_column, 
                             test_size,
                             subject_id_column_name,
                             class_healthy_name):

    training_subjects_list=[]
    test_subjects_list =[]
    hgg_count = 0
    lgg_count = 0

    class_dict = get_class_label_dict(df, class_column)
    for key in class_dict:
        if key != class_healthy_name:
            df_class =  df[df[class_column] == key]
            subject_id_class =  np.unique(df_class[subject_id_column_name]).tolist()            
            train_subjects, test_subjects = train_test_split(subject_id_class ,
                                                test_size = test_size, #set test_size=0.3 or 30%
                                                random_state = 42, 
                                                shuffle = True) 
            training_subjects_list.extend(train_subjects)
            test_subjects_list.extend(test_subjects)
    return training_subjects_list, test_subjects_list


def create_and_save_split(meta_csv_path, 
                         class_column,
                         test_size,
                         subject_id_column_name,
                         class_healthy_name,
                         csv_save_path):
    # df = pd.read_csv(meta_csv_path)
    df = get_csv_after_discard(meta_csv_path)
    training_subjects_list, test_subjects_list = generate_subject_id_splits(df,
                                                                          class_column, 
                                                                          test_size, 
                                                                          subject_id_column_name, 
                                                                          class_healthy_name)
    train_df = df[df[subject_id_column_name].isin(training_subjects_list)].reset_index(drop=True)
    test_df = df[df[subject_id_column_name].isin(test_subjects_list)].reset_index(drop=True)
    
    train_df.to_csv(csv_save_path / 'train_df.csv')
    test_df.to_csv(csv_save_path / 'test_df.csv')
    return train_df, test_df


def load_train_test_split(csv_save_path):
    if  os.path.exists(csv_save_path  / 'train_df.csv'):
        train_df_path = csv_save_path / 'train_df.csv'
        test_df_path = csv_save_path / 'test_df.csv'
        try:
            train_df = pd.read_csv(csv_save_path / 'train_df.csv', index_col=0)
            test_df = pd.read_csv(csv_save_path / 'test_df.csv', index_col=0)  
        except FileNotFoundError:
            raise FileNotFoundError(f"CSV file not found in {csv_path}")
    return train_df, test_df


def check_if_overlapping_split(train_df, test_df, subject_id_column_name):    
    training_subjects_list=[]
    test_subjects_list =[]
    training_subjects_list.extend(np.unique(train_df[subject_id_column_name]).tolist())
    test_subjects_list.extend(np.unique(test_df[subject_id_column_name]).tolist())
    result_split = any(item in test_subjects_list for item in training_subjects_list)
    print(result_split)
    return 


def get_train_test_df_split(meta_csv_path , 
                            class_column_name,
                            test_size,
                            subject_id_column_name,
                            class_healthy_name,
                            splits_csv_path, 
                            split_exists ):
    if split_exists:
        if  splits_csv_path is not None:
            train_df, test_df = load_train_test_split(splits_csv_path)
    else:
        train_df, test_df =  create_and_save_split(meta_csv_path , 
                                            class_column_name,
                                            test_size,
                                            subject_id_column_name,
                                            class_healthy_name,
                                            splits_csv_path)

    train_class_dict = get_class_label_dict(train_df, class_column_name)
    print('train_class_dict :', train_class_dict)
    test_class_dict = get_class_label_dict(test_df, class_column_name)
    print('test_class_dict: ',test_class_dict)
    check_if_overlapping_split(train_df, test_df, subject_id_column_name)

    return train_df, test_df


# def main():
#     pass
#     return

# if __name__=='__main__':    
    # meta_csv_path = '/home/shsingh/knowledge_distillation/dataset/scratch/dataframes/meta_data_survival_mapping.csv'
    # class_column_name = 'label_flair'
    # subject_id_column_name = 'subject_id'
    # class_healthy_name ='healthy'
    # split = True  
    # test_size = 0.25
    # splits_csv_path = Path('/home/shsingh/knowledge_distillation/dataset/scratch/dataframes/split_by_class_label')
    
    # if split:
    #     train_df, test_df = load_train_test_csv(splits_csv_path)
    # else:
    #     train_df, test_df =  create_and_save_split(meta_csv_path , 
    #                                          class_column_name,
    #                                          test_size,
    #                                          subject_id_column_name,
    #                                          class_healthy_name,
    #                                          csv_save_path)

    # train_class_dict = get_class_label_dict(train_df, class_column_name)
    # print('train_class_dict :', train_class_dict)
    # test_class_dict = get_class_label_dict(test_df, class_column_name)
    # print(test_class_dict)

    # check_if_overlapping_split(train_df, test_df)