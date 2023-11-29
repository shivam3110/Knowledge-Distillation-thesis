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


def remove_columns(meta_csv_path):    
    df_meta = pd.read_csv(meta_csv_path)
    del df_meta['Brats20ID']
    del df_meta['Age']
    del df_meta['Survival_days']
    del df_meta['Extent_of_Resection']
    return df_meta


def get_class_label_dict(df, class_column):
    class_dict = df[class_column].value_counts().to_dict()
    return class_dict


def get_minority_class_values(df_meta, class_dict, 
                              subject_id_column_name, 
                              class_column_name):
    
    samples_per_subject = max(df_meta.groupby([subject_id_column_name])[class_column_name].count())
    # print('samples_per_subject:', samples_per_subject)
    minority_samples = min(class_dict.values())
    minority_class = [k for k, v in class_dict.items() if v == minority_samples]
    minority_class = minority_class[0]
    print('minority_class:',minority_class)

    return minority_class, samples_per_subject


def get_minority_class_train_test_split(df_meta, class_dict,
                                  minority_class,test_size,
                                  subject_id_column_name,
                                  class_column_name):    
    df = df_meta.copy()
    for key in class_dict:
        if key == minority_class:
            df_class =  df[df[class_column_name] == key]
            subject_id_list =  np.unique(df_class[subject_id_column_name]).tolist()
            
            train_subjects_minority, test_subjects_minority = train_test_split(subject_id_list ,
                                            test_size = test_size, 
                                            random_state = 42, 
                                            shuffle = True)            
    print(len(train_subjects_minority), len(test_subjects_minority))
    return train_subjects_minority, test_subjects_minority


def get_majority_class_train_test_split(df_meta, class_dict,
                                        minority_class, healthy,
                                        discard, class_column_name,
                                        subject_id_column_name,
                                        test_subjects_minority):    
    train_subjects_other = []
    test_subjects_other = []
    df = df_meta.copy()    
    for key in class_dict:
        if key!=healthy and key!=discard and key!=minority_class:            
            df_class_majority =  df[df[class_column_name] == key]
            subject_id_class =  np.unique(df_class_majority[subject_id_column_name]).tolist()            
            test_subjects = random.sample(subject_id_class, len(test_subjects_minority))            
            train_subjects = list(set(subject_id_class) - set(test_subjects))            
            train_subjects_other.extend(train_subjects)
            test_subjects_other.extend(test_subjects)
    
    return train_subjects_other, test_subjects_other


def create_and_save_split(meta_csv_path, 
                         class_column_name,
                         test_size, 
                         subject_id_column_name,
                         healthy, discard,
                         csv_save_path):
    df_meta = remove_columns(meta_csv_path)
    class_dict = get_class_label_dict(df_meta, class_column_name)
    minority_class, samples_per_subject =  get_minority_class_values(df_meta, class_dict, 
                                                                    subject_id_column_name, 
                                                                    class_column_name)
    train_subjects_minority, test_subjects_minority = get_minority_class_train_test_split(df_meta, class_dict,
                                                                                        minority_class,test_size,
                                                                                        subject_id_column_name,
                                                                                        class_column_name)
    training_subjects_list, test_subjects_list = get_majority_class_train_test_split(df_meta, class_dict,
                                                                                    minority_class, healthy,
                                                                                    discard, class_column_name,
                                                                                    subject_id_column_name,
                                                                                    test_subjects_minority)
    training_subjects_list.extend(train_subjects_minority)
    test_subjects_list.extend(test_subjects_minority)

    print("TRAIN:",len(training_subjects_list ))
    print("TEST:",len(test_subjects_list ))


    df = df_meta.copy()

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
    print('Overlap exist:', result_split)
    return 


def get_train_test_df_split(meta_csv_path , 
                            class_column_name,
                            test_size,
                            subject_id_column_name,
                            healthy, discard,
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
                                            healthy, discard,
                                            splits_csv_path)

    train_class_dict = get_class_label_dict(train_df, class_column_name)
    print('train_class_dict :', train_class_dict)
    test_class_dict = get_class_label_dict(test_df, class_column_name)
    print('test_class_dict: ',test_class_dict)
    check_if_overlapping_split(train_df, test_df, subject_id_column_name)

    return train_df, test_df


def main():

    meta_csv_path = '/home/shsingh/knowledge_distillation/dataset/scratch/dataframes/meta_data_survival_mapping.csv'

    class_column_name = 'label_flair'
    subject_id_column_name = 'subject_id'

    healthy = 'healthy'
    discard = 'discard'
    test_size = 0.3

    split_exists =  False

    splits_csv_path = Path('/home/shsingh/knowledge_distillation/dataset/scratch/dataframes/train_test_splits')

    train_df, test_df = get_train_test_df_split(meta_csv_path , 
                                                class_column_name,
                                                test_size,
                                                subject_id_column_name,
                                                healthy, discard,
                                                splits_csv_path, 
                                                split_exists )


if __name__=='__main__':

    main()