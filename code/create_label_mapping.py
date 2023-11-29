import glob
import os
from pathlib import Path
import random

import imageio  
import numpy as np
import pandas as pd
from PIL import Image
import torch

from paths import DATASETS_DIR, MICCAI_BraTS2020_Data

np.random.seed(1000)
torch.manual_seed(1000)

def get_mod_label(subject, seg, mod, df_name):
    label = ""       
    #Check if HEALTHY. 
    #Healthy:if the seg != [2,3] i.e. non-enhancing tumour and nunique_modularity != 0 assign label == Healthy
    seg_unique = np.unique(seg)
    mod_unique = np.unique(mod)
    mod_sum = np.sum(mod)
    enhancing_seg_lbl = [2,3] #TODO: take this label as user input:: enhancing_seg_lbl
    non_enhancing_seg_lbl = [0,1] #TODO: take this label as user input::non_enhancing_seg_lbl
    if (mod_sum!=0 and (all(i not in  seg_unique for i in enhancing_seg_lbl))):
        return 'healthy'
    elif(mod_sum!=0 and (any(i in seg_unique for i in enhancing_seg_lbl))):
        label = df_name.loc[df_name['BraTS_2020_subject_ID']== subject]['Grade'].values[0]
        return label        
        
def get_all_mod_label(subject,seg_path, flair_path, t1_path, t1ce_path, t2_path, df_name):
    seg = np.load(seg_path)
    flair = np.load(flair_path)
    t1 = np.load(t1_path)
    t1ce = np.load(t1ce_path)
    t2 = np.load(t2_path)
    
    mod_list = [flair, t1, t1ce, t2]
    lbl_discard = 'discard'
    #Check if only  0's mod and seg
    if (np.sum(mod_list)) == 0: 
        return lbl_discard, lbl_discard, lbl_discard, lbl_discard
    else:
        flair_lbl = get_mod_label(subject, seg, flair, df_name)
        t1_lbl = get_mod_label(subject, seg, t1, df_name)
        t1ce_lbl = get_mod_label(subject, seg, t1ce, df_name)
        t2_lbl = get_mod_label(subject, seg, t2, df_name)
    return flair_lbl, t1_lbl, t1ce_lbl, t2_lbl

def create_mapping_csv(dataset_dir, df_name):
    map_df = pd.DataFrame(columns=['subject_id','image_idx','image_path','label_flair','label_t1', 'label_t1ce', 'label_t2'])
    subject_ids = sorted(os.listdir(dataset_dir))
    substring_subject_id ='BraTS20_'
    for subject in subject_ids:
        if substring_subject_id in subject:
            subject_path = dataset_dir / subject
            subject_modularities = sorted(os.listdir(subject_path))
            for modularity in subject_modularities:
                if '_seg' in modularity:
                    subject_seg_dir = subject_path / modularity
                    subject_seg_files = sorted(os.listdir(subject_seg_dir))
                    for file in subject_seg_files:
                        subject_seg_path = subject_seg_dir / file
                        image_id = str(file).split(".")[0]
                        subject_seg_path_str = str(subject_seg_path)
                        subject_flair_path = subject_seg_path_str.replace('seg', 'flair')
                        subject_t1_path = subject_seg_path_str.replace('seg', 't1')
                        subject_t1ce_path = subject_seg_path_str.replace('seg', 't1ce') 
                        subject_t2_path = subject_seg_path_str.replace('seg', 't2')
                        lbl_falir, lbl_t1, lbl_tce, lbl_t2 = get_all_mod_label(subject, subject_seg_path,subject_flair_path,
                                                                               subject_t1_path,subject_t1ce_path,
                                                                               subject_t2_path, df_name)
                        df_tmp = pd.DataFrame([(subject, image_id, subject_seg_path,lbl_falir, lbl_t1, lbl_tce, lbl_t2 )], 
                                          columns=['subject_id','image_idx','image_path','label_flair','label_t1', 'label_t1ce', 'label_t2'])
#                         map_df = map_df.append(df_tmp) 
                        map_df = pd.concat([map_df, df_tmp], axis =0) 
    map_df_reset = map_df.copy()
    map_df_reset = map_df_reset.reset_index(drop=True)
    return map_df, map_df_reset
                     

def main():
    return 0

if __name__ == "__main__":
    dataset_name= 'MICCAI_BraTS20_trainigdata_npz'
    meta_csv = 'name_mapping.csv'
    survival_csv = 'survival_info.csv'

    dataset_dir = DATASET_DIR / dataset_name
    #Load meta_csv from raw_data_dir
    meta_csv_path = MICCAI_BraTS2020_Data  / str(meta_csv)
    survival_csv_path = MICCAI_BraTS2020_Data  / str(survival_csv)

    df_meta =  pd.read_csv(meta_csv_path)
    df_meta =  df_name.drop(['BraTS_2017_subject_ID','BraTS_2018_subject_ID','TCGA_TCIA_subject_ID','BraTS_2019_subject_ID'], axis =1)
    df_meta.head()

    survival_df =  pd.read_csv(survival_csv_path)
    survival_df.head()

    main()  