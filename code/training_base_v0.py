"""Train base-classifier(ResNet18) model.
    exp_01: Train  ResNet18, ResNet50 classifier
    exp_02: Train KD as [student= ResNet18 and teacher = ResNet50]
    exp_03: Train KD as [student= ResNet50  and teacher =  ResNet18]
export LD_LIBRARY_PATH =/opt/

python code/training_base_v0.py \
 --mapping_csv "/home/shsingh/knowledge_distillation/dataset/scratch/dataframes/meta_data_survival_mapping.csv" \
 --mod "flair" \
 --save_dir "flair/exp_01/resnet18_base_no_weihgts/"\
 --num_epochs=50 \
 --pre_trained_weights=True \
 --device_num=5
"""

import os
import sys
import argparse
import glob
import random
from pathlib import Path

# import imageio 
import matplotlib.pyplot as plt
# import nibabel as nib
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.preprocessing import LabelEncoder 
# from tensorflow.keras.utils import to_categorical
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import models
from tqdm import tqdm

from dataset import CustomDataset_from_csv, split_dataset, get_dataloader, preprocess_metadata_csv, get_data_from_csv
from evaluation import plot_training_cruves, get_confusion_matrix, calculate_metrics
from models import get_resnet18_model, get_resnet18_model, get_resnet50_model, get_resnet18_model_no_preweights, get_teacher_student_model
from paths import DATASETS_DIR, MICCAI_BraTS2020_Data, OUTPUT_DIR, DATAFRAME_DIR
# from resnet import ResNet18

np.random.seed(31101995)
torch.manual_seed(31101995)

def parse_args():
    """Parse th earguments and return args"""
    parser = argparse.ArgumentParser(description="Create dataset script.")
    parser.add_argument(
        "--mapping_csv",
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
        "--save_dir",
        type=str,
        default=None,
        required=True,
        help="Path to saving dir",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=None,
        required=True,
        help="Format(png,jpg,npy)",
    )
    parser.add_argument(
        "--pre_trained_weights",
        type=bool,
        default=None,
        required=True,
        help="If to use ImageNet initial weights",
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
    if args.mapping_csv is None:
        raise ValueError("Need a mapping-csv to load.")
    if args.save_dir is None:
        raise ValueError("Need a Saving Folder.")
    if args.num_epochs is None:
        raise ValueError("num of epochs should be non-zero.")
    if args.pre_trained_weights is None:
        raise ValueError('To use ImageNet weights: YES/NO')
    if args.device_num is None:
        raise ValueError("device number(cuda).")
    return args

def preprocess_metadata_csv(csv_path):
    """Load meta csv. Filter samples to be discarded."""
    meta_data = pd.read_csv(csv_path)
    meta_data = meta_data[meta_data.label_flair != 'discard']
    meta_data = meta_data[meta_data.label_t1 != 'discard']
    meta_data = meta_data[meta_data.label_t1ce != 'discard']
    meta_data = meta_data[meta_data.label_t2 != 'discard']
    meta_data_df  = meta_data.reset_index(drop =True)
    return meta_data_df

def get_base_model(num_classes, device_index):
    device = torch.device(device_index) if torch.cuda.is_available() else torch.device('cpu')
    # model = ResNet18(pretrained = True)
    model = models.resnet18(weights=True)
    # model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    n_inputs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(n_inputs, 64),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.4),
        nn.Linear(64, num_classes),
    )
    return model.to(device)

def training(model, train_dataloader, val_dataloader, test_dataloader, n_epochs, saving_dir, device_index):
    print_every = 5
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 5, gamma = 0.5)
    # device = torch.device('cuda:7') if torch.cuda.is_available() else torch.device('cpu')
    device = torch.device(device_index) if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    valid_loss_min = np.Inf
    val_loss = []
    val_acc = []
    val_f1_scores = []

    train_loss = []
    train_acc = []
    train_f1_scores = []
    total_step = len(train_dataloader)
    for epoch in tqdm(range(n_epochs)):
    # range(1, n_epochs+1):
        running_loss = 0.0
        train_f1 = 0.0
        correct = 0
        total=0
        # lr_scheduler.step()
        print(f'Epoch {epoch}\n')
        print('LR:', lr_scheduler.get_last_lr())
        for batch_idx, (inputs, targets) in enumerate(train_dataloader):
            inputs, targets = inputs.to(device), targets.squeeze_().to(device)
            optimizer.zero_grad()
             #Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            #Backward and optimize
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            # Calculate f1-metrics 
            f1 = calculate_metrics(outputs, targets)

            running_loss += loss.item()
            _,pred = torch.max(outputs, dim=1)
            correct += torch.sum(pred==target_).item()
            total += target_.size(0)
            train_f1 += f1
            if (batch_idx) % 20 == 0:
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' .format(epoch, n_epochs, batch_idx, total_step, loss.item()))

        epoch_train_f1_score = train_f1 / total
        epoch_train_acc_score = correct / total
        epoch_train_loss = running_loss / total

        train_f1_scores.append(epoch_train_f1_score)
        train_loss.append(epoch_train_loss)
        train_acc.append(epoch_train_acc_score)

        print(f'\ntrain-loss: {np.mean(train_loss):.4f}, train-acc: {(100 * correct/total):.4f}')
        print(f'Epoch {epoch+1}/{n_epochs}, Train Loss: {epoch_train_loss:.4f},Train acc: {epoch_train_acc_score:.4f},Train F1 Score: {epoch_train_f1_score:.4f}')

        running_f1_score = 0.0
        batch_loss = 0
        total_t=0
        correct_t=0
        with torch.no_grad():
            model.eval()
            for data_t, target_t in (val_dataloader):
                data_t, target_t = data_t.to(device), target_t.squeeze_().to(device)
                outputs_t = model(data_t)
                loss_t = criterion(outputs_t, target_t)
                f1_val = calculate_metrics(outputs_t, target_t)                
                batch_loss += loss_t.item()
                _,pred_t = torch.max(outputs_t, dim=1)
                correct_t += torch.sum(pred_t==target_t).item()
                total_t += target_t.size(0)
                running_f1_score+= f1_val

            epoch_val_loss = batch_loss / total_t
            epoch_val_f1_score = running_f1_score / total_t
            epoch_val_acc = correct_t / total_t

            val_loss.append(epoch_val_loss)
            val_f1_scores.append(epoch_val_f1_score)
            val_acc.append(epoch_val_acc)


            val_acc.append(100 * correct_t/total_t)
            val_loss.append(batch_loss/len(val_dataloader))
            network_learned = batch_loss < valid_loss_min
            print(f'Epoch {epoch+1}/{n_epochs}, Train Loss: {epoch_val_loss:.4f},Train acc: {epoch_val_acc:.4f},Train F1 Score: {epoch_val_f1_score:.4f}')
            print(f'validation loss: {np.mean(val_loss):.4f}, validation acc: {(100 * correct_t/total_t):.4f}\n')
            if network_learned:
                valid_loss_min = batch_loss
                model_saving_dir = Path(saving_dir)
                torch.save(model.state_dict(), model_saving_dir / 'resnet18_base.pt')
                print('Improvement-Detected, save-model')
    model.train()
    plot_training_cruves(train_acc, val_acc,train_loss, val_loss, saving_dir )
    get_confusion_matrix(model.eval(), test_dataloader, saving_dir, device_index)
    return


def start_train(model, train_dataloader, val_dataloader,test_dataloader,  n_epochs, saving_dir, device_index):
    print_every = 1
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 3, gamma = 0.5)
    device = torch.device(device_index) if torch.cuda.is_available() else torch.device('cpu')

    valid_loss_min = np.Inf
    val_loss = []
    val_f1_scores = []
    val_acc = []

    train_loss = []
    train_f1_scores = []
    train_acc = []
    total_step = len(train_dataloader)

    for epoch in tqdm(range(n_epochs)):        
        running_loss  = 0.0
        train_f1 = 0.0
        train_correct = 0.0
        total_train = 0
        # lr_scheduler.step()
        print(f'Epoch {epoch}\n')
        print('LR:', lr_scheduler.get_last_lr())
        for batch_idx, (inputs, targets) in enumerate(train_dataloader):
            inputs, targets = inputs.to(device), targets.squeeze_().to(device)       
            optimizer.zero_grad()
            #Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            # Backward and optimize
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            
            # Calculate metrics
            f1 = calculate_metrics(outputs, targets)            
            #running_loss += loss.item()
            running_loss  += loss.item() * targets.size(0)
            train_f1 += f1
            _,pred = torch.max(outputs, dim=1)
            train_correct += torch.sum(pred==targets).item()
            total_train += targets.size(0)
    #         total += target_.size(0)
            if (batch_idx) % 200 == 0:
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' .format(epoch, 
                                                                        n_epochs,
                                                                        batch_idx, 
                                                                        total_step, loss.item()))
        epoch_train_f1_score = train_f1 / total_train
        epoch_train_acc_score = train_correct / total_train
        epoch_train_loss = running_loss / total_train
        
        train_f1_scores.append(epoch_train_f1_score)
        train_loss.append(epoch_train_loss)
        train_acc.append(epoch_train_acc_score)
        
        print(f'Epoch {epoch+1}/{n_epochs}, train Loss: {epoch_train_loss:.4f},train acc: {epoch_train_acc_score:.4f},train f1_score: {epoch_train_f1_score:.4f}')
        # Validation
        batch_loss = 0.0
        running_f1_score = 0.0
        running_corrects = 0
        correct_t = 0
        running_total = 0
        with torch.no_grad():
            model.eval()
            for data_t, target_t in (val_dataloader):
                data_t, target_t = data_t.to(device), target_t.squeeze_().to(device)
                outputs_t = model(data_t)
                loss_t = criterion(outputs_t, target_t)
                # Calculate metrics
                f1_val = calculate_metrics(outputs_t, target_t)
                running_f1_score+= f1_val
    #             running_auc_score+= auc_val                
                batch_loss += loss_t.item()
                _,pred_t = torch.max(outputs_t, dim=1)
                correct_t += torch.sum(pred_t==target_t).item()
                running_total  += target_t.size(0)
                
            epoch_val_loss = batch_loss / running_total
            epoch_val_f1_score = running_f1_score / running_total
            epoch_val_acc = correct_t / running_total

            val_loss.append(epoch_val_loss)
            val_f1_scores.append(epoch_val_f1_score)
            val_acc.append(epoch_val_acc)

            network_learned = batch_loss < valid_loss_min
            print(f'Epoch {epoch+1}/{n_epochs}, val Loss: {epoch_val_loss:.4f},val acc: {epoch_val_acc:.4f},val f1_core: {epoch_val_f1_score:.4f}')
            if network_learned:
                valid_loss_min = batch_loss
                model_saving_dir = Path(saving_dir)
                torch.save(model.state_dict(), model_saving_dir / 'resnet18_base.pt')
                print('Improvement-Detected, save-model')

    plot_training_cruves(train_acc, val_acc,train_loss, val_loss, train_f1_scores,val_f1_scores,saving_dir )
    get_confusion_matrix(model.eval(), test_dataloader, saving_dir, device_index)
    return model 

def main():

    args = parse_args()
    csv_path = args.mapping_csv
    output_dir = OUTPUT_DIR / str(args.save_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok = True)
    n_epochs = args.num_epochs
    to_use_pre_trained_weights = args.pre_trained_weights
    device_num = args.device_num
    mod = args.mod
    num_classes = 3
    test_split = 0.2
    val_split = 0.1
    batch_size = 64
    split_exists =False
    train_df, test_df, val_df  = get_data_from_csv(csv_path,split_exists, test_split, val_split,mod, output_dir,  DATAFRAME_DIR )
    # data_df = preprocess_metadata_csv(csv_path)
    # train_df, test_df, val_df = split_dataset(data_df, test_split, val_split,mod, output_dir)
    train_dataloader, val_dataloader, test_dataloader =  get_dataloader(train_df, test_df, val_df, mod, batch_size)
    device_index = f'cuda:{device_num}'
    # model = get_base_model(num_classes, device_index)
    renet18 = get_resnet18_model_no_preweights(num_classes,  device_index)
    # renet50 = get_resnet50_model(num_classes, device_index)
    model = start_train(renet18, train_dataloader, val_dataloader,test_dataloader, n_epochs , output_dir, device_index)

    # args = parse_args()
    # csv_path = args.mapping_csv
    # output_dir = OUTPUT_DIR / str(args.save_dir)
    # if output_dir is  None:
    #     os.makedirs(output_dir, exist_ok = True)
    # n_epochs = args.num_epochs
    # device_num = args.device_num
    # mod = args.mod
    # num_classes = 3
    # test_split = 0.2
    # val_split = 0.1
    # batch_size = 64

    # data_df = preprocess_metadata_csv(csv_path)
    # train_df, test_df, val_df = split_dataset(data_df, test_split, val_split,mod, output_dir)
    # train_dataloader, val_dataloader, test_dataloader =  get_dataloader(train_df, test_df, val_df, mod, batch_size )
    # device_index = f'cuda:{device_num}'
    # model = get_base_model(num_classes, device_index)
    # training(model, train_dataloader, val_dataloader,test_dataloader, n_epochs , output_dir, device_index)


if __name__ == "__main__":
    # device_index = f'cuda:{device_num}'
    # device = torch.device('cuda:7') if torch.cuda.is_available() else torch.device('cpu')
    # device = torch.device(device_index) if torch.cuda.is_available() else torch.device('cpu')
    main()
