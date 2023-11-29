"""Train base-classifier(ResNet18) model.
    exp_01: Train  ResNet18, ResNet50 classifier
    exp_02: Train KD as [student= ResNet18 and teacher = ResNet50]
    exp_03: Train KD as [student= ResNet50  and teacher =  ResNet18]
-------------------------------------------------------------------------------
1. model_name == 'resnet18':
2. model_name == 'resnet50':
3. model_name == 'teacher_student':
4. model_name == 'student_teacher':
5. model_name == 'teacher_teacher':
6. model_name == 'student_student':
7. model_name = 'bidirectional_kd':   

------------------------------------------------------------------------------------
modularities:
flair
flair_t1ce_t2
t1ce
--------------------------------------------------------------------------------------

python code/training_kd_v0.py\
 --mapping_csv "/home/shsingh/knowledge_distillation/dataset/scratch/dataframes/meta_data_survival_mapping.csv" \
 --mod "flair_t1ce_t2" \
 --model "teacher_teacher" \
 --save_dir "flair_t1ce_t2/kd_size/resnet50_resnet50/"\
 --num_epochs 40 \
 --pre_trained_weights "True" \
 --device_num=6
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

from dataset import CustomDataset_from_csv, split_dataset, get_dataloader,get_multimodal_dataloader, preprocess_metadata_csv, get_data_from_csv
from evaluation import plot_training_cruves, get_confusion_matrix, calculate_metrics
from models import get_model_by_name
from paths import DATASETS_DIR, MICCAI_BraTS2020_Data, OUTPUT_DIR, DATAFRAME_DIR
# from resnet import ResNet18
from torch.utils.tensorboard import SummaryWriter


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
        "--model",
        type=str,
        default=None,
        required=True,
        help="model name: ResNet18 or ResNet50 or KD.",
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
        type=str,
        default=False,
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
    if args.model is None:
        raise ValueError("Please provide name the model to use.")
    if args.save_dir is None:
        raise ValueError("Need a Saving Folder.")
    if args.num_epochs is None:
        raise ValueError("num of epochs should be non-zero.")
    if args.pre_trained_weights is None:
        raise  ValueError("to use pretrained weights, True/False.")
    if args.device_num is None:
        raise ValueError("device number(cuda).")
    return args


def loss_fn_kd(student_pred, teacher_pred, alpha, T):
    """
    Compute the knowledge-distillation (KD) loss given outputs, labels.
    "Hyperparameters": temperature and alpha
    NOTE: the KL Divergence for PyTorch comparing the softmaxs of teacher
    and student expects the input tensor to be log probabilities! 
    """
    # alpha = 0.9
    # T = 3
    KD_loss = nn.KLDivLoss(reduction="batchmean")(F.log_softmax(student_pred/T, dim=1), F.softmax(teacher_pred/T, dim=1))
    
#     KD_loss = nn.KLDivLoss()(F.log_softmax(stud_labels/T, dim=1),
#                              F.softmax(teacher_outputs/T, dim=1)) * (alpha * T * T) 

    return KD_loss

def train_kd_teacher_student(teacher, student, train_dataloader,val_dataloader,
                            test_dataloader, n_epochs , saving_dir, device_index,
                            model_name, alpha, T ):

    print_every = 1
    # Define the regularization strength (lambda)
    lambda_ = 0.001
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(student.parameters(), lr=0.0001,  weight_decay=lambda_)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 3, gamma = 0.5)
    # criterion = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(student.parameters(), lr=0.0001)
    #using learning rate scheduler hampers smooth training.
    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 5, gamma = 0.001)
    device = torch.device(device_index) if torch.cuda.is_available() else torch.device('cpu')
    teacher.to(device)
    student.to(device)

    valid_loss_min = np.Inf
    val_loss = []
    val_f1_scores = []
    val_acc = []

    train_loss = []
    train_f1_scores = []
    train_acc = []
    total_step = len(train_dataloader)

    # alpha = 0.9    
    for epoch in tqdm(range(n_epochs)):  
        running_loss  = 0.0
        train_f1 = 0.0
        train_correct = 0.0
        total_train = 0
        lr_scheduler.step()
        print(f'Epoch {epoch}\n')
        student.train()
        for batch_idx, (inputs, targets) in enumerate(train_dataloader):
            inputs, targets = inputs.to(device), targets.squeeze_().to(device)  
            optimizer.zero_grad()
            #Forward pass
            #get SOFT-LABEL of the teacher model
            with torch.no_grad():
                teacher.eval()
                teacher_pred = teacher(inputs) 
            teacher_pred = teacher_pred.to(torch.float32)            
            #get SOFTMAX-LABEL of the studnet model  
            outputs = student(inputs)
            # outputs = outputs.to(torch.float32)  

            #Calculating the loss between soft_labels(teacher) and the student preds
            kd_loss = loss_fn_kd(outputs, teacher_pred, alpha, T)
            classfication_loss = criterion(outputs, targets)
            
            loss = alpha* classfication_loss + (1-alpha) * kd_loss  

            # Apply L2 regularization
            l2_reg = torch.tensor(0., requires_grad=True)
            for name, param in student.named_parameters():
                if 'weight' in name:
                    l2_reg = l2_reg + torch.norm(param, p=2) ** 2
            loss = loss + lambda_ * l2_reg
            # Backward and optimize
            loss.backward()
            optimizer.step()
            # lr_scheduler.step()            
            
            # Calculate metrics
            f1 = calculate_metrics(outputs, targets) 
            #get batch-loss
            # running_loss += loss.item()
            running_loss  += loss.item()                        
            _,pred = torch.max(outputs, dim=1)
            train_correct += torch.sum(pred==targets).item()
            train_f1 += f1
            total_train += targets.size(0)
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
        print('Epoch[{}/{}], train Loss: {:.4f},train acc: {:.4f}, train f1_score:{:.4f}'.format(epoch+1, n_epochs,
                                                                                            epoch_train_loss, 
                                                                                            epoch_train_acc_score,
                                                                                            epoch_train_f1_score,
                                                                                        ))        
        # Validation
        batch_loss = 0.0
        running_f1_score = 0.0
        running_corrects = 0
        correct_t = 0
        running_total = 0
        with torch.no_grad():
            student.eval()
            for data_t, target_t in (val_dataloader):
                data_t, target_t = data_t.to(device), target_t.squeeze_().to(device)
                #OFFLINE_TRAINING: Getting soft labels from the Teacher model
                with torch.no_grad():
                    teacher.eval()
                    teacher_pred_t = teacher(data_t) 

                teacher_pred_t = teacher_pred_t.to(torch.float32) 
                #get SOFTMAX-LABEL of the student model  
                outputs_t = student(data_t)
                outputs_t = outputs_t.to(torch.float32) 

                #Calculating the loss between soft_labels(teacher) and the student preds
                kd_loss_t = loss_fn_kd(outputs_t, teacher_pred_t, alpha, T)
                classfication_loss_t = criterion(outputs_t, target_t)                
                loss_t = alpha* classfication_loss_t + (1-alpha) * kd_loss_t  
                # Calculate metrics
                f1_val = calculate_metrics(outputs_t, target_t)
                               
                batch_loss += loss_t.item()
                _,pred_t = torch.max(outputs_t, dim=1)
                correct_t += torch.sum(pred_t==target_t).item()
                running_f1_score+= f1_val
                running_total  += target_t.size(0)
                
            epoch_val_loss = batch_loss / running_total
            epoch_val_f1_score = running_f1_score / running_total
            epoch_val_acc = correct_t / running_total

            val_loss.append(epoch_val_loss)
            val_f1_scores.append(epoch_val_f1_score)
            val_acc.append(epoch_val_acc)

            network_learned = batch_loss < valid_loss_min
            print('Epoch[{}/{}], val Loss: {:.4f},val acc: {:.4f}, val f1_score:{:.4f}'.format(epoch+1, n_epochs,
                                                                                    epoch_val_loss, 
                                                                                    epoch_val_acc,
                                                                                    epoch_val_f1_score,
                                                                                )) 
            # print(f'Epoch {epoch+1}/{n_epochs}, val Loss: {epoch_val_loss:.4f},val acc: {epoch_val_acc:.4f},val f1_core: {epoch_val_f1_score:.4f}')
            if network_learned:
                valid_loss_min = batch_loss
                model_saving_dir = Path(saving_dir)
                model_saving_path = model_saving_dir / model_name
                torch.save(student.state_dict(), model_saving_dir / f'{model_name}_offline_kd.pt')
                print('Improvement-Detected, save-model')

    plot_training_cruves(train_acc, val_acc,train_loss, val_loss, train_f1_scores,val_f1_scores,saving_dir )
    get_confusion_matrix(student.eval(), test_dataloader, saving_dir, device_index)       
    return  student
         

def main():

    args = parse_args()
    csv_path = args.mapping_csv
    model_name = args.model
    output_dir = OUTPUT_DIR / str(args.save_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok = True)
    n_epochs = args.num_epochs
    device_num = args.device_num
    mod = args.mod
    to_use_pre_trained_weights = args.pre_trained_weights

    num_classes = 3
    test_split = 0.2
    val_split = 0.1
    batch_size = 64
    alpha = 0.9
    T = 3
    seed_everything(seed= 31101995)
    writer = SummaryWriter()
    split_exists = True
    train_df, val_df, test_df  = get_data_from_csv(csv_path, split_exists, test_split, val_split, mod, output_dir,  DATAFRAME_DIR )

    if mod == 'flair_t1ce_t2':
        train_dataloader, val_dataloader, test_dataloader =  get_multimodal_dataloader(train_df, test_df, val_df, batch_size)
    else:
        train_dataloader, val_dataloader, test_dataloader =  get_dataloader(train_df, test_df, val_df, mod, batch_size)
    device_index = f'cuda:{device_num}'
    # train_dataloader, val_dataloader, test_dataloader =  get_dataloader(train_df, test_df, val_df, mod, batch_size)
    # device_index = f'cuda:{device_num}'
    teacher, student = get_model_by_name(model_name,num_classes, mod,device_index, to_use_pre_trained_weights )
    # trained_model = start_train(model, train_dataloader, val_dataloader,test_dataloader, n_epochs , output_dir, device_index, model_name)
    trained_student_model = train_kd_teacher_student(teacher, student, 
                                                     train_dataloader,val_dataloader, test_dataloader,
                                                     n_epochs,  output_dir, device_index, model_name,
                                                     alpha, T )
    # model = training(renet50, train_dataloader, val_dataloader,test_dataloader, n_epochs , output_dir, device_index)




if __name__ == "__main__":
    # device_index = f'cuda:{device_num}'
    # device = torch.device('cuda:7') if torch.cuda.is_available() else torch.device('cpu')
    # device = torch.device(device_index) if torch.cuda.is_available() else torch.device('cpu')
    main()
