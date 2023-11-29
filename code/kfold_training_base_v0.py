"""Train base-classifier(ResNet18) model.
    exp_01: Train  ResNet18, ResNet50 classifier
    exp_02: Train KD as [student= ResNet18 and teacher = ResNet50]
    exp_03: Train KD as [student= ResNet50  and teacher =  ResNet18]
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

python code/kfold_training_base_v0.py \
 --mapping_csv "/home/shsingh/knowledge_distillation/dataset/scratch/dataframes/meta_data_survival_mapping.csv" \
 --mod "flair"\
 --model "resnet50"\
 --save_dir "flair/baseline/resnet50/"\
 --num_epochs 50 \
 --pre_trained_weights "True" \
 --class_column_name 'label_flair' \
 --subject_id_column_name 'subject_id' \
 --class_healthy_name 'healthy' \
 --split True\
 --test_size 0.3\
 --device_num=3
"""
import os   
import sys
import argparse
import glob
import random
from tqdm import tqdm

# import imageio 
import matplotlib.pyplot as plt
# import nibabel as nib
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
from sklearn.preprocessing import LabelEncoder 
from sklearn.utils.class_weight import compute_class_weight

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import models
from torch.utils.tensorboard import SummaryWriter

#unbalanced TEST_SET
# from kfold_dataset import (prepare_train_data_split,
#                             create_groupKFold_splits,
#                             get_df_using_kfold_indexes,
#                             get_dataloader,
#                             get_multimodal_dataloader)

from kfold_balanced_dataset import (create_StratifiedGroupKFold_splits,
                            get_df_using_kfold_indexes, 
                            get_dataloader,
                            get_multimodal_dataloader)

# from create_data_split import get_train_test_df_split 
from create_balanced_data_split import get_train_test_df_split
from kfold_evaluation import plot_training_cruves, get_confusion_matrix, calculate_metrics, plot_lgg_training_curves
from models import get_model_by_name
from paths import DATASETS_DIR, MICCAI_BraTS2020_Data, OUTPUT_DIR, DATAFRAME_DIR


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
        help="total #epochs",
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
        "--split",
        type=bool,
        default=None,
        required=True,
        help="if train_df and test_df split exisits?",
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.25,
        required=False,
        help="test split fraction",
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

    if args.class_column_name is None:
        raise  ValueError("required class_column name.")
    if args.subject_id_column_name is None:
        raise  ValueError("required subject_id column name.")
    if args.class_healthy_name is None:
        raise  ValueError("required class_healthy_name  as in the class_column in df.")
    if args.split is None:
        raise  ValueError("If train and test df exists?")

    if args.device_num is None:
        raise ValueError("device number(cuda).")
    return args


def calculate_class_weights(train_df, class_column_name):

    if train_df.isnull().values.any():
        train_df = train_df.fillna('healthy')

    classes_list = train_df[class_column_name].unique().tolist()
    print(classes_list)
    label_col = train_df[class_column_name]

    class_weights = compute_class_weight(class_weight = "balanced",
                                        classes = classes_list,
                                        y = label_col) 

    class_weights_dict = dict(zip(classes_list, class_weights))    
    return class_weights, class_weights_dict


def start_train(model, train_dataloader, val_dataloader, test_dataloader, n_epochs, saving_dir, device_index, model_name, fold, class_weights):
    print_every = 1
    class_idx = 2    

    print(fold)
    writer = SummaryWriter()
												 
    lambda_ = 0.0001
    # criterion = nn.CrossEntropyLoss()
    #use weighted cross_entropy. Calculate the weights by inverting the #samples per class
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=lambda_)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)
    device = torch.device(device_index) if torch.cuda.is_available() else torch.device('cpu')

    valid_loss_min = np.Inf
    val_loss = []
    val_f1_scores = []
    val_acc = []

    val_acc_lgg = []
    

    train_loss = []
    train_f1_scores = []
    train_acc = []
    total_step = len(train_dataloader)

    train_lgg_acc = []

    for epoch in tqdm(range(n_epochs)):
        running_loss = 0.0
        train_f1 = 0.0
        train_correct = 0.0
        total_train = 0

        correct_train_lgg = 0.0
        total_lgg = 0.0

        lr_scheduler.step()
        print(f'Epoch {epoch}\n')
												  
        model.train()
        
        for batch_idx, (inputs, targets) in enumerate(train_dataloader):
            inputs, targets = inputs.to(device), targets.squeeze_().to(device)
            # class_idx = torch.tensor(class_idx).unsqueeze(0).expand_as(targets).to(device)
            optimizer.zero_grad()
						 
            outputs = model(inputs)
            loss_model = criterion(outputs, targets)
									 
            l2_reg = torch.tensor(0., requires_grad=True)
            for name, param in model.named_parameters():
                if 'weight' in name:
                    l2_reg = l2_reg + torch.norm(param, p=2) ** 2
            loss = loss_model + lambda_ * l2_reg
            # loss = loss_model 

            writer.add_scalar("Loss/train", loss, epoch)
								   
            loss.backward()
            optimizer.step()
		   
            f1 = calculate_metrics(outputs, targets)
										
            # running_loss += loss.item() * targets.size(0)
            running_loss += loss.item() 
            train_f1 += f1
            _, pred = torch.max(outputs, dim=1)
            train_correct += torch.sum(pred == targets).item()
            total_train += targets.size(0)
            # print('running_loss: ',running_loss)
            # print( 'total_train:'	, total_train)

            #learning curves only for minority class: LGG
            # correct_train_lgg += ((outputs == class_idx) & (targets == class_idx)).sum().item()
            total_lgg += (targets == class_idx).sum().item()
            # total_lgg += ((outputs.argmax(dim=1) == class_idx) & (targets == class_idx)).sum().item()
            correct_train_lgg += ((outputs.argmax(dim=1) == class_idx) & (targets == class_idx)).sum().item()

            if (batch_idx) % 200 == 0:
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' .format(epoch, n_epochs, batch_idx, total_step, loss.item()))
               																					 
        epoch_train_f1_score = train_f1 / total_train
        epoch_train_acc_score = train_correct / total_train
        epoch_train_loss = running_loss / total_train

        epoch_lgg_train = correct_train_lgg/ total_lgg
        print('correct_train_lgg:', correct_train_lgg)
        print('total_train:', total_lgg)
        print('epoch_lgg_train:', epoch_lgg_train)
        
        train_f1_scores.append(epoch_train_f1_score)
        train_loss.append(epoch_train_loss)
        train_acc.append(epoch_train_acc_score)

        train_lgg_acc.append(epoch_lgg_train)

        print('Epoch[{}/{}], train Loss: {:.4f}, train acc: {:.4f}, train f1_score: {:.4f}'.format(epoch+1, n_epochs,
                                                                                                epoch_train_loss, 
                                                                                                epoch_train_acc_score,
                                                                                                epoch_train_f1_score,
                                                                                            ))  																		
        # Validation			   
        model.eval()
        with torch.no_grad():
            batch_loss = 0.0
                            
            running_f1_score = 0.0
            running_corrects = 0						 
            running_total = 0

            running_correct_lgg = 0
            running_total_lgg = 0
            
            for data_t, target_t in (val_dataloader):
                data_t, target_t = data_t.to(device), target_t.squeeze_().to(device)
                outputs_t = model(data_t)
                loss_t = criterion(outputs_t, target_t)                                            
                                    
                f1_val = calculate_metrics(outputs_t, target_t)
                running_f1_score += f1_val
                                                                
                batch_loss += loss_t.item()
                writer.add_scalar("Loss/validation", batch_loss, epoch)
                _, pred_t = torch.max(outputs_t, dim=1)
                running_corrects += torch.sum(pred_t == target_t).item()
                                                                    
                running_total += target_t.size(0)    

                running_correct_lgg += (target_t == class_idx).sum().item()
                # running_total_lgg += (target_t == class_idx).sum()    
                running_total_lgg +=  ((outputs_t.argmax(dim=1) == class_idx) & (target_t == class_idx)).sum().item()     
            
            # print('batch_loss: ',batch_loss)
            # print( len(val_dataloader)	)
            # print('running_total:', running_total)
            # epoch_val_loss = batch_loss / running_total		
            epoch_val_loss = batch_loss / len(val_dataloader)										 
            epoch_val_f1_score = running_f1_score / running_total
            epoch_val_acc = running_corrects / running_total

            epoch_val_acc_lgg = running_correct_lgg/ running_total_lgg

            val_loss.append(epoch_val_loss)
            val_f1_scores.append(epoch_val_f1_score)
            val_acc.append(epoch_val_acc)

            val_acc_lgg.append(epoch_val_acc_lgg)

            network_learned = batch_loss < valid_loss_min
            print(f'Epoch {epoch+1}/{n_epochs}, val Loss: {epoch_val_loss:.4f}, val acc: {epoch_val_acc:.4f}, val f1_score: {epoch_val_f1_score:.4f}')
            if network_learned:
                valid_loss = batch_loss
                model_saving_dir = Path(saving_dir)
                # model_saving_path = model_saving_dir / model_name / f'fold_{fold}'
                model_saving_path = model_saving_dir /f'fold{fold}'
                # print('model_saving_path:', model_saving_path)
                if not os.path.exists(model_saving_path):
                    os.makedirs(model_saving_path)
                torch.save(model.state_dict(), model_saving_path / f'{model_name}_base_fold_{fold}.pt')
                print('Improvement-Detected, save-model')

    plot_training_cruves(train_acc, val_acc,train_loss, val_loss, train_f1_scores,val_f1_scores,saving_dir, fold )
    get_confusion_matrix(model.eval(), test_dataloader, saving_dir, device_index, fold)

    plot_lgg_training_curves(train_lgg_acc, val_acc_lgg, saving_dir,  fold)
    return model 


def train_split_df_kfold_dict(df_indexed,
                                mod, 
                                batch_size,
                                # model,
                                n_epochs , 
                                output_dir,
                                device_index, 
                                model_name,
                                class_column_name,
                                subject_id_column_name,
                                kfold_dict, k, 
                                test_df, 
                                splits_csv_path,
                                num_classes,
                                to_use_pre_trained_weights):   
    test_df = test_df[test_df[class_column_name]!= 'discard'].reset_index(drop=True)
    fold =0
    models = []
    for i in range(k):
        if fold < k:  
            print('FOLD: ', fold)     

            model = get_model_by_name(model_name, num_classes, 
                                        mod, device_index,
                                        to_use_pre_trained_weights ) 
            train_index =  kfold_dict[f'train_{fold}']
            val_index = kfold_dict[f'val_{fold}']
            train_df, val_df = get_df_using_kfold_indexes(df_indexed, 
                                                        class_column_name, 
                                                        subject_id_column_name,
                                                        train_index,
                                                        val_index,
                                                        splits_csv_path,
                                                        fold)            
            print('val_df:', len(val_df[class_column_name]))
            class_weights, class_weights_dict = calculate_class_weights(train_df, class_column_name)     
            print('class_weights:', class_weights_dict)            
            # Convert class_weights to a tensor
            class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device_index)
            # dataloader loop and continue to trainig
            if mod == 'flair_t1ce_t2':
                train_dataloader, val_dataloader, test_dataloader =  get_multimodal_dataloader(train_df,
                                                                                            test_df, 
                                                                                            val_df,
                                                                                            batch_size)
            else:
                train_dataloader, val_dataloader, test_dataloader =  get_dataloader(train_df,
                                                                                    test_df,
                                                                                    val_df,
                                                                                    mod,
                                                                                    batch_size)
            trained_model = start_train(model.to(device_index),
                                        train_dataloader, 
                                        val_dataloader,
                                        test_dataloader,
                                        n_epochs , 
                                        output_dir, 
                                        device_index,
                                        model_name,
                                        fold, class_weights)
        models.append(trained_model)

        fold+=1
    return trained_model



def main():

    args = parse_args()
    meta_csv_path = args.mapping_csv
    model_name = args.model

    output_dir = OUTPUT_DIR / str(args.save_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok = True)

    n_epochs = args.num_epochs
    device_num = args.device_num
    mod = args.mod
    to_use_pre_trained_weights = args.pre_trained_weights

    class_column_name = args.class_column_name
    subject_id_column_name = args.subject_id_column_name
    class_healthy_name = args.class_healthy_name
    split_exists = args.split
    test_size = args.test_size

    # meta_csv_path = '/home/shsingh/knowledge_distillation/dataset/scratch/dataframes/meta_data_survival_mapping.csv'
    # class_column_name = 'label_flair'
    # subject_id_column_name = 'subject_id'
    # class_healthy_name ='healthy'
    # split = True  
    # test_size = 0.25

    k = 3
    num_classes = 3
    batch_size = 64
    # test_split = 0.2
    discard = 'discard'
    split_exists = True
    splits_csv_path = DATAFRAME_DIR / 'train_test_splits'

    train_df, test_df  = get_train_test_df_split(meta_csv_path , 
                                                class_column_name,
                                                test_size,
                                                subject_id_column_name,
                                                class_healthy_name,discard,
                                                splits_csv_path, 
                                                split_exists )


    #XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX TO TRAIN ONLY UNBALACED (test)Kfold: poor results XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX    
    # train_df_indexed = prepare_train_data_split(meta_csv_path, train_df, subject_id_column_name)

    # #get KFOLD dict with respective indexes for each splits
    # kfold_dict  = create_groupKFold_splits(train_df_indexed,class_column_name, subject_id_column_name, k)

    # device_index = f'cuda:{device_num}'

    # #get model using name
    # model = get_model_by_name(model_name, num_classes, 
    #                             mod, device_index,
    #                             to_use_pre_trained_weights )

    # trained_model =  train_split_df_kfold_dict(train_df_indexed,
    #                                             mod, 
    #                                             batch_size,
    #                                             model,
    #                                             n_epochs , 
    #                                             output_dir,
    #                                             device_index, 
    #                                             model_name,
    #                                             class_column_name,
    #                                             subject_id_column_name,
    #                                             kfold_dict, k, 
    #                                             test_df, 
    #                                             splits_csv_path)
    #XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX TO TRAIN ONLY UNBALACED (test)Kfold: poor results XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX   

    #********************************************** TO TRAIN ONLY BALACED (test)Kfold **********************************************
    
    kfold_dict, kfold_dist_dict = create_StratifiedGroupKFold_splits(k, train_df,
                                                                    subject_id_column_name,
                                                                    class_column_name)
    device_index = f'cuda:{device_num}'

    trained_model =  train_split_df_kfold_dict(train_df,
                                                mod, 
                                                batch_size,
                                                # model,
                                                n_epochs , 
                                                output_dir,
                                                device_index, 
                                                model_name,
                                                class_column_name,
                                                subject_id_column_name,
                                                kfold_dict, k, 
                                                test_df, 
                                                splits_csv_path,
                                                num_classes,
                                                to_use_pre_trained_weights)



    #********************************************** TO TRAIN ONLY BALACED (test)Kfold **********************************************                                            
                                
    # writer.flush()


if __name__ == "__main__":
    # device_index = f'cuda:{device_num}'
    # device = torch.device('cuda:7') if torch.cuda.is_available() else torch.device('cpu')
    # device = torch.device(device_index) if torch.cuda.is_available() else torch.device('cpu')
    main()
    torch.cuda.empty_cache()
    # writer.close()
