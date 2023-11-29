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
np.random.seed(31101995)
torch.manual_seed(31101995)

def save_list_to_csv(input_list, list_name, saving_dir):
    with open(f"{saving_dir}/{list_name}.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerows([input_list])

def load_model_checkpoint(PATH):    

    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    checkpoint = torch.load(PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    n_epochs = checkpoint['epoch']
    loss = checkpoint['loss']
    lr_scheduler = checkpoint['lr']
    return model, n_epochs, loss, lr_scheduler

def plot_lgg_training_curves(train_lgg_acc, val_acc_lgg, saving_dir,  fold):

    save_dir =  saving_dir / f'fold{fold}' 
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)\

    fig = plt.figure(figsize=(10,5))
    plt.title(f"Train-Validation LGG accuracy_fold{fold}")
    plt.plot(train_lgg_acc, label='train')
    plt.plot(val_acc_lgg, label='validation')
    plt.xlabel('num_epochs', fontsize=12)
    plt.ylabel('accuracy', fontsize=12)
    plt.legend(loc='best')
    plt.savefig(save_dir / f'training_lgg_acc_curve_fold{fold}.png',  bbox_inches='tight')
    save_list_to_csv(train_lgg_acc, f'list_train_lgg_acc_fold{fold}', save_dir)
    save_list_to_csv(val_acc_lgg, f'list_val_lgg_acc_fold{fold}', save_dir)


def plot_training_cruves(train_acc, val_acc, train_loss, val_loss, train_f1_scores, val_f1_scores, saving_dir, fold): 

    save_dir =  saving_dir / f'fold{fold}' 
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    fig = plt.figure(figsize=(10,5))
    plt.title(f"Train-Validation Accuracy_fold{fold}")
    plt.plot(train_acc, label='train')
    plt.plot(val_acc, label='validation')
    plt.xlabel('num_epochs', fontsize=12)
    plt.ylabel('accuracy', fontsize=12)
    plt.legend(loc='best')
    plt.savefig(save_dir / f'training_acc_curve_fold{fold}.png',  bbox_inches='tight')
    save_list_to_csv(train_acc, f'list_train_acc_fold{fold}', save_dir)
    save_list_to_csv(val_acc, f'list_val_acc_fold{fold}', save_dir)

    fig = plt.figure(figsize=(10,5))
    plt.title(f"Train-Validation F1 score_fold{fold}")
    plt.plot(train_f1_scores, label='train')
    plt.plot(val_f1_scores, label='validation')
    plt.xlabel('num_epochs', fontsize=12)
    plt.ylabel('f1 score', fontsize=12)
    plt.legend(loc='best')
    plt.savefig(save_dir/ f'training_f1_curve_fold{fold}.png',  bbox_inches='tight')
    save_list_to_csv(train_f1_scores, f'list_train_f1_fold{fold}', save_dir)
    save_list_to_csv(val_f1_scores, f'list_val_f1_fold{fold}', save_dir)

    fig = plt.figure(figsize=(10,5))
    plt.title(f"Train-Validation Loss_fold{fold}")
    plt.plot(train_loss, label='train')
    plt.plot(val_loss, label='validation')
    plt.xlabel('num_epochs', fontsize=12)
    plt.ylabel('loss', fontsize=12)
    plt.legend(loc='best')
    plt.savefig(save_dir / f'training_loss_curve_fold{fold}.png',  bbox_inches='tight')
    save_list_to_csv(train_loss, f'list_train_loss_fold{fold}', save_dir)
    save_list_to_csv(val_loss, f'list_val_loss_fold{fold}', save_dir)
    return

def calculate_metrics(outputs, labels):
    _, preds = torch.max(outputs, 1) 
    f1 = f1_score(labels.cpu().detach().numpy(),
                  preds.cpu().detach().numpy(), 
                  average = 'weighted')
                #   average='macro')
    return f1


def get_normalized_confusion_matix(y_true_list, y_pred_list, save_dir, fold):

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



def get_confusion_matrix(model, dataloader, saving_dir, device_index, fold):  
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

    save_dir = saving_dir / f'fold{fold}' 
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
    get_normalized_confusion_matix(y_true_list, y_pred_list, save_dir, fold)
    return confusion_matrix_df, classification_report_df

