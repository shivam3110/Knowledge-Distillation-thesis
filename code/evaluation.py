import numpy as np
from matplotlib import pyplot as plt
import pandas as pd 
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import f1_score, roc_auc_score, roc_curve, auc
import torch 
import csv
from tqdm import tqdm
np.random.seed(31101995)
torch.manual_seed(31101995)

def save_list_to_csv(input_list, list_name, saving_dir):
    with open(f"{saving_dir}/{list_name}.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerows([input_list])

def load_model_checkpoint(PATH):    
    #Loading model-checkpoints to continue training
    # model = ResNet50() #TODO: Change model layers
    # model.fc = nn.Sequential(
    #             nn.Linear(2048, 3, bias = True),
    #             )
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    checkpoint = torch.load(PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    n_epochs = checkpoint['epoch']
    loss = checkpoint['loss']
    lr_scheduler = checkpoint['lr']
    return model, n_epochs, loss, lr_scheduler

def plot_training_cruves(train_acc, val_acc,train_loss, val_loss, train_f1_scores, val_f1_scores, saving_dir, fold ):    
    fig = plt.figure(figsize=(10,5))
    plt.title("Train-Validation Accuracy")
    plt.plot(train_acc, label='train')
    plt.plot(val_acc, label='validation')
    plt.xlabel('num_epochs', fontsize=12)
    plt.ylabel('accuracy', fontsize=12)
    plt.legend(loc='best')
    plt.savefig(saving_dir / 'training_acc_curve.png',  bbox_inches='tight')
    save_list_to_csv(train_acc, 'list_train_acc', saving_dir)
    save_list_to_csv(val_acc, 'list_val_acc', saving_dir)

    fig = plt.figure(figsize=(10,5))
    plt.title("Train-Validation F1 score")
    plt.plot(train_f1_scores, label='train')
    plt.plot(val_f1_scores, label='validation')
    plt.xlabel('num_epochs', fontsize=12)
    plt.ylabel('f1 score', fontsize=12)
    plt.legend(loc='best')
    plt.savefig(saving_dir /'training_f1_curve.png',  bbox_inches='tight')
    save_list_to_csv(train_f1_scores, 'list_train_f1', saving_dir)
    save_list_to_csv(val_f1_scores, 'list_val_f1', saving_dir)

    fig = plt.figure(figsize=(10,5))
    plt.title("Train-Validation Loss")
    plt.plot(train_loss, label='train')
    plt.plot(val_loss, label='validation')
    plt.xlabel('num_epochs', fontsize=12)
    plt.ylabel('loss', fontsize=12)
    plt.legend(loc='best')
    plt.savefig(saving_dir /'training_loss_curve.png',  bbox_inches='tight')
    save_list_to_csv(train_loss, 'list_train_loss', saving_dir)
    save_list_to_csv(val_loss, 'list_val_loss', saving_dir)
    return

def calculate_metrics(outputs, labels):
    _, preds = torch.max(outputs, 1) 
    f1 = f1_score(labels.cpu().detach().numpy(),
                  preds.cpu().detach().numpy(), 
                  average = 'weighted')
                #   average='macro')
    return f1

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
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            y_test_pred = model(x_batch)
            _, y_pred_tag = torch.max(y_test_pred, dim = 1)
                        
            y_pred_list.append(y_pred_tag.cpu().numpy())
            y_true_list.append(y_batch.cpu().numpy())      
    y_pred_list = [i[0] for i in y_pred_list]
    y_true_list = [i[0] for i in y_true_list]

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
    plt.title('Confusion matrix of the classifier')
    fig.colorbar(cax)
    ax.set_xticklabels([''] + labels)
    ax.set_yticklabels([''] + labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(saving_dir / 'confusion_matrix.png')
    plt.show()
    confusion_matrix_df = pd.DataFrame(confusion_matrix(y_true_list, y_pred_list),index=labels)
    confusion_matrix_df.to_csv( saving_dir / 'confusion_matrix_df.csv',index=False )
    classification_report_df.to_csv( saving_dir / 'classification_report_df.csv',index=False )
    return confusion_matrix_df, classification_report_df

# def get_confusion_matrix(model, dataloader, saving_dir, device_index):    
#     # device = torch.device('cuda:7') if torch.cuda.is_available() else torch.device('cpu') 
#     device = torch.device(device_index) if torch.cuda.is_available() else torch.device('cpu')   
#     y_pred_list = []
#     y_true_list = []    
#     with torch.no_grad():
#         model.eval()
#         for x_batch, y_batch in dataloader:
#             x_batch, y_batch = x_batch.to(device), y_batch.to(device)
#             y_test_pred = model(x_batch)
#             _, y_pred_tag = torch.max(y_test_pred, dim = 1)
                        
#             y_pred_list.append(y_pred_tag.cpu().numpy())
#             y_true_list.append(y_batch.cpu().numpy())      
#     y_pred_list = [i[0] for i in y_pred_list]
#     y_true_list = [i[0] for i in y_true_list]
    
#     print("Pred: ",len(y_pred_list))
#     print("True: ",len(y_true_list))
#     print(classification_report(y_true_list, y_pred_list))
#     print(confusion_matrix(y_true_list, y_pred_list))

#     confusion_matrix_df = pd.DataFrame(confusion_matrix(y_true_list, y_pred_list))
#     classification_report_df.rename(columns=label_dict, inplace=True)

#     fig, ax = plt.subplots(figsize=(7,5))         
#     sns_confusion_matrix = sns.heatmap(confusion_matrix_df, annot=True, ax=ax)
#     # sns_confusion_matrix.savefig(saving_dir / 'sns_confusion_matrix.png', dpi=4000)
#     figure =  sns_confusion_matrix.get_figure()
#     figure.savefig(saving_dir / 'confusion_matrix.png', dpi=4000)
#     return confusion_matrix_df