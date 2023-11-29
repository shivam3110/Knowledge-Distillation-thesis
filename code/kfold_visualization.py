import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



def get_accuracy_plotting_df(df_acc):    
    df = pd.DataFrame(df_acc.T)
    df.reset_index(level=0, inplace=True)
    # Rename the column
    df = df.rename(columns={'index': 'accuracy'})
    # Convert the 'accuracy' column to numeric type
    df['accuracy'] = df['accuracy'].astype(str).str.replace(r'(\d+\.\d+)\.(\d+)', r'\1\2')
    df['accuracy'] = pd.to_numeric(df['accuracy'], errors='coerce')
    # Round the 'accuracy' column to 4 decimal places
    df['accuracy'] = df['accuracy'].round(4)
    df = df.rename_axis('epoch')
    return df


def plot_accuracy_dfs(dfs):
    # Create a list of legend labels for each dataframe
    legend_labels = ['baseline_resnet18', 'baseline_resnet50', 'kd_ts','kd_st','kd_ss','kd_tt']
    # Create a colormap for line colors
    colormap = plt.cm.get_cmap('tab10')
#     colormap = plt.cm.get_cmap('gist_heat')
    # Plot the data
    plt.figure(figsize=(10, 6))
    for i, df in enumerate(dfs):
        plt.plot(df.index, df['accuracy'], color=colormap(i), label=legend_labels[i])
    # Set plot title and labels
    plt.title('Accuracy vs Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    # Add legend
    plt.legend()
    # Show the plot
    plt.show()


def get_multiple_accuracy_plot_df(mod, split, metric):
    
    path ='/home/shsingh/knowledge_distillation/results'
    if split == 'train':
        list_name = f'list_train_{metric}'
    elif split == 'val':
        list_name = f'list_val_{metric}'
    
    train_acc_baseline18_df = pd.read_csv(f'{path}/{mod}/baseline/resnet18/{list_name}.csv')
    df_baseline18 = get_accuracy_plotting_df(train_acc_baseline18_df)

    train_acc_baseline50_df = pd.read_csv(f'{path}/{mod}/baseline/resnet50/{list_name}.csv')
    df_baseline50 = get_accuracy_plotting_df(train_acc_baseline50_df)

    train_acc_kd_ts_df = pd.read_csv(f'{path}/{mod}/kd/resnet50_resnet18/{list_name}.csv')
    df_kd_ts = get_accuracy_plotting_df(train_acc_kd_ts_df)

    train_acc_kd_st_df = pd.read_csv(f'{path}/{mod}/kd_size/resnet18_resnet50/{list_name}.csv')
    df_kd_st = get_accuracy_plotting_df(train_acc_kd_st_df)

    train_acc_kd_ss_df = pd.read_csv(f'{path}/{mod}/kd_size/resnet18_resnet18/{list_name}.csv')
    df_kd_ss = get_accuracy_plotting_df(train_acc_kd_ss_df)

    train_acc_kd_tt_df = pd.read_csv(f'{path}/{mod}/kd_size/resnet50_resnet50/{list_name}.csv')
    df_kd_tt = get_accuracy_plotting_df(train_acc_kd_tt_df)

    df_plot = [df_baseline18,df_baseline50, df_kd_ts,df_kd_st, df_kd_ss, df_kd_tt]

    return df_plot



df_plot= get_multiple_accuracy_plot_df('flair', 'train')
df_plot

plot_accuracy_dfs(df_plot)


df_plot_val= get_multiple_accuracy_plot_df('flair', 'val','f1')
df_plot_val

plot_accuracy_dfs(df_plot_val)

#################XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX#############################


#PLOT multiple baseline performance(train and val: Accuracy and loss)
def get_corrected_acc(x):
    dec_count =  x.count(".")
    if dec_count>1:
        before_dec = x.split('.')[0]
        after_dec = x.split('.')[1]
        acc = f'{before_dec}.{after_dec}'
        return acc
    else:
        return x
 
def get_accuracy_plotting_df(df_acc, model, fold, metric):    
    df = pd.DataFrame(df_acc.T)
    df.reset_index(level=0, inplace=True)

    if metric == 'acc':
        metric_col = 'accuracy'
    else:
        metric_col = metric 
        
    df = df.rename(columns={'index': f'{metric_col}'})
    df[metric_col] = df[metric_col].apply(lambda x:get_corrected_acc(x) )
    # Convert the 'accuracy' column to numeric type
    df[metric_col] = pd.to_numeric(df[metric_col], errors='coerce')
    df[metric_col] = df[metric_col].round(4)
    df['model_name'] = model
    df['fold'] = f'fold{fold}'
    df['model'] = df['model_name'].astype(str) + '_' + df['fold'].astype(str)  
    return df

def plot_accuracy_dfs(dfs, models, metric, split):
    
    if metric == 'acc':
        metric_col = 'accuracy'
    else:
        metric_col = metric 

    colormap = plt.cm.get_cmap('tab10')
    plt.figure(figsize=(10, 6))
    for i, df in enumerate(dfs):
        plt.plot(df.index, df[metric_col], color=colormap(i), label=models[i])
    # Set plot title and labels
    plt.title(f'{split} {metric} vs Epoch')
    plt.xlabel('Epoch')
    plt.ylabel(f'{metric}')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()

def plot_multiple_model_accuracy(models, folds, csv_path, metric):
    
    #metric = acc or loss or F1
    all_train_acc = []
    all_val_acc = []
    model_names = []
    
    for model in models:
        for fold in folds:                       
            df_train = pd.read_csv(csv_path / model / f'fold{fold}' / f'list_train_{metric}_fold{fold}.csv')
            df_val = pd.read_csv(csv_path / model / f'fold{fold}' / f'list_val_{metric}_fold{fold}.csv')
            
            df_train_acc = get_accuracy_plotting_df(df_train, model, fold, metric)
            df_val_acc = get_accuracy_plotting_df(df_val, model, fold, metric)
            model_name = np.unique(df_train_acc['model'])
#             print('model_name:', model_name)
            model_names.extend(model_name)
    
            all_train_acc.append(df_train_acc)
            all_val_acc.append(df_val_acc)   
        
    plot_accuracy_dfs(all_train_acc, model_names, metric,  'Train')
    plot_accuracy_dfs(all_val_acc, model_names, metric,  'Validation')
    return 


### GET MULTIPLE ACCURACY PLOT FOR CLASSIFIER:

models = ['densenet121', 'efficientnet_v2_s', 'resnet18', 'mobilenet_v2']
folds = [0, 1, 2]
metric = 'acc' #'loss', 'f1'
csv_path = Path('/home/shsingh/knowledge_distillation/kfold_result/flair/baseline/') 
plot_multiple_model_accuracy(models, folds, csv_path, metric)

#------------------------### GET MULTIPLE ACCURACY PLOT------------------------------------
models = ['densenet121', 'efficientnet_v2_s', 'resnet18', 'mobilenet_v2']
folds = [0, 1, 2]
metric = 'loss' #'loss', 'f1'
csv_path = Path('/home/shsingh/knowledge_distillation/kfold_result/flair/baseline/') 
plot_multiple_model_accuracy(models, folds, csv_path, metric)


#################XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX#############################

####### XXXXXXXXXXXXXXXXXXXXXXXX Performance Metric for Each fold, each Model and Class XXXXXXXXXXXX##############

def plot_flscore_multiple_model(df):
    plt.figure(figsize=(10, 6))
    # Define a palette with colors for each class
    palette = sns.color_palette("husl", 3)
    # Melt the DataFrame to have 'model' as x, 'value' as y, and 'variable' as hue
    df_melted = df.melt(id_vars=['model_name', 'fold', 'model'], var_name='variable', value_name='value')
    # Use seaborn to create the grouped line plot with markers
    sns.lineplot(x='model', y='value', hue='variable', style='variable', markers=True, data=df_melted, palette=palette)
    # Set labels and title
    plt.xlabel('Model')
    plt.ylabel('Value')
    plt.title('Performance Metric for Each Model and Class')
    plt.xticks(rotation=45)
    plt.legend( title='Class')
    # Add annotations with actual values
    for i, row in df_melted.iterrows():
        x_coord = i % len(df_melted['model'].unique())  # Cycle through x-coordinates for each model
        plt.text(x_coord, row['value'] + 0.005, f'{row["value"]:.2f}', ha='center', va='bottom', fontsize=8, fontweight='light', color='black')

    # Show the plot
    plt.tight_layout()
    plt.show()
    return

def change_df_index_name(input_df):
    df = input_df.rename(index={0: 'precision', 1: 'recall', 2: 'f1score', 3: 'total_samples'})
    del df['accuracy']
    del df['macro avg']
    del df['weighted avg']
    return df

def plot_model_classification_report_df(csv_path , models, folds):
    # Initialize an empty DataFrame to store metrics
    all_metrics = pd.DataFrame()
    
    for model in models:
        for fold in folds:
            metric_data = pd.read_csv(csv_path / model / f'fold{fold}' / f'classification_report_df_fold{fold}.csv')
            metric_data = change_df_index_name(metric_data)
                       
            metric_data['model_name'] = model
            metric_data['fold'] = f'fold{fold}'
            metric_data['model'] = metric_data['model_name'].astype(str) + '_' + metric_data['fold'].astype(str)
            all_metrics = pd.concat([all_metrics, metric_data])
            
    df = all_metrics.loc['f1score']
    plot_flscore_multiple_model(df)
    return df

#----------------------------------------------------------------------------------------------
######### BASELINE ########
models = ['densenet121', 'efficientnet_v2_s', 'resnet18', 'mobilenet_v2']
folds = [0, 1, 2]

#----------------------------------------------------------------------------------------------
######### KD ########
csv_path = Path('/home/shsingh/knowledge_distillation/kfold_result/flair/kd/')
# expt = 'kd'#baseline
models = ['efficientnet_densenet121','efficientnet_mobilenet']
folds = [0,1,2]
df = plot_model_classification_report_df(csv_path , models, folds)
df
#----------------------------------------------------------------------------------------------



#----------------------------------------------------------------------------------------------
######### KD_SIZE ########
models = ['mobilenet_efficientnet', 'efficientnet_efficientnet','mobilenet_mobilenet'] 


############################################
# densenet121_efficientnet
# densenet121_mobilenet
# densenet121_densenet121

#----------------------------------------------------------------------------------------------
#PLOT multiple baseline performance(train and val: Accuracy and loss)