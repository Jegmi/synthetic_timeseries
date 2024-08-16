import numpy as np
import sys
import pickle
import yaml
from pathlib import Path

import pandas as pd

from matplotlib import pyplot as plt
import matplotlib
# !pip install seaborn
import seaborn as sns

sys.path.append('/Users/jegmij01/Library/Mobile Documents/com~apple~CloudDocs/24-mt-sinai/24-Ipek-CPP-project/source/')
sys.path.append('/Users/jegmij01/Library/Mobile Documents/com~apple~CloudDocs/24-mt-sinai/24-Ipek-CPP-project/source/utils/')

#!pip install tqdm

# custom data, vars, functions
from scripts.config import PROJECT_PATH, RAW_PATH, PROCESSED_PATH, LOG_PATH, CONFIG_PATH, SOURCE_PATH, FIG_PATH, ID_SHORT, DPI
from models.models import get_models
from utils.statistics_ops import eval_cv, gini
from utils.data_massaging import add_numeric_hash_column
from utils.plotting_utils import legend

from sklearn.decomposition import PCA
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

def make_features(x_, n_tile):
    """
    Generates feature vectors from the input array by computing counts of specific values,
    number of changes, and average distance between changes.

    Parameters:
    x_ (np.ndarray): Input array.
    n_tile (int): Number of tiles to split the input array into.

    Returns:
    tlin (np.ndarray): Linear space array for plotting.
    features (np.ndarray): Array of computed features.
    """
    # Initialize an array to hold the feature vectors
    x = x_.reshape([n_tile, -1]).copy()  # tile
    features = np.zeros((len(x), 6))  # 6 features per tile

    for i in range(x.shape[0]):
        row = x[i]

        # Count of each value
        count_1 = np.sum(row == 1)
        count_2 = np.sum(row == 2)
        count_3 = np.sum(row == 3)
        count_4 = np.sum(row == 4)

        # Number of changes
        changes = np.sum(np.diff(row) != 0)

        # Average distance between changes
        if changes < 2:
            avg_dist_between_changes = 0
        else:
            change_indices = np.where(np.diff(row) != 0)[0] + 1
            avg_dist_between_changes = np.mean(np.diff(change_indices))

        # Store the features in the array
        features[i] = [count_1, count_2, count_3, count_4, changes, avg_dist_between_changes]

    tlin = np.arange(len(features[:,0])) * len(x_) // n_tile

    return tlin, features


def get_dataset_features(x_a, n_tile, y):
    """
    Generates and plots features for a given dataset.

    Parameters:
    x_a (list of np.ndarray): List of input arrays.
    n_tile (int): Number of tiles to split each input array into.
    y (pd.Series): Series of target values for coloring the plots.

    Returns:
    np.ndarray: Array of all computed features for further analysis.
    """
    colors = matplotlib.cm.jet(y.values / y.max())
    f, axs = plt.subplots(2, 2, figsize=[10, 8])

    all_features = []

    for j, (xi, c) in enumerate(zip(x_a, colors)):
        tlin, features = make_features(xi, n_tile)
        all_features.append(features.reshape(-1))  # for analysis later

        ax = axs.reshape(-1)
        for i in range(4):
            ax[i].scatter(tlin, features[:, i + 1], color=c, label=f'{y.values[j]}', alpha=0.7, s=10)
            ax[i].set_title(feat_names[i])

        plt.legend()

    return np.array(all_features)


from sklearn.metrics import roc_curve, roc_auc_score

def auc_plot(y_true, y_pred_prob, ax):
    # AUC curve, no uncertainty bars

    # Calculate the ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    
    # Calculate the AUC
    auc = roc_auc_score(y_true, y_pred_proba)
        
    # Plot the ROC curve
    plt.sca(ax)
    plt.plot(fpr, tpr, lw=2, label='ROC curve (AUC = %0.2f)' % auc) #color='blue', 
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')    
    plt.title(f'AUC: {auc:.2f}')

    return auc

def auc_with_ci(pred_out, median=True):
    # total AUC curve with confidence intervals across CV-splits

    model_names = list(pred_out.keys())
    n_splits, _, n_points = np.array(pred_out[model_names[0]]).shape
    
    # To store AUC values for each model and split
    auc_values = {model: [] for model in model_names}
    roc_data = {model: {'fpr': [], 'tpr': []} for model in model_names}
    
    # Compute AUC values and ROC curves for each split
    for model in model_names:
        for i in range(n_splits):
            pred_out_model = np.array(pred_out[model])
            y_test = pred_out_model[i, 0, :]
            y_pred_proba = pred_out_model[i, 1, :]
            auc = roc_auc_score(y_test, y_pred_proba)
            auc_values[model].append(auc)
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            roc_data[model]['fpr'].append(fpr)
            roc_data[model]['tpr'].append(tpr)
    
    # Plotting
    plt.figure(figsize=(10, 8))

    out = []
    
    for model in model_names:

        # x-axis in AUC for mean and median
        fpr_mu = np.linspace(0, 1, n_points)

        # make y-axis        
        tpr_interps = [np.interp(fpr_mu, roc_data[model]['fpr'][i], roc_data[model]['tpr'][i]) for i in range(n_splits)]            
                            
        if median:       
            # statistics
            auc_median = np.percentile(auc_values[model], 50)
            auc_q25 = np.percentile(auc_values[model], 25)
            auc_q75 = np.percentile(auc_values[model], 75)

            # curve
            tpr_median = np.median(tpr_interps,axis=0)
            tpr_lower = np.percentile(tpr_interps, 25, axis=0)
            tpr_upper = np.percentile(tpr_interps, 75, axis=0)
            
            plt.plot(fpr_mu, tpr_median, lw=2, label=f'{model} (AUC = {auc_median:.3f}, q25 = {auc_q25:.3f}, q75 = {auc_q75:.3f})')
            plt.fill_between(fpr_mu, tpr_lower, tpr_upper , alpha=0.2)
            
            tag = '(median, q25, q75)'            
            out.append([auc_median, auc_q25, auc_q75])
            
        else:
            # statistics
            auc_mean = np.mean(auc_values[model])
            auc_std = np.std(auc_values[model])
            tag = '(mean, std)'
            out.append([auc_mean, auc_mean - auc_std, auc_mean + auc_std])

            # curve:
            tpr_mean = np.mean(tpr_interps,axis=0)
            tpr_std = np.std(tpr_interps,axis=0)
            tpr_lower = tpr_mean - tpr_std
            tpr_upper = tpr_mean + tpr_std
            
            plt.plot(fpr_mu, tpr_mean, lw=2, label=f'{model} (AUC = {auc_mean:.3f} +/- {auc_std:.3f}')            
            plt.fill_between(fpr_mu, tpr_lower, tpr_upper , alpha=0.2)
                        
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic {tag}')
    plt.legend(loc="lower right")
    plt.show()

    return out


# 1. Load and prep data

ls '/Users/jannes.jegminat/Library/Mobile Documents/com~apple~CloudDocs/24-mt-sinai/24-Ipek-CPP-project/data/processed'

import csv

# Load the YAML file
with open(Path(CONFIG_PATH) / 'config.yaml', 'r') as file:
    data = yaml.safe_load(file)

# Extract the targets
#targets = data['targets']

#data_file = Path(PROCESSED_PATH) / 'static_XY.pkl' # each df is one subject
#with open(data_file, 'rb') as file:
#    XY = pickle.load(file)

# load subjects with >10 days of data or more
with open(f'{PROCESSED_PATH}/subject_list_with_11d_or_more.csv', 'r') as file:
    reader = csv.reader(file)
    subject_list = list(reader)[0]

dfs = [pd.read_csv(f'{PROCESSED_PATH}/fitbit_1min/{subj_id}_activity_1min.csv') for subj_id in subject_list]

### Spot and remove discontinuouities

for i,(df,sid) in enumerate(zip(dfs, subject_list)):
    len(df)
    jumps = np.where(np.diff(df['time_counter']) != 1)[0]
    if len(jumps)>0:
        print(i, sid, jumps)
        print('remove jump')
        ax=df.reset_index().plot('index','time_counter', title=f'{i},{sid}', label='with jump')        
        left = df.iloc[:jumps[0]]
        right = df.iloc[jumps[0]+1:]
        if len(left) > len(right):
            df = left
        else:
            df = right
        df.reset_index().plot('index','time_counter', title=f'{i},{sid}',ax=ax, color='r', label='without jump')
        plt.show()
        plt.close()

### Make features

feats = []
min_per_bin = 60
for i, df in enumerate(dfs):

    # activity feat
    activity = df.value.values    
    n_tiles = activity.shape[0]/min_per_bin # bin 1h
    feat = make_features(activity[:int(min_per_bin * n_tiles)],int(n_tiles))[1]

    # hour feat
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['hour'] = df['datetime'].dt.hour
    hour_feat = df['hour'].values[min_per_bin*np.arange(int(n_tiles))]

    # subject feature
    subj_feat = np.ones(int(n_tiles)) * i
    
    feats.append( np.c_[hour_feat, feat.copy(), subj_feat] )

plt.title("one subject's activity features")
plt.plot(feat[:,0]), plt.plot(feat[:,1]), plt.plot(feat[:,2]), plt.plot(feat[:,3])

### Predict with fixed look back window

#from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import log_loss, roc_auc_score, brier_score_loss, precision_recall_curve, auc
from sklearn.model_selection import ShuffleSplit #, #LeaveOneOut
from sklearn.ensemble import RandomForestClassifier
# RF, is slightly better
from sklearn.metrics import confusion_matrix
import tqdm
from sklearn.linear_model import LogisticRegression

feat = feats[0]

target1 = (feat[:,1] == 60)
target2 = (feat[:,3] > 0) | (feat[:,4] > 0)
target1.sum(), target2.sum()

np.array(X_tot).shape

feat[t-lbw : t].shape, feat.shape #, np.arange(lbw,len(feat))

np.array(y_tot).shape

np.array(X_tot).reshape(len(X_tot),-1).shape

plt.pcolor(X_tot[0].reshape(lbw,-1)[:,2:])

def my_splits(data_N, n_splits, lbw):
    """ Create a generator for n_splits splits of a dataset of time series data of length data_N.
    All splits, including the last one, will have the same size.
    
    Args:
    data_N (int): The total number of data points.
    n_splits (int): The number of splits to generate.
    lbw (int): The look back window to prevent data leakage.
    
    Yields:
    tuple: (train_index, test_index) for each split.
    """
    split_size = data_N // n_splits
    usable_data_N = split_size * n_splits  # Ensure all splits have equal size
    idx_all = np.arange(usable_data_N)

    for n in range(n_splits):
        # Calculate test indices
        test_start = n * split_size
        test_end = (n + 1) * split_size
        test_index = idx_all[test_start:test_end]

        # Calculate train indices, considering the look back window
        train_index_left = idx_all[:max(0, test_start - lbw)]
        train_index_right = idx_all[min(usable_data_N, test_end + lbw):]
        train_index = np.concatenate([train_index_left, train_index_right])

        yield train_index, test_index

feat[:,[0,2,3,4,5,6,7]]

sum([len(ff) for ff in feats]), len(y_tot)/24



models = {#'day_time':RandomForestClassifier(),
        'hour':None,
          #'lbw24':RandomForestClassifier(),
          'lbw12':RandomForestClassifier(),
          #'lbw6':RandomForestClassifier(),
          #'lbw4':RandomForestClassifier(),
          'lbw2':RandomForestClassifier(),
          'lazy': None,          
          #'no_id':RandomForestClassifier(),          
         }

n_splits = 10

pred_out = {}

for model_name, model in models.items():
    print('start with', model_name)
    pred_out[model_name] = []

    if model_name[:3] == 'lbw':
        lbw = int(model_name[3:]) # extract lookback window
    else:
        lbw = 1
    
    feature_idx = [0,2,3,4,5,6,7] # rm sedentary
    
    # make look back data
    X_tot, y_tot = [], [] 
    for feat in feats: # per subject
        target = feat[:,1] == min_per_bin # 60min
        
        for t in np.arange(lbw,len(feat)): # make tiles
            y_tot.append(target[t])
            X_tot.append(feat[t-lbw : t])
    
    # select features
    X_tot = np.array(X_tot)[:,:,feature_idx].reshape(len(X_tot),-1)
    y_tot = np.array(y_tot)
    
    metrics = [lambda x, y : log_loss(x,y, labels=[True,False]), brier_score_loss]
    
    np.random.seed(0)
    
    split_gen = ShuffleSplit(n_splits=n_splits, test_size=0.1, random_state=42).split(X_tot)
    split_gen = my_splits(data_N=len(X_tot), n_splits=n_splits, lbw=lbw)
    
    out = []
    
    split = 0
    for (train_index, test_index) in tqdm.tqdm(split_gen):
            
        X_train, X_test = X_tot[train_index], X_tot[test_index]    
        y_train, y_test = y_tot[train_index], y_tot[test_index]    

        if model_name == 'lazy':
            activity_feat = [1,2,3]
            N = len(X_test)
            # assume next hour is the same as previous hour
            y_pred_proba = (1 - X_test.reshape((N,lbw,-1))[:,-1,activity_feat].sum(axis=-1)/min_per_bin) # N, LBW, dim; min_per_bin = 60
            y_pred = (X_test.reshape((N,lbw,-1))[:,-1,activity_feat].sum(axis=-1)==0).astype(int)            
        elif model_name == 'hour':
            x_hour = X_train[:,0]
            # contains hour:prob {0: 0.34, 1: 0.56, ect}            
            pred_dict = pd.DataFrame(y_train, index=x_hour).reset_index().groupby('index').mean().to_dict()[0]
            y_pred_proba = np.array([pred_dict[hour] for hour in X_test[:,0]])                        
            y_pred = np.round(y_pred_proba).astype(int)        
        else:
            model.fit(X_train, y_train)        
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:,1]
    
        # save per split
        pred_out[model_name].append([y_test, y_pred_proba])

        out.append({'split':split, 'model':model_name})
        split += 1
        cm = dict(zip(['TN', 'FP', 'FN', 'TP'], list(confusion_matrix(y_test, y_pred, normalize='true').reshape(-1))))        
        out[-1].update(cm)
        #print(out[-1])
        
        # Evaluate
        for metric in metrics: #  = [log_loss, roc_auc_score, brier_score_loss]            
            name = metric.__name__
            out[-1][name] = metric(y_test, y_pred)
        
