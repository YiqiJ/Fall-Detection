import os

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def load_label(path):
    name2acts = dict()
    unique_act = set()
    with open(path) as f:
        for line in f.readlines():
            line = line.strip().split(',')
            name = line[0]
            acts = line[1:]
            acts = [act.split(' ') for act in acts]
            for act in acts:
                unique_act.add(act[0])
            name2acts[name] = acts
        act2id = dict()
        for act in unique_act:
            if act == 'lay': continue
            act2id[act] = len(act2id)
        act2id['lay'] = len(act2id)
    return name2acts, act2id

def save_label2idx(act2id, path):
    with open(os.path.join(path, 'act2id.txt'), 'w+') as f:
        for (act, idx) in act2id.items():
            f.write('{}:{}\n'.format(act, idx))
    print ("label to idx dict saved")

def label_data(path, folders, label_path):
    name2acts, act2id = load_label(label_path)
    save_label2idx(act2id, path)
    results = []
    for folder in folders:
        cur_path = os.path.join(path, folder)
        for file_name in os.listdir(cur_path):
            cur_file_path = os.path.join(cur_path, file_name)
            name = folder + '_' + file_name.split('.')[0]
            with open(cur_file_path) as f:
                data = pd.read_csv(cur_file_path, sep = ' ')
                acts = name2acts[name]
                data['label'] = np.nan
                for (act, frame) in acts:
                    frame = int(min(int(frame), len(data) - 1))
                    data.loc[[frame], ['label']] = act2id[act]
                data.fillna(method='bfill', inplace = True)
                data.to_csv(os.path.join(path, name + '.txt'), index = None, header = None)

def plot_data(path):
    for file_name in os.listdir(path):
        if len(file_name.split('_')) == 3:
            file_path = os.path.join(path, file_name)
            data = pd.read_csv(file_path, sep = ',', header = None)
            data = data[data[90] != 7]
            x = data.values
            pca = PCA(n_components = 1)
            pca.fit(x)
            x = pca.transform(x)
            fig = plt.figure()
            
            plt.title(file_name.split('.')[0])
            plt.plot(range(len(x)), x)
            fig.savefig(os.path.join('fig', file_name.split('.')[0] + '.png'))
            break

def read_data(path, num_components):
    results = []
    cnts = 0
    for file_name in os.listdir(path):
        if len(file_name.split('_')) == 3:
            file_path = os.path.join(path, file_name)
            data = pd.read_csv(file_path, sep = ',', header = None)
            data = data[data[90] != 7]
            x = data.values[:,:-1]
            y = data.values[:,-1]
            pos = [2, 4, 5, 6]
            ix = np.isin(y, pos)
            y[:] = 0
            y[ix] = 1
            pca = PCA(n_components = num_components)
            pca.fit(x)
            x = pca.transform(x)
            #if file_name.split('.')[0].split('_')[2] == '3':
                #if file_name.split('.')[0].split('_')[1] == '6':
            cnts += len(x) 
            results.append((x,y))
            
    return results
if __name__ == '__main__':
    """
    folders = ['1', '2', '3', '4', '5', '6']
    label_path = 'data/label.txt'
    label_data(path, folders, label_path)
    """
    path = 'data'
    #plot_data(path)
    data = read_data(path, 10)
