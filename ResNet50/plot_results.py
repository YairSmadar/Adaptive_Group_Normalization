import torch
import numpy as np
import pandas as pd
from IPython.display import display, HTML
import glob
import matplotlib.pyplot as plt
from numpy.ma.core import append

from argparse import ArgumentParser

# MODELS_LOC = "/content/drive/MyDrive/AGN/saved_models/"
MODELS_LOC = r'C:\a_work\CS2\Adaptive-Norm\latest-python-code\exp\\'


parser = ArgumentParser(description='plot results')
parser.add_argument('--models_dir_path', default=MODELS_LOC, type=str)
args = parser.parse_args()

data = {'description': [],
        'method': [],
        'lr': [],
        'top1 max': [],
        'argmax': [],
        'top1 last': [],
        'top1 last average 5': [],
        'current length': [],
        }
df = pd.DataFrame(data=data)

top1s, losses, methods, descriptions = [], [], [], []
for file in glob.glob(args.models_dir_path + '**/*.tar', recursive=True):

    if '\\' in file:
        file_array_of_names = file.split('\\')[-1].split('_')
    elif '/' in file:
        file_array_of_names = file.split('/')[-1].split('_')
    else:
        file_array_of_names = file.split('_')

    method = file_array_of_names[1] + file_array_of_names[5]
    if 'GN' in method and not 'SGN' in method and not 'RGN' in method:
        method = 'GN'
    method += ' ' + file_array_of_names[5]
    lr = file_array_of_names[3][:-2]
    method += ' lr ' + lr
    riar = file_array_of_names[6][:1]
    method += ' riar ' + riar

    far_groups = file_array_of_names[-1]

    if "far_groups" in far_groups:
        method += ' far_groups'

    loaded_model = torch.load(file, map_location=torch.device('cpu'))
    top1_max = max(loaded_model['test_prcition1'])
    argmax = np.argmax(loaded_model['test_prcition1'])
    top1_last = loaded_model['test_prcition1'][-1]
    average_5 = np.mean(loaded_model['test_prcition1'][-5:])
    current_length = len(loaded_model['test_prcition1'])

    top1s.append(loaded_model['test_prcition1'])
    losses.append(loaded_model['test_losses'])
    methods.append(method)

    append_list = [file.split('/')[-1], method, lr, top1_max, argmax, top1_last, average_5, current_length]
    df = df.append(pd.Series(append_list, index=df.columns), ignore_index=True)

df.to_csv(MODELS_LOC + 'df.csv')

[plt.plot(top1) for top1 in top1s]
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(methods)
plt.savefig(MODELS_LOC + 'accuracy.png')
plt.show()
plt.close()

[plt.plot(loss) for loss in losses]
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(methods)
plt.savefig(MODELS_LOC + 'losses.png')
plt.show()
plt.close()
