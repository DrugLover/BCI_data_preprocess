from physionet_lab import Utils
import numpy as np
import warnings
import os

def label_transform(y):
    # y -> label
    # Left hand:1, Right hand:2, Foot:3, L+R hands:4
    new_y = [0 for i in range(len(y))]
    for i in range(len(y)):
        if y[i] == 'L':
            new_y[i] = 0
        elif y[i] == 'R':
            new_y[i] = 1
        elif y[i] == 'F':
            new_y[i] = 2
        elif y[i] == 'LR':
            new_y[i] = 3
    return new_y

warnings.filterwarnings('ignore')
exclude = [38, 88, 89, 92, 100, 104] #6个人的数据有问题
subjects = [n for n in np.arange(1, 110) if n not in exclude]
runs = [4, 6, 8, 10, 12, 14]
data_path = "E:/datasets/PhysioNet_MI_Dataset"
# x, y = Utils.epoch(Utils.eeg_settings(Utils.del_annotations(Utils.concatenate_runs(
#             Utils.load_data(subjects=subjects, runs=runs, data_path=data_path)))), exclude_base=True)



labels = ['L', 'R', 'F', 'LR']
cnt = [0, 0, 0, 0]
new_path = 'data/physionet'
for subject in subjects:
    subject_path = os.path.join(new_path, 'subject'+str(subject))
    print(subject_path)
    os.mkdir(subject_path)
    # subject_path = 'data/test'
    for label in labels:
        label_path = os.path.join(subject_path, label)
        os.mkdir(label_path)
    x, y = Utils.epoch(Utils.eeg_settings(Utils.del_annotations(Utils.concatenate_runs(
                Utils.load_data(subjects=[subject], runs=runs, data_path=data_path)))), exclude_base=True)
    x = x.reshape(x.shape[0], 1, x.shape[1], x.shape[2])
    x = x[:, :, :, :640]
    # print(x.shape)
    y = label_transform(y)
    for i in range(len(y)):
        cnt[y[i]] += 1
        file_path = os.path.join(subject_path, labels[y[i]], str(cnt[y[i]])+'.npy')
        np.save(file_path, x[i])
