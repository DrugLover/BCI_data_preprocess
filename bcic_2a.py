import mne
from mne.io import read_raw_gdf
import numpy as np
import os

base_path = r"E:\datasets\bci_competition\BCICIV_2a\BCICIV_2a_gdf\A0"
subjects = [i for i in range(1, 10)]

path = r"E:\datasets\bci_competition\BCICIV_2a\BCICIV_2a_gdf\A09T.gdf"
raw = read_raw_gdf(path, preload=True)

def load_gdf(path, eog=False):
    '''
    :param path: gdf_path
    :param eog: keep eog or not
    :return: eeg data->x and label->y
    '''
    raw = read_raw_gdf(path, preload=True)
    raw.filter(4., 36., fir_design='firwin')
    events, event_dict = mne.events_from_annotations(raw)
    if eog:
        exclude = 'bads'
    else:
        exclude = ['bads', 'EOG-left', 'EOG-central', 'EOG-right']
    picks = mne.pick_types(raw.info, meg=False, eeg=True, eog=False, stim=False,
                           exclude=exclude)
    tmin, tmax = 0., 4.

    # left_hand = 769, right_hand = 770, feet = 771, tongue = 772
    # 若要2分类在dict里改就行
    if event_dict['772'] == 8:
        event_id = {'769': 5, '770': 6, '771': 7, '772': 8}
    else:
        event_id = {'769': 7, '770': 8, '771': 9, '772': 10}
    epochs = mne.Epochs(raw, events, event_id, tmin, tmax, proj=True, picks=picks,
                        baseline=None, preload=True)
    x = epochs.get_data()
    if event_dict['772'] == 8:
        y = epochs.events[:, -1] - 4
    else:
        y = epochs.events[:, -1] - 6


    # transfrom y to one-hot code
    # import tensorflow as tf
    # y = tf.keras.utils.to_categorical(y)

    return x[:1000], y
# x, y = load_gdf(path)

subject_path = r"E:\datasets\bci_competition\BCICIV_2a\BCICIV_2a_gdf"
base_path = r"E:\datasets\bci_competition\BCICIV_2a\bci2a_npy_4s_4_36"
label_path = r"E:\datasets\bci_competition\BCICIV_2a\true_labels"
os.mkdir(base_path)
xs = []
ys = []

# for subject in subjects:
#
#     path = subject_path + "\\A0" + str(subject) + 'T.gdf'
#     x, y = load_gdf(path, eog=True)
#     xs.append(x)
#     ys.append(y)
#
#     x_path = "subject" + str(subject) + "_data"
#     y_path = "subject" + str(subject) + "_label"
#     np.save(os.path.join(base_path, x_path), x)
#     np.save(os.path.join(base_path, y_path), y)
#
# xs = np.array(xs)
# ys = np.array(ys)
# np.save(os.path.join(base_path, 'xs'), xs)
# np.save(os.path.join(base_path, 'ys'), ys)