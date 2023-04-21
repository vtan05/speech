import os, torch, librosa, glob
import numpy as np
from math import ceil
import torch.nn.functional as F
from sklearn.decomposition import PCA
from params import params


class Utils:

    @staticmethod
    def get_filename(file_path):
        return os.path.basename(os.path.splitext(file_path)[0])

    @staticmethod
    def get_sync_fr(a, b, a_name='bs', b_name='ds'):
        #print("[Before Sync] {}: {}, {}: {}, diff: {}".format(a_name, a.shape, b_name, b.shape, a.shape[0] - b.shape[0]))
        if a.shape[0] > b.shape[0]:
            for i in range(a.shape[0] - b.shape[0]):
                a = np.delete(a, -1, 0)
        elif a.shape[0] < b.shape[0]:
            for i in range(b.shape[0] - a.shape[0]):
                b = np.delete(b, -1, 0)
        return a, b

    @staticmethod
    def pad_first_last_fr(data, win_size): # Data manager
        padded_data = data
        half_size = int(win_size/2)
        temp = np.repeat([data[0]], half_size, axis=0)
        padded_data = np.insert(padded_data, 0, temp, axis=0)
        temp = np.repeat([data[-1]], half_size-1, axis=0)
        padded_data = np.insert(padded_data, -1, temp, axis=0)
        return padded_data

    @staticmethod
    def get_pps_skipping_silence(phonemes): # phoneme per second / phoneme index : 0 ~ 51 (total number: 52)
        fps = params.fps
        n_frame = phonemes.shape[0]
        prev_p = phonemes[0,0]
        cnt = 0 if prev_p == 51 else 1
        for i in range(1, n_frame):
            if phonemes[i, 0] != prev_p and phonemes[i, 0] != 51:
                cnt += 1
                prev_p = phonemes[i, 0]
        return cnt / (n_frame/fps)

    @staticmethod
    def frame_to_window(data, win_size):
        window_data = np.array([])
        for fr in range(data.shape[0] -(win_size-1)):
            if window_data.size != 0:
                window_data = np.vstack([window_data, np.array([data[fr:fr+win_size, :]])])
            else:
                window_data = np.array([data[fr:fr+win_size, :]])
        return window_data

    @staticmethod
    def exponential_smoothing(data, alpha=0.2): #data: [fr, dim]
        for i in range(data.shape[0]):
            if i == 0:
                continue
            data[i] = ((1-alpha) * data[i-1]) + (alpha * data[i])
        return data
