import glob
import pickle
import random
import os
import numpy as np

from params import params
from utils import Utils
from torch.utils.data import Dataset, DataLoader
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


class SREDataset(Dataset):
    def __init__(self, audio_data, pho_data):
        self.audio_data = audio_data
        self.pho_data = pho_data

    def __getitem__(self, index):
        audio = self.audio_data[index]
        phoneme = self.pho_data[index]

        # 1. get one second window starting from a random frame (2 consecutive frames for smooth loss)
        start_fr = random.randint(0, audio.shape[0] - params.srnet_win_size - 1)
        audio_feature = audio[start_fr:start_fr + params.srnet_win_size + 1, :]
        pho_feature = phoneme[start_fr:start_fr + params.srnet_win_size + 1, :]

        # 2. Get the number of phoneme during the window
        pps_1, pps_2 = Utils.get_pps_skipping_silence(pho_feature[:-1]), Utils.get_pps_skipping_silence(pho_feature[1:]) # phoneme per second
        #print("pps: {},".format(pps_1))
        pps = np.vstack([pps_1, pps_2])

        # 3. Normalization        

        ds_mean_std = np.load('{}/mean_std/ds_librispeech.npy'.format(params.features_train_path))
        mean, std = ds_mean_std[:params.feature_dim], ds_mean_std[params.feature_dim:]
        audio_feature = (audio_feature - mean) / (std + 1e-6)
        audio_feature = np.nan_to_num(audio_feature)

        #pipeline = Pipeline([('pca', PCA(n_components=200))])
        #audio_feature = pipeline.fit_transform(audio_feature)

        return audio_feature, pps

    def __len__(self):
        return len(self.audio_data)
    

def get_dataloader():

    with open('{}/train_librispeech.pkl'.format(params.feature_path), 'rb') as f:
        audio_features = pickle.load(f)
        phonemes = pickle.load(f)
    train_set = SREDataset(audio_features, phonemes)

    with open('{}/valid_librispeech.pkl'.format(params.feature_path), 'rb') as f:
        audio_features = pickle.load(f)
        phonemes = pickle.load(f)
    valid_set = SREDataset(audio_features, phonemes)

    train_loader = DataLoader(train_set, batch_size=params.batch_size, num_workers=1, shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=params.batch_size, num_workers=1, shuffle=False)

    return train_loader, valid_loader


def get_features(phone_files, features_trainval_path):

    ds_list = []
    pho_list = []

    # Implement PCA (Uncomment)
    #pipeline = Pipeline([('scaling', StandardScaler()), ('pca', PCA(n_components=200))])

    # Get Mean & Std (Uncomment)
    stats = np.ndarray([])

    for i, pho_path in enumerate(phone_files):
        filename = Utils.get_filename(pho_path).split("_phoneme")[0]

        if os.path.isfile("{}/{}.npy".format(features_trainval_path, filename)):
            ds_data = np.load("{}/{}.npy".format(features_trainval_path, filename))
            pho_data = np.load(pho_path)

            # Implement PCA (Uncomment)
            #ds_data = pipeline.fit_transform(ds_data)
            #print(ds_data.shape)

            # Get Mean & Std (Uncomment)
            stats = np.concatenate([stats, ds_data], axis=0) if stats.size > 1 else ds_data 

            ds_data, pho_data = Utils.get_sync_fr(ds_data, pho_data, "ds", "pho")
        
            ds_data = Utils.pad_first_last_fr(ds_data, params.srnet_win_size)
            pho_data = Utils.pad_first_last_fr(pho_data, params.srnet_win_size)

            ds_list.append(ds_data)
            pho_list.append(pho_data)

    # Get Mean & Std (Uncomment)
    mean, std = np.mean(stats, axis=0), np.std(stats, axis=0)
    np.save('{}/mean_std/ds_librispeech.npy'.format(features_trainval_path), np.concatenate([mean, std], axis=0).astype(np.float32))

    return ds_list, pho_list


def make_datafile():

    phone_files = []
    phone_files += glob.glob('{}/*_phoneme.npy'.format(params.phoneme_train_path))
    audio_features, phonemes = get_features(phone_files, params.features_train_path)
    print(np.array(audio_features).shape)

    # with open('{}/train_librispeech.pkl'.format(params.feature_path), 'wb') as f:
    #     pickle.dump(audio_features, f)
    #     pickle.dump(phonemes, f)

    # phone_files = []
    # phone_files += glob.glob('{}/*_phoneme.npy'.format(params.phoneme_valid_path))
    # audio_features, phonemes = get_features(phone_files, params.features_valid_path)

    # with open('{}/valid_librispeech.pkl'.format(params.feature_path), 'wb') as f:
    #     pickle.dump(audio_features, f)
    #     pickle.dump(phonemes, f)

if __name__ == '__main__':
    make_datafile()
    