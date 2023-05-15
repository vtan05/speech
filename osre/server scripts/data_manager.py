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
        start_fr = random.randint(0, audio.shape[0] - params.srnet_win_size - 0)
        audio_feature = audio[start_fr:start_fr + params.srnet_win_size + 0, :]
        pho_feature = phoneme[start_fr:start_fr + params.srnet_win_size + 0, :]

        # 2. Get the number of phoneme during the window
        pps_1, pps_2 = Utils.get_pps_skipping_silence(pho_feature[:-1]), Utils.get_pps_skipping_silence(pho_feature[1:]) # phoneme per second
        pps = np.vstack([pps_1, pps_2])

        # 3. Normalization        
        ds_mean_std = np.load('{}/mean_std/train_{}.npy'.format(params.features_path, params.data))
        mean, std = ds_mean_std[:params.feature_dim], ds_mean_std[params.feature_dim:]
        audio_feature = (audio_feature - mean) / (std + 1e-6)
        audio_feature = np.nan_to_num(audio_feature)

        return audio_feature, pps

    def __len__(self):
        return len(self.audio_data)
    

def get_dataloader():

    with open('{}/train_{}.pkl'.format(params.features_path, params.data), 'rb') as f:
        audio_features = pickle.load(f)
        phonemes = pickle.load(f)
    train_set = SREDataset(audio_features, phonemes)

    with open('{}/valid_{}.pkl'.format(params.features_path, params.data), 'rb') as f:
        audio_features = pickle.load(f)
        phonemes = pickle.load(f)
    valid_set = SREDataset(audio_features, phonemes)

    train_loader = DataLoader(train_set, batch_size=params.batch_size, num_workers=16, shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=params.batch_size, num_workers=16, shuffle=False)

    return train_loader, valid_loader


def get_features(filename, data, path):

    ds_data = np.load("{}/{}.npy".format(path, filename))
    pho_data = data[filename]["phoneme"] 

    ds_data, pho_data = Utils.get_sync_fr(ds_data, pho_data, "ds", "pho")
        
    if ds_data.size != 0:
        ds_data = Utils.pad_first_last_fr(ds_data, params.srnet_win_size)
        pho_data = Utils.pad_first_last_fr(pho_data, params.srnet_win_size)

    return ds_data, pho_data


def make_datafile():

    # Training Set for LibriMEAD
    stats = np.ndarray([])
    audio_features = []
    phonemes = []

    print("Processing MEAD Data ")
    for i in range(1,10):
        print("Processing Training Data " + str(i))
        with open('{}/mead_train_'.format(params.pickle_path) + str(i) + '.pkl', 'rb') as f:
            data = pickle.load(f)

        for ff in data:
            audio, phone = get_features(ff, data, params.mead_train_path)
            
            if audio.size != 0: 
                stats = np.concatenate([stats, audio], axis=0) if stats.size > 1 else audio 
                audio_features.append(audio)
                phonemes.append(phone)

    print("Processing Librispeech Data ")
    phone_files = []
    phone_files += glob.glob('{}/*_phoneme.npy'.format(params.phoneme_train_path))
    
    for i, pho_path in enumerate(phone_files):
        filename = Utils.get_filename(pho_path).split("_phoneme")[0]

        if os.path.isfile("{}/{}.npy".format(params.librispeech_train_path, filename)):
            ds_data = np.load("{}/{}.npy".format(params.librispeech_train_path, filename))
            pho_data = np.load(pho_path)

            ds_data, pho_data = Utils.get_sync_fr(ds_data, pho_data, "ds", "pho")
            ds_data = Utils.pad_first_last_fr(ds_data, params.srnet_win_size)
            pho_data = Utils.pad_first_last_fr(pho_data, params.srnet_win_size)

            stats = np.concatenate([stats, ds_data], axis=0) if stats.size > 1 else ds_data 
            audio_features.append(ds_data)
            phonemes.append(pho_data)


    mean, std = np.mean(stats, axis=0), np.std(stats, axis=0)
    np.save('{}mean_std/train_{}.npy'.format(params.features_path[:-1], params.data), np.concatenate([mean, std], axis=0).astype(np.float32))

    with open('{}/train_{}.pkl'.format(params.features_path, params.data), 'wb') as f:
        pickle.dump(audio_features, f)
        pickle.dump(phonemes, f)


#     # Validation Set for LibriMEAD
#     print("Processing Validation Data")
#     stats = np.ndarray([])
#     audio_features = []
#     phonemes = []
#     print("Processing MEAD Data ")
#     with open('{}/mead_valid_1.pkl'.format(params.pickle_path), 'rb') as f:
#         data = pickle.load(f)

#     for ff in data:
#         audio, phone = get_features(ff, data, params.mead_valid_path)
                
#         if audio.size != 0: 
#             stats = np.concatenate([stats, audio], axis=0) if stats.size > 1 else audio 
#             audio_features.append(audio)
#             phonemes.append(phone)

#     print("Processing Librispeech Data ")
#     phone_files = []
#     phone_files += glob.glob('{}/*_phoneme.npy'.format(params.phoneme_valid_path))
    
#     for i, pho_path in enumerate(phone_files):
#         filename = Utils.get_filename(pho_path).split("_phoneme")[0]

#         if os.path.isfile("{}/{}.npy".format(params.librispeech_valid_path, filename)):
#             ds_data = np.load("{}/{}.npy".format(params.librispeech_valid_path, filename))
#             pho_data = np.load(pho_path)

#             ds_data, pho_data = Utils.get_sync_fr(ds_data, pho_data, "ds", "pho")
#             ds_data = Utils.pad_first_last_fr(ds_data, params.srnet_win_size)
#             pho_data = Utils.pad_first_last_fr(pho_data, params.srnet_win_size)

#             stats = np.concatenate([stats, ds_data], axis=0) if stats.size > 1 else ds_data 
#             audio_features.append(ds_data)
#             phonemes.append(pho_data)

#     mean, std = np.mean(stats, axis=0), np.std(stats, axis=0)
#     np.save('{}mean_std/valid_librimead.npy'.format(params.features_path[:-1]), np.concatenate([mean, std], axis=0).astype(np.float32))

#     with open('{}/valid_librimead.pkl'.format(params.features_path), 'wb') as f:
#         pickle.dump(audio_features, f)
#         pickle.dump(phonemes, f)


    # # Test Set for MEAD 
    # print("Processing Test Data")
    # stats = np.ndarray([])
    # audio_features = []
    # phonemes = []
    # with open('{}/mead_origin_valid.pkl'.format(params.pickle_path), 'rb') as f:
    #     data = pickle.load(f)

    # for ff in data:
    #     audio, phone = get_features(ff, data, params.features_test_path)
                
    #     if audio.size != 0: 
    #         # Get Mean & Std (Uncomment)
    #         stats = np.concatenate([stats, audio], axis=0) if stats.size > 1 else audio 

    #         audio_features.append(audio)
    #         phonemes.append(phone)

    # mean, std = np.mean(stats, axis=0), np.std(stats, axis=0)
    # np.save('{}/mean_std/ds_mead.npy'.format(params.features_test_path[:-1]), np.concatenate([mean, std], axis=0).astype(np.float32))

    # with open('{}/test_mead.pkl'.format(params.features_path), 'wb') as f:
    #     pickle.dump(audio_features, f)
    #     pickle.dump(phonemes, f)


if __name__ == '__main__':
    make_datafile()
    