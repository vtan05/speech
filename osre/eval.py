import sys, os, torch, librosa, glob
import numpy as np
from params import params
from train import Runner
from utils import Utils
from scipy.stats import pearsonr
import matplotlib.pyplot as plt


def eval(pho_path, plot=True):

    filename = Utils.get_filename(pho_path).split("_phoneme")[0]

    with torch.no_grad():
        runner = Runner(params)
        runner.model.load_state_dict(torch.load(params.model_path + 'model_{}.pth'.format(params.model))['model_state_dict'])

        # ds normalize
        feature = np.load("{}/{}.npy".format(params.features_valid_path, filename))
        ds_mean_std = np.load('{}/mean_std/ds_librispeech.npy'.format(params.features_valid_path))
        mean, std = ds_mean_std[:params.feature_dim], ds_mean_std[params.feature_dim:]
        feature = (feature - mean) / (std + 1e-6)

        # frame to window
        feature_speed = Utils.pad_first_last_fr(feature, params.srnet_win_size)
        feature_speed = Utils.frame_to_window(feature_speed, params.srnet_win_size)
        
        feature_speed = torch.from_numpy(feature_speed).float().to(runner.device)
        feature_speed = feature_speed.view(feature_speed.size(0), feature_speed.size(1), feature_speed.size(2))

        # Inference
        pred_pps = runner.model(feature_speed)
        pred_pps = pred_pps.cpu().numpy()

        # ------------------------------------------------------------------------ #
        # Test 1) Resulting speed graph
        # ------------------------------------------------------------------------ #
        np.savetxt('{}{}\{}.csv'.format(params.eval_path, params.model, filename), pred_pps, delimiter=",")
        # for fr in range(pred_pps.shape[0]):
        #     print("{} frame: {}".format(fr, pred_pps[fr]))
        # print("mean: {}".format(pred_pps.mean())) #regression

        if plot:
            x = np.arange(0, pred_pps.shape[0], 1)
            plt.figure(figsize=(12, 4))
            plt.plot(x, pred_pps, 'bo', label='librispeech', linewidth=0.001)
            plt.xlabel('frame')
            plt.ylabel('Predicted PPS (Phonemes Per Second)')
            plt.legend()
            plt.ylim(0.0, 20.0)
            #plt.show()
            plt.savefig('{}{}\{}.png'.format(params.eval_path, params.model, filename))
            plt.close()


        # ------------------------------------------------------------------------ #
        # Test 2) Pearson Coeff.
        # ------------------------------------------------------------------------ #
        print('Calculating Pearson Coefficient.....')

        # PPS
        gt_phoneme = np.load(pho_path)
        pred_pps, gt_phoneme = Utils.get_sync_fr(pred_pps, gt_phoneme, "pred", "gt")

        # frame to window
        gt_phoneme = Utils.pad_first_last_fr(gt_phoneme, params.srnet_win_size)
        gt_phoneme = Utils.frame_to_window(gt_phoneme, params.srnet_win_size)

        gt = []
        for i in range(gt_phoneme.shape[0]):
            gt_i = Utils.get_pps_skipping_silence(gt_phoneme[i])
            gt.append(gt_i)

        test_corr, _ = pearsonr(np.squeeze(pred_pps), np.array(gt)) # return: correlation, p-value
        #print(test_corr)
    
    return test_corr


if __name__ == '__main__':
    phone_files = []
    phone_files += glob.glob('{}/*_phoneme.npy'.format(params.phoneme_valid_path))

    #speed_array = []
    corr_array = []
    for i, pho_path in enumerate(phone_files):
        corr = eval(pho_path)
        #speed_array.append(speed)
        corr_array.append(corr)

    print('Pearson Coefficient Avg = ' + str(np.mean(np.array(corr_array))))

        

