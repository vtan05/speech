import sys, os, torch, librosa, glob
import numpy as np
from params import params
from train import Runner
from utils import Utils
from scipy.stats import pearsonr


def eval(pho_path):

    filename = Utils.get_filename(pho_path).split("_phoneme")[0]

    with torch.no_grad():
        runner = Runner(params)
        runner.model.load_state_dict(torch.load(params.model_path + 'model.pth')['model_state_dict'])

        # ds normalize
        feature = np.load("{}/{}.npy".format(params.features_valid_path, filename))
        ds_mean_std = np.load('{}/mean_std/ds_librispeech.npy'.format(params.features_valid_path))
        mean, std = ds_mean_std[:params.feature_dim], ds_mean_std[params.feature_dim:]
        feature = (feature - mean) / (std + 1e-6)

        # frame to window
        feature = Utils.pad_first_last_fr(feature, params.srnet_win_size)
        feature = Utils.frame_to_window(feature, params.srnet_win_size)
        
        feature = torch.from_numpy(feature).float().to(runner.device)
        feature = feature.view(feature.size(0), feature.size(1), feature.size(2))

        # Inference
        pred_pps = runner.model(feature)
        pred_pps = pred_pps.cpu().numpy()

        # ------------------------------------------------------------------------ #
        # Test 1) Resulting speed graph
        # ------------------------------------------------------------------------ #
        np.savetxt('{}{}\{}.csv'.format(params.eval_path, '100_epoch', filename), pred_pps, delimiter=",")
        # for fr in range(pred_pps.shape[0]):
        #     print("{} frame: {}".format(fr, pred_pps[fr]))
        # print("mean: {}".format(pred_pps.mean())) #regression


        # ------------------------------------------------------------------------ #
        # Test 2) Pearson Coeff.
        # ------------------------------------------------------------------------ #
        #print('Calculating Pearson Coefficient.....')
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
    
    return pred_pps.mean(), test_corr


if __name__ == '__main__':
    phone_files = []
    phone_files += glob.glob('{}/*_phoneme.npy'.format(params.phoneme_valid_path))

    speed_array = []
    corr_array = []
    for i, pho_path in enumerate(phone_files):
        speed, corr = eval(pho_path)
        speed_array.append(speed)
        corr_array.append(corr)

    print('Speed = ' + str(np.mean(np.array(speed_array))))
    print('Pearson Coeff = ' + str(np.mean(np.array(corr_array))))

        

