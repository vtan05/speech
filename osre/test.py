import sys, os, torch, librosa
import numpy as np
from params import params
from train import Runner
from utils import Utils
from scipy.stats import pearsonr


def main():
    # make path
    if os.path.isdir('{}'.format(hparams.test_path)) == False:
        os.mkdir('{}'.format(hparams.test_path))


    with torch.no_grad():
        runner = Runner(params)
        runner.model.load_state_dict(torch.load(params.model_path)['model_state_dict'])

        # Deepspeech
        if os.path.isfile('{}/data/deepspeech/feature/{}.npy'.format(hparams.root, hparams.test_audio_name)):
            feature = np.load('{}/data/deepspeech/feature/{}.npy'.format(hparams.root, hparams.test_audio_name))
        else:
            feature = get_ds_feature('{}/data/test_audio/{}/{}.wav'.format(hparams.root, hparams.test_audio_folder, hparams.test_audio_name))
            np.save('{}/data/deepspeech/feature/{}.npy'.format(hparams.root, hparams.test_audio_name), feature)

        # # ds normalize
        ds_mean_std = np.load('{}/mean_std/ds_librispeech.npy'.format(hparams.feature_path))
        mean, std = ds_mean_std[:params.ds_dim], ds_mean_std[params.ds_dim:]
        feature = (feature - mean) / (std + 1e-6)

        # frame to window
        feature = Utils.pad_first_last_fr(feature, params.srnet_win_size)
        feature = Utils.frame_to_window(feature, params.srnet_win_size)
        
        feature = torch.from_numpy(feature).float().to(runner.device)
        feature = feature.view(feature.size(0), 1, feature.size(1), feature.size(2))

        # Inference
        pred_pps = runner.model(feature)
        pred_pps = pred_pps.cpu().numpy()

        # ------------------------------------------------------------------------ #
        # Test 1) Resulting speed graph
        # ------------------------------------------------------------------------ #
        # np.savetxt('{}/{}/{}.csv'.format(hparams.test_path, hparams.model, hparams.test_audio_name), pred_pps, delimiter=",")
        # for fr in range(pred_pps.shape[0]):
        #     print("{} frame: {}".format(fr, pred_pps[fr]))
        # print("mean: {}".format(pred_pps.mean())) #regression


        # ------------------------------------------------------------------------ #
        # Test 2) Pearson Coeff.
        # ------------------------------------------------------------------------ #
        print('Calculating Pearson Coefficient.....')
        gt_phoneme = np.load('{}/data/test_audio/{}/{}_phoneme.npy'.format(hparams.root, hparams.test_audio_folder, hparams.test_audio_name))
        pred_pps, gt_phoneme = Utils.get_sync_fr(pred_pps, gt_phoneme, "pred", "gt")

        # frame to window
        gt_phoneme = Utils.pad_first_last_fr(gt_phoneme, params.srnet_win_size)
        gt_phoneme = Utils.frame_to_window(gt_phoneme, params.srnet_win_size)

        gt = []
        for i in range(gt_phoneme.shape[0]):
            gt_i = Utils.get_pps(gt_phoneme[i])
            gt.append(gt_i)

        test_corr, _ = pearsonr(np.squeeze(pred_pps), np.array(gt)) # return: correlation, p-value
        print(test_corr)


if __name__ == '__main__':
    main()
