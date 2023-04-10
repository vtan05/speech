import librosa
import PyOctaveBand
import scipy.io.wavfile
import numpy as np
import matplotlib.pyplot as plt

from scipy import signal
from scipy.signal import hilbert
from scipy.stats import skew, kurtosis
from params import params

def mfcc(y, sr):

    window = signal.windows.hamming(params.hamm_length)

    # Mel-Spectrogram
    S = librosa.feature.melspectrogram(y=y, sr=sr, hop_length=params.hop_length, window=window, 
                                       win_length=params.hamm_length)
    S_dB = librosa.power_to_db(S, ref=np.max)

    # MFCC + Deltas (n_mfcc = 13)
    mfcc = librosa.feature.mfcc(S=S_dB, n_mfcc=params.nmels)
    mfcc_delta = librosa.feature.delta(mfcc)
    mfcc_delta2 = librosa.feature.delta(mfcc_delta, order=2)

    # Compute MFCC Statistics
    mfcc_stats = []

    for j in range(params.nmels):
        mfcc_stats.append(np.mean(mfcc[j]))
        mfcc_stats.append(np.mean(mfcc_delta[j]))
        mfcc_stats.append(np.mean(mfcc_delta2[j]))

        mfcc_stats.append(np.std(mfcc[j]))
        mfcc_stats.append(np.std(mfcc_delta[j]))
        mfcc_stats.append(np.std(mfcc_delta2[j]))

        mfcc_stats.append(np.max(mfcc[j]))
        mfcc_stats.append(np.max(mfcc_delta[j]))
        mfcc_stats.append(np.max(mfcc_delta2[j]))

        mfcc_stats.append(skew(mfcc[j]))
        mfcc_stats.append(skew(mfcc_delta[j]))
        mfcc_stats.append(skew(mfcc_delta2[j]))

        mfcc_stats.append(kurtosis(mfcc[j]))
        mfcc_stats.append(kurtosis(mfcc_delta[j]))
        mfcc_stats.append(kurtosis(mfcc_delta2[j]))

        mfcc_stats.append(np.mean(np.absolute(mfcc[j] - np.mean(mfcc[j]))))
        mfcc_stats.append(np.mean(np.absolute(mfcc_delta[j] - np.mean(mfcc_delta[j]))))
        mfcc_stats.append(np.mean(np.absolute(mfcc_delta2[j] - np.mean(mfcc_delta2[j]))))

    return mfcc_stats

if __name__ == '__main__':
    y, sr = librosa.load('test.wav', sr=params.sampling_rate)

    # Octave spectra and bands in time domain
    spl, freq, xb = PyOctaveBand.octavefilter(y, sr, order=9, show=1, sigbands=1)
    butter_filter = 

    # Store signal in bands in separated wav files
    for idx in range(len(freq)):
        #scipy.io.wavfile.write("test_"+str(round(freq[idx]))+"_Hz.wav", sr, xb[idx]/np.max(xb[idx]))
        signal = xb[idx]/np.max(xb[idx])
        hilbert_trans = hilbert(signal)
        amplitude_envelope = np.abs(hilbert_trans)

        plt.plot(amplitude_envelope)
        plt.show()



    # windows = []
    # for i in range(0, len(y) - params.window_length, params.window_shift):
    #     sig = y[i:i + params.window_length]
    #     mfcc_stats = mfcc(sig, sr)
    #     windows.append(mfcc_stats)

    # print(np.array(windows).shape)