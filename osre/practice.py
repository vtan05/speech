import librosa
import PyOctaveBand
import scipy.io.wavfile
import numpy as np
import matplotlib.pyplot as plt

from scipy import signal
from scipy.signal import hilbert
from scipy.stats import skew, kurtosis
from params import params
from dft import goertzel

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


#def ems():
    
if __name__ == '__main__':
    y, sr = librosa.load('test.wav', sr=params.sampling_rate)

    # Octave spectra and bands in time domain
    spl, freq, xb = PyOctaveBand.octavefilter(y, sr, order=8, show=0, sigbands=1)
    print(len(freq))

    # Butterworth Filter (4th order, 30Hz cut-off freq)
    butter_filter = signal.butter(4, 2 * np.pi * 30, 'low', fs=sr, output='sos')

    ems_stats = []
    # Store signal in bands in separated wav files
    for idx in range(len(freq)):
        #scipy.io.wavfile.write("test_"+str(round(freq[idx]))+"_Hz.wav", sr, xb[idx]/np.max(xb[idx]))
        signal = xb[idx]/np.max(xb[idx])

        hilbert_trans = hilbert(signal)
        analytic_sig = np.abs(signal + hilbert_trans)

        imp = scipy.signal.unit_impulse(len(analytic_sig))
        filtd = scipy.signal.sosfilt(butter_filter, imp)

        env = filtd * analytic_sig
        env = env - np.mean(env)

        # Peak frequency and amplitude (0 - 10 Hz)
        freqs, results = goertzel(env, sr, (0, 10))
        peak_amp = np.max(np.array(results)[:,2])
        ems_stats.append(peak_amp)

        index = np.array(results)[:,2].argmax()
        peak_freq = freqs[index]
        ems_stats.append(peak_freq)
        
        # Energy from 3 - 6 Hz
        freqs, results = goertzel(env, sr, (3, 6))
        energy_36 = np.sum(np.abs(np.array(results)[:,2]))
        ems_stats.append(energy_36)

        # Energy from 0 - 4 Hz
        freqs, results = goertzel(env, sr, (0, 4))
        energy_04 = np.sum(np.abs(np.array(results)[:,2]))
        ems_stats.append(energy_04)

        # Energy from 4 - 10 Hz
        freqs, results = goertzel(env, sr, (4, 10))
        energy_410 = np.sum(np.abs(np.array(results)[:,2]))
        ems_stats.append(energy_410)

        # Energy Ratio between 0 - 4 Hz and 4 - 10 Hz
        energy_ratio = energy_04 / energy_410
        ems_stats.append(energy_ratio)

    print(np.array(ems_stats).shape)

    # windows = []
    # for i in range(0, len(y) - params.window_length, params.window_shift):
    #     sig = y[i:i + params.window_length]
    #     mfcc_stats = mfcc(sig, sr)
    #     windows.append(mfcc_stats)

    # print(np.array(windows).shape)