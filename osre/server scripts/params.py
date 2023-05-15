class Params(object):

    def __init__(self):

        # Audio Parameters
        self.fps = 60.0
        self.srnet_win_size = int(1 * self.fps)
        self.sampling_rate = 16000
        self.window_length = int(1 * self.sampling_rate)
        self.window_shift = int((self.sampling_rate / self.fps) - self.fps - 1)
        self.hop_length = int(0.010 * self.sampling_rate)
        self.hamm_length = int(0.020 * self.sampling_rate)
        self.nmels = 13
        
        # Network Parameters
        self.device = 1
        self.data = 'librimead'
        self.model = 'librimead'
        self.batch_size = 64
        self.learning_rate = 0.00005 
        self.num_epochs = 2000 
        self.feature_dim = 294
        
        self.features_path = r'/host_data/van/librimead/.' # Pickle Files
        self.model_path = r'/host_data/van/librimead/models/.' 
        self.tensorboard_path = r'/host_data/van/librimead/models/tensorboard/.'
        self.test_path = r'/host_source/van/speech/osre/test/.' # Test folder containing audio wav files

        # MEAD Dataset (Pickle Processing)
        #self.features_test_path = r'/host_data/van/mead/test/.'

        # MEAD + Librispeech (Pickle Processing)
        #self.pickle_path = r'/host_data/van/augmented_data/.'
        #self.mead_train_path = r'/host_data/van/mead/train/.'
        #self.mead_valid_path = r'/host_data/van/mead/valid/.'
        #self.phoneme_train_path = r'/host_data/van/librispeech_phoneme/train/.'
        #self.phoneme_valid_path = r'/host_data/van/librispeech_phoneme/valid/.'
        #self.librispeech_train_path = r'/host_data/van/librispeech/features/train/.'
        #self.librispeech_valid_path = r'/host_data/van/librispeech/features/valid/.'

params = Params()
