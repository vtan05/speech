class Params(object):

    def __init__(self):

        # Model
        self.type = 'osre'
        self.device = 1

        # Audio Parameters
        self.fps = 60.0
        self.srnet_win_size = int(1 * self.fps)
        self.sampling_rate = 16000
        self.window_length = int(1 * self.sampling_rate)
        self.window_shift = int((self.sampling_rate / self.fps) - self.fps - 1)
        self.hop_length = int(0.010 * self.sampling_rate)
        self.hamm_length = int(0.020 * self.sampling_rate)
        self.nmels = 13
        
		# Local Data Path
        self.audio_train_path = r'C:\Users\vanta\Desktop\SRNet\data\srnet\train\librispeech\*.wav'
        self.audio_valid_path = r'C:\Users\vanta\Desktop\SRNet\data\srnet\valid\librispeech\*.wav'

        self.phoneme_train_path = r'C:\Users\vanta\Desktop\SRNet\data\srnet\train\librispeech\.'
        self.phoneme_valid_path = r'C:\Users\vanta\Desktop\SRNet\data\srnet\valid\librispeech\.'

        self.features_train_path = r'C:\Users\vanta\Desktop\SRNet\others_src\osre\data\librispeech\train\.'
        self.features_valid_path = r'C:\Users\vanta\Desktop\SRNet\others_src\osre\data\librispeech\valid\.'

        self.feature_path = r'C:\Users\vanta\Desktop\SRNet\others_src\osre\data\librispeech\features\.'
        self.eval_path = r'C:\Users\vanta\Desktop\SRNet\others_src\osre\data\librispeech\eval\.'
        self.model_path = r'C:\Users\vanta\Desktop\SRNet\others_src\osre\models\.'
        self.tensorboard_path = r'C:\Users\vanta\Desktop\SRNet\others_src\osre\models\tensorboard\.'

        # Server Data Path
        # self.audio_train_path = r'/host_data/van/librispeech/train/*.wav'
        # self.audio_valid_path = r'/host_data/van/librispeech/valid/*.wav'
        # self.features_train_path = r'/host_data/van/librispeech/features/train/.'
        # self.features_valid_path = r'/host_data/van/librispeech/features/valid/.'

        # Network Parameters
        self.type = 'osre'
        self.batch_size = 64
        self.learning_rate = 0.00005 # 3e-6
        self.num_epochs = 2000 # 100
        self.stop_epoch = 5
        self.feature_dim = 294

params = Params()
