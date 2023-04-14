class Params(object):

    def __init__(self):

        # Audio Parameters
        self.fps = 60.0
        self.sampling_rate = 16000
        self.window_length = int(1 * self.sampling_rate)
        self.window_shift = int((self.sampling_rate / self.fps) - self.fps - 1)
        self.hop_length = int(0.010 * self.sampling_rate)
        self.hamm_length = int(0.020 * self.sampling_rate)
        self.nmels = 13
        
		# Data Path
        self.audio_train_path = r'C:\Users\vanta\Desktop\SRNet\data\srnet\train\librispeech\*.wav'
        self.audio_valid_path = r'C:\Users\vanta\Desktop\SRNet\data\srnet\valid\librispeech\*.wav'
        self.features_train_path = r'C:\Users\vanta\Desktop\SRNet\others_src\osre\data\librispeech\train\.'
        self.features_valid_path = r'C:\Users\vanta\Desktop\SRNet\others_src\osre\data\librispeech\valid\.'

        # Model
        self.batch_size = 64
        self.learning_rate = 3e-6
        self.num_epochs = 100
        self.stop_epoch = 5

		# # Dimension
		# self.stft_dim = 513
		# self.mfcc_dim = 39
		# self.ds_dim = 29

		# # feature path
		# self.root = '/source/SJ/SRNet'
		# self.feature_path = self.root + '/feature/{}'.format(self.type)
		# self.dataset_path = self.root + '/data/{}'.format(self.type)
		# self.test_path = self.root + '/data/{}/test'.format(self.dataset)
		# self.tensorboard_path = self.root + '/models/{}/tensorboard/model_{}.pth'.format(self.type, self.model)
		# self.model_path = self.root + '/models/{}/model_{}.pth'.format(self.type, self.model)

params = Params()
