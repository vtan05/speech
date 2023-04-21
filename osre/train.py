import sys, os, torch
from torch.utils.tensorboard import SummaryWriter
import data_manager as data_manager
import model as model
import numpy as np
from params import params


class EarlyStopper:
    def __init__(self, patience=10, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_valid_loss = np.inf

    def early_stop(self, valid_loss):
        if valid_loss < self.min_valid_loss:
            self.min_valid_loss = valid_loss
            self.counter = 0
        elif valid_loss > (self.min_valid_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
    

class Runner(object):
    def __init__(self, params):
        self.learning_rate = params.learning_rate
        self.model = model.OSRENet()
        self.criterion = torch.nn.MSELoss()
        self.device = torch.device("cpu")

        # GPU Setting
        if params.device > 0:
            torch.cuda.set_device(params.device - 1)
            self.model.cuda(params.device - 1)
            self.device = torch.device("cuda:" + str(params.device - 1))
            self.criterion.cuda(params.device - 1)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        #self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1000, gamma=0.5)


    def run(self, dataloader, mode='train'):
        self.model.train() if mode == 'train' else self.model.eval()

        epoch_loss = {'loss': 0.0}

        for batch, (audio, pps) in enumerate(dataloader):
            # divide the consecutive frames
            audio = torch.cat([audio[:, :params.srnet_win_size, :], audio[:, 1:, :]], dim=0)
            pps = torch.cat([pps[:, 0], pps[:, 1]], dim=0) # phonemes per second

            audio = audio.to(self.device).float()
            audio = audio.view(audio.size(0), audio.size(1), audio.size(2))
            pps = pps.to(self.device).float()
            self.optimizer.zero_grad()

            # prediction
            pred_pps = self.model(audio)

            # gt loss
            gt_loss = torch.sqrt(self.criterion(pred_pps, pps))

            loss = gt_loss

            if mode == 'train':
                loss.backward()
                self.optimizer.step()

            epoch_loss['loss'] += audio.size(0) * gt_loss.item()

        epoch_loss['loss'] = epoch_loss['loss'] / (len(dataloader.dataset) * 2)

        return epoch_loss


def device_name(device):
    device_name = 'CPU' if device == 0 else 'GPU:' + str(device - 1)
    return device_name


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    train_loader, valid_loader = data_manager.get_dataloader()
    #early_stopper = EarlyStopper()
    runner = Runner(params)
    min_valid_loss = 1000
    saved_epoch = 0

    print("parameters # of the model : {}".format(count_parameters(runner.model)))
    print('Training on ' + device_name(params.device))

    if os.path.isfile(params.model_path):
        checkpoint = torch.load(params.model_path)
        runner.model.load_state_dict(checkpoint['model_state_dict'])
        runner.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        min_valid_loss = checkpoint['valid_loss']
        saved_epoch = checkpoint['epoch']
        runner.learning_rate = checkpoint['lr']
        print("saved epoch: {}".format(saved_epoch))
        print("saved train loss: {}".format(checkpoint['train_loss']))
        print("saved valid loss: {}".format(min_valid_loss))

    # make path
    if os.path.isdir('{}'.format(params.model_path)) == False:
        os.mkdir('{}'.format(params.model_path))
    if os.path.isdir('{}/tensorboard'.format(params.model_path)) == False:
        os.mkdir('{}/tensorboard'.format(params.model_path))

    writer = SummaryWriter(params.tensorboard_path)
    for epoch in range(params.num_epochs):
        epoch += saved_epoch + 1

        train_loss = runner.run(train_loader, 'train')
        valid_loss = runner.run(valid_loader, 'eval')

        # EarlyStopping
        #if early_stopper.early_stop(valid_loss['loss']):             
        #    break

        # Tensorboard
        # writer.add_scalars('Accuracy', {'train': train_loss['accuracy'], 'valid': valid_loss['accuracy']},  epoch)
        writer.add_scalars('Loss', {'train': train_loss['loss'], 'valid': valid_loss['loss'],}, epoch)

        # Save
        # if min_valid_loss > valid_loss['loss']:
        #     min_valid_loss = valid_loss['loss']
        if epoch % 20 == 0:
            torch.save({'model_state_dict': runner.model.state_dict(),
                        'optimizer_state_dict': runner.optimizer.state_dict(),
                        'train_loss': train_loss['loss'],
                        'valid_loss': valid_loss['loss'],
                        'epoch': epoch,
                        'lr': runner.learning_rate
                        }, params.model_path + 'model_2000.pth')

            print("[Epoch %d] [Train : %.4f] [Valid : %.4f] --- Saved model" % (
                epoch, train_loss['loss'], valid_loss['loss']))
        else:
            print("[Epoch %d] [Train : %.4f] [Valid : %.4f]" % (
                epoch, train_loss['loss'], valid_loss['loss']))

        #runner.scheduler.step()


if __name__ == '__main__':
    main()
