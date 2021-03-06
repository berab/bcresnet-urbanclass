import torch
from torch import nn
from torch.utils.data import DataLoader
import torchaudio.transforms as tf
from model import BCResNet as Net
from urbansoundDataset import UrbanSoundDataset
import numpy as np
from torch.autograd import Variable

BATCH_SIZE = 128
EPOCHS = 10
LEARNING_RATE = 0.1
ANNOTATIONS_FILE = './UrbanSound8K/metadata/UrbanSound8K.csv'
AUDIO_DIR = './UrbanSound8K/audio'
SAMPLE_RATE = 16000
NUM_SAMPLES = 32500  # we are getting one second from the expected audio (since sr = 22050 as well)
NUM_LABELS = 10
ALPHA = 0.3

def create_data_loader(train_data, batch_size):
    train_dataloader = DataLoader(train_data, batch_size=batch_size)
    return train_dataloader


def train_single_epoch(model, data_loader, loss_fn, optimiser, device):
    for input, target in data_loader:
        input, target = input.to(device), target.to(device)
        input, target_a, target_b, lam = mixup(input, target,
                                                       ALPHA, device)

        input, target_a, target_b = map(Variable, (input,
                                                      target_a, target_b))
        # calculate loss
        prediction = model(input)
        loss = mixup_criterion(loss_fn, prediction.reshape(prediction.shape[0],prediction.shape[1]), target_a, target_b, lam)

        # backpropagate error and update weights
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

    print(f"loss: {loss.item()}")

def mixup(x, y, alpha=1.0, device='cuda'):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)


    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(loss_fn, pred, y_a, y_b, lam):
    return lam * loss_fn(pred, y_a) + (1 - lam) * loss_fn(pred, y_b)

def train(model, data_loader, loss_fn, optimiser, device, epochs):
    for i in range(epochs):
        print(f"Epoch {i+1}")
        train_single_epoch(model, data_loader, loss_fn, optimiser, device)
        print("---------------------------")
        
    print("Finished training")

if __name__ == "__main__":

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    # instantiating our dataset we handled before

    mel_spectrogram = tf.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=1024,
        hop_length=512,
        n_mels=40
    )
    # hop_length is usually n_fft/2
    # ms = mell_spectogram(signal)
    
    usd = UrbanSoundDataset(ANNOTATIONS_FILE,
                            AUDIO_DIR,
                            mel_spectrogram,
                            SAMPLE_RATE,
                            NUM_SAMPLES,
                            device)

    train_dataloader = create_data_loader(usd, BATCH_SIZE)

    # construct model and assign it to device
    net = Net().to(device)
    # print(net)

    # initialise loss funtion + optimiser
    loss_fn = nn.CrossEntropyLoss()

    optimiser = torch.optim.Adam(net.parameters(),
                                 lr=LEARNING_RATE)

    # train model
    train(net, train_dataloader, loss_fn, optimiser, device, EPOCHS)

    # save model
    torch.save(net.state_dict(), "bcresnet_urban.pth")
    print("Trained bcresnet saved at bcresnet_urban.pth")