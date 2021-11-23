import torch
import torchaudio
from torch.utils.data import Dataset
import torchaudio.transforms as tf
import torch.nn.functional as F
import pandas as pd
import os

class UrbanSoundDataset(Dataset):
   # annotatio files -> path to csv, audio_dir -> path to dir containing the audio set
   def __init__(self, 
               annotations_file, 
               audio_dir, 
               transformation, 
               target_sample_rate,
               num_samples, 
               device):
      self.annotations = pd.read_csv(annotations_file)
      self.audio_dir = audio_dir
      self.transformation = transformation.to(device)
      self.target_sample_rate = target_sample_rate
      self.num_samples = num_samples
      self.device = device

   def __len__(self):
      return len(self.annotations)

   def __getitem__(self, index):
      #deep learning models requires fixed datasets in shape etc.
      #like mel spectograms. their shape is fixed.
      audio_sample_path = self._get_audio_sample_path(index)
      label = self._audio_sample_label(index)
      signal, sr = torchaudio.load(audio_sample_path)
      signal = signal.to(self.device)
      signal = self._resample_if_necessary(signal, sr)

      # signal -> (num_channels, samples) -> (2, 16000) -> (1, 16000) [by mixing down]
      signal = self._mix_down_if_necessary(signal)

      signal = self._cut_if_necessary(signal)
      signal = self._right_pad_if_necessary(signal)

      signal = self.transformation(signal)
      return signal, label

   def _get_audio_sample_path(self, index):
      fold = f"fold{self.annotations.fold[index]}"
      file = self.annotations.slice_file_name[index]
      path = os.path.join(self.audio_dir, fold, file)
      return path

   def _audio_sample_label(self, index):
      return self.annotations.classID[index]

   def _resample_if_necessary(self, signal, sr):
      
      if( sr != self.target_sample_rate):
         resample = tf.Resample(sr, self.target_sample_rate).to(self.device)
         signal = resample(signal)

      return signal
   
   def _mix_down_if_necessary(self, signal):
      if(signal.shape[0] > 1): # more than one channel
         signal = torch.mean(signal, dim=0, keepdim=True) #mixing down
      return signal

   def _cut_if_necessary(self, signal):
      if(signal.shape[1] > self.num_samples):
         signal = signal[:, :self.num_samples]
      return signal

   def _right_pad_if_necessary(self, signal):
      if(signal.shape[1] < self.num_samples):
         pad_num = self.num_samples - signal.shape[1]
         signal = F.pad(signal, (0,pad_num))
      return signal

if __name__ == '__main__':

   ANNOTATIONS_FILE = './UrbanSound8K/metadata/UrbanSound8K.csv'
   AUDIO_DIR = './UrbanSound8K/audio'
   SAMPLE_RATE = 16000
   NUM_SAMPLES = 32500 # we are getting one second from the expected audio (since sr = 22050 as well)

   if torch.cuda.is_available():
      device = 'cuda'
   else:
      device = 'cpu'

   print(f"Using the device {device}")

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
   
   print(f"There are {len(usd)} samples in the dataset.")

   signal, label = usd[1]

   print(f"First signal is {signal} and dim {signal.shape} with label {label}.")