import numpy as np
import pandas as pd
import torch
import os

from preprocess import get_data, create_dataloaders, normalize_signals
from train_utils import TrainerDeepSVDD

from indicators import duration_openep, num_peaks_openep, uni_slope_annotation_time, bip_slope_annotation_time, integral_EGM


patients_directory = 'patients/'
os.makedirs('patients', exist_ok=True)

indicators_directory = 'indicators/'
os.makedirs('indicators', exist_ok=True)

train_patients = [''] # list of train patients

test_patient = '' # test patient


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_unipolar = np.empty((0,64))
train_bipolar = np.empty((0,64))

for patient in train_patients:
    p_unipolar, p_bipolar, _, _, _, _, _, _= get_data(f'{patients_directory}/{patient}_unipolars.npy', f'{patients_directory}/{patient}_bipolars.npy', f'{patients_directory}/{patient}.csv')
    train_unipolar = np.append(train_unipolar, p_unipolar, axis=0)
    train_bipolar = np.append(train_bipolar, p_bipolar, axis=0)

unipolar, bipolar, original_unipolar, original_bipolar, signal_indexes, unip_voltages, bip_voltages, ats = get_data(f'{patients_directory}/{test_patient}_unipolar.npy', f'{patients_directory}/{test_patient}_bipolars.npy', f'{patients_directory}/{test_patient}.csv')

normalized_train_bipolar = normalize_signals(train_bipolar)
normalized_bipolar = normalize_signals(bipolar)

# for pytorch
normalized_bipolar = normalized_bipolar.reshape(normalized_bipolar.shape+(1,)).transpose(0,2,1).astype(np.float32)
normalized_train_bipolar = normalized_train_bipolar.reshape(normalized_train_bipolar.shape+(1,)).transpose(0,2,1).astype(np.float32)


train_dl, val_dl, test_dl = create_dataloaders(normalized_train_bipolar, normalized_bipolar, batch_size=512)


# DEEP SVDD NETWORK

deep_SVDD = TrainerDeepSVDD(1, 64, 3, 16, device=device)

deep_SVDD.pretrain(train_dl, val_dl, num_epochs=100)
deep_SVDD.train(train_dl, num_epochs=50)
deep_svdd_scores = deep_SVDD.eval(test_dl)

# CREATE CSV INDICATORS

durations = []
for b in bipolar:
    durations.append(duration_openep(b))

peaks = []
for b in bipolar:
    peaks.append(num_peaks_openep(b))

uni_slope_annotations = []
for u in unipolar:
    uni_slope_annotations.append(uni_slope_annotation_time(u))

bip_slope_annotations = []
for b in bipolar:
    bip_slope_annotations.append(bip_slope_annotation_time(b))

integral_EGMs = []
for b in bipolar:
    integral_EGMs.append(integral_EGM(b))

patient_df = pd.read_csv(f'{patients_directory}/{test_patient}.csv')
df= pd.DataFrame({'ind': signal_indexes})
df_merged = pd.merge(left=patient_df, right=df, left_on='ind', right_on='ind')

df = pd.DataFrame({'x': df_merged['x'], 'y': df_merged['y'], 'z': df_merged['z'], 'AT': ats, 'BipVoltage': bip_voltages, 'UnipVoltage': unip_voltages,  'Peaks': peaks, 'SignalDuration': durations, 'UnipolarSlope': uni_slope_annotations, 'BipolarSlope': bip_slope_annotations, 'Integral': integral_EGMs,  'deep_svdd_scores':deep_svdd_scores})
df.to_csv(f'{indicators_directory}/{test_patient}.csv')



