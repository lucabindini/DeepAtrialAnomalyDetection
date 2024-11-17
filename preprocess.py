import numpy as np
import pandas as pd
from torch.utils import data
from torch.utils.data import DataLoader
import torch


def normalize_signals(signals):
    signals_norm = np.zeros_like(signals)
    for i in range(signals.shape[0]):
        s = signals[i, :]
        normalized_row = (s - np.min(s)) / (np.max(s) - np.min(s))
        signals_norm[i, :] = normalized_row
    return signals_norm

def get_data(unipolar_file, bipolar_file, csv_file):

    U = np.load(unipolar_file)
    B = np.load(bipolar_file)
    
    df_patient = pd.read_csv(csv_file)

    signal_indexes = []
    unipolar_voltages = []
    bipolar_voltages = []
    activation_times = []

    # Gaussian filter
    sigma = 0.8
    x = np.arange(-3*sigma, 3*sigma, 1)
    gaussian = np.exp(-(x/sigma)**2/2)
    gaussian /= gaussian.sum()
    
    CU, CB = [],[]
    OU, OB = [], []
    for i in range(U.shape[0]):

        # Centering EGMs in unipolar derivative argmin
        peak_position = np.gradient(U[i]).argmin()

        if B[i,peak_position-32:peak_position+32].shape[0] == 64:

            u = U[i,peak_position-32:peak_position+32]
            b = B[i,peak_position-32:peak_position+32]

            u = np.convolve(u, gaussian, mode='same')
            b = np.convolve(b, gaussian, mode='same')

            CU.append(u)
            CB.append(b)

            OU.append(U[i,peak_position-32:peak_position+32])
            OB.append(B[i,peak_position-32:peak_position+32])

            signal_indexes.append(i)
            unipolar_voltages.append(df_patient['UnipVoltage'][i])
            bipolar_voltages.append(df_patient['BipVoltage'][i])
            activation_times.append(df_patient['AT'][i])

    return np.array(CU), np.array(CB), np.array(OU), np.array(OB), np.array(signal_indexes), np.array(unipolar_voltages), np.array(bipolar_voltages), np.array(activation_times)


class CustomDataset(data.Dataset):
    """This class is needed to processing batches for the dataloader."""
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        """return transformed items."""
        x = self.data[index]
        return x

    def __len__(self):
        """number of samples."""
        return len(self.data)


def create_dataloaders(train_bipolar, test_bipolar, batch_size):
                                    
    data_train = CustomDataset(train_bipolar)
    train_dataset, val_dataset = torch.utils.data.random_split(data_train, [int(0.8 * len(data_train)), len(data_train) - int(0.8 * len(data_train))])
    dataloader_train = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    dataloader_val = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)


    data_test = CustomDataset(test_bipolar)
    dataloader_test = DataLoader(data_test, batch_size=batch_size, shuffle=False)
    return dataloader_train, dataloader_val, dataloader_test