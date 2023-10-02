import pandas as pd
import torch
import os
import numpy as np  # If your data is not already in a numpy array
from torch.utils.data import Dataset, DataLoader,random_split
import matplotlib.pyplot as plt
import matplotlib.cm as cm

current_dir = os.getcwd()

#root_dir = current_dir
root_dir = os.path.dirname(current_dir)
def load_file_to_csv(file_path,normalized=False):
    path = f"{root_dir}/data/{file_path}"
    df = pd.read_csv(path)
    tensor = torch.tensor(df.values)

    # Normalize the array values to the range [0, 1]
    normalized_array = (tensor - tensor.min()) / (tensor.max() - tensor.min())
    # Map the normalized array to RGBA values using the colormap
    rgba_image = torch.tensor(cm.viridis(normalized_array)).permute(2,0,1)
    return rgba_image



def get_data():

    Cars = []
    Drones = []
    People = []

    for folder in os.listdir(os.path.join(root_dir, 'data/Cars')):
        if not folder.startswith('.'):
            for file in os.listdir(os.path.join(root_dir, f'data/Cars/{folder}')):
                Cars.append(load_file_to_csv(f'Cars/{folder}/{file}'))

    for folder in os.listdir(os.path.join(root_dir, 'data/Drones')):
        if not folder.startswith('.'):
            for file in os.listdir(os.path.join(root_dir, f'data/Drones/{folder}')):
                Drones.append(load_file_to_csv(f'Drones/{folder}/{file}'))

    for folder in os.listdir(os.path.join(root_dir, 'data/People')):
        if not folder.startswith('.'):
            for file in os.listdir(os.path.join(root_dir, f'data/People/{folder}')):
                People.append(load_file_to_csv(f'People/{folder}/{file}'))
    return torch.stack(Cars),torch.stack(Drones),torch.stack(People)



# Define terrain types
terrains = ['mountain', 'grass', 'sea']


# Function to simulate radar signal for a given terrain
def simulate_radar_signal(terrain):
    assert terrain in ['mountain','grass','sea']
    np.random.seed(42)  # For reproducibility
    signal = np.random.rand(10, 61)  # Initialize signal with random noise

    # Modify signal based on terrain type
    if terrain == 'mountain':
        signal[:, :20] += 1  # Increase power in the first 20 frequency bins
    elif terrain == 'grass':
        signal[:, 20:40] += 1  # Increase power in the middle 20 frequency bins
    elif terrain == 'sea':
        signal[:, 40:] += 1  # Increase power in the last 21 frequency bins

    return torch.from_numpy(signal)

def get_terrain_data(amount=500):
    terrains = ['mountain','grass','sea']
    data = []
    for terrain in terrains:
        for _ in range(100):  # Number of samples per terrain
            data.append(simulate_radar_signal(terrain))
    return torch.stack(data)
def get_data_loader(batch_size,shuffle=True,add_terrain=False):
    cars,drones,pepople = get_data()
    if add_terrain:
        terrain = get_terrain_data(500)
    class CustomDataset(Dataset):
        def __init__(self, cars, drones, people,terrain=None):
            self.data = []
            if terrain:
                self.data.extend([(sample, torch.tensor([1, 0, 0,0])) for sample in cars])  # [1, 0, 0,0] for cars
                self.data.extend([(sample, torch.tensor([0, 1, 0,0])) for sample in drones])  # [0, 1, 0,0] for drones
                self.data.extend([(sample, torch.tensor([0, 0, 1,0])) for sample in people])  # [0, 0,1,0] for people
                self.data.extend([(sample, torch.tensor([0, 0, 0,1])) for sample in terrain])  # [0, 0,0,1] for people
            else:
                self.data.extend([(sample, torch.tensor([1, 0, 0])) for sample in cars])  # [1, 0, 0,0] for cars
                self.data.extend([(sample, torch.tensor([0, 1, 0])) for sample in drones])  # [0, 1, 0,0] for drones
                self.data.extend([(sample, torch.tensor([0, 0, 1])) for sample in people])  # [0, 0,1,0] for people

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            return self.data[idx]
    if add_terrain:
        dataset = CustomDataset(cars, drones, pepople,terrain)
    else:
        dataset = CustomDataset(cars, drones, pepople)

    total_size = len(dataset)
    train_size = int(0.7 * total_size)  # 70% for training
    val_size = int(0.15 * total_size)  # 15% for validation
    test_size = total_size - train_size - val_size  # 15% for testing

    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    return DataLoader(train_dataset, batch_size=32, shuffle=True),DataLoader(val_dataset, batch_size=32, shuffle=True),DataLoader(test_dataset, batch_size=32, shuffle=True)








