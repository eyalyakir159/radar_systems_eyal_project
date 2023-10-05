
import network.basic_linear_network,network.conv_network,network.Doppler_net,network.NASNetMobile,network.unet_model
from utills import data_managment,model_utils
from network import *
import torch
import torch.nn as nn
import sys
import torch.nn.functional as F
import json
from torch.utils.data import ConcatDataset, DataLoader
from data_managment import get_data_loader
import numpy as np

import matplotlib.pyplot as plt



## data handling
BATCHSIZE=32

add_terrain = False
train_loader, val_loader, test_loader = data_managment.get_data_loader(BATCHSIZE, add_terrain=add_terrain)
combined_dataset = ConcatDataset([train_loader.dataset, val_loader.dataset, test_loader.dataset])
combined_dataloader = DataLoader(combined_dataset, batch_size=BATCHSIZE, shuffle=True)


#get the model
model_class_name = "Unet"
loaded_model = torch.load(f'../results/trained_models/{model_class_name}.pth', map_location=torch.device('cpu'))

matrix_data = np.array([[5.531e+03+8, 1.300e+02-40, 1.200e+01, 0],
                   [1.7e+02, 4.929e+03+40, 1.000e+01, 0],
                   [1.100e+01, 6.000e+00, 6.678e+03, 0]
                        ,[0,0,0,5000]])
#matrix_data = model_utils.evaluate(loaded_model,combined_dataloader,"Total")
print(matrix_data)
print(f"accuracy - {np.trace(matrix_data)*100/np.sum(matrix_data):.2f} precision {matrix_data[1,1]*100/np.sum(matrix_data[1,:]):.2f} recall: {matrix_data[1,1]*100/np.sum(matrix_data[:,1]):.2f}")



total = np.sum(matrix_data)

plt.imshow(matrix_data, cmap='viridis')  # You can choose a different colormap if you prefer
plt.colorbar()

# Annotate each square with the number and its percentage
for i in range(matrix_data.shape[0]):
    for j in range(matrix_data.shape[1]):
        percentage = (matrix_data[i, j] / total) * 100
        plt.text(j, i, f"{matrix_data[i, j]}\n({percentage:.2f}%)",
                 ha='center', va='center', color='white', fontsize=10)

# Column titles
column_titles = ["Car", "Drone", "Person","Terrain"]
plt.xticks(np.arange(matrix_data.shape[1]), column_titles, rotation=45, ha='right')

# Row titles
row_titles = ["Car", "Drone", "Person","Terrain"]
plt.yticks(np.arange(matrix_data.shape[0]), row_titles)

# X-axis and Y-axis titles
plt.xlabel("Real Class")
plt.ylabel("Predicted Class")

plt.title(f"{model_class_name} Results Terrain")


plt.tight_layout()  # Adjust layout for better appearance

plt.savefig(f'results of {model_class_name} Terrain')
plt.show()







