import numpy as np

import network.basic_linear_network,network.conv_network,network.Doppler_net,network.NASNetMobile,network.unet_model
from utills import data_managment,model_utils
from network import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import json

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



#hyper paramaters
LEARING_RATE = 0.0002
BATCHSIZE=32
EPCHOS = 100

out_classes = 4
#network stuff

#models = [network.basic_linear_network.SimpleLinearNetwork(10*61*4,out_classes),network.conv_network.CustomCNN(4,out_classes),
        #  network.Doppler_net.DopplerNet(4,out_classes),network.unet_model.UNet(4,3,out_classes)]

#model = network.basic_linear_network.SimpleLinearNetwork(10*61*4,3) ##choose model
#model = network.conv_network.CustomCNN(4,3)
#model = network.Doppler_net.DopplerNet()
#model = network.NASNetMobile.ModifiedMobileNetV2(3)
model = network.unet_model.UNet(4,4).to(device)

criterion = nn.BCEWithLogitsLoss() ## becuse it is normal labling
#criterion = nn.CrossEntropyLoss()


# get data
add_terrain = True
train_loader, val_loader, test_loader = data_managment.get_data_loader(BATCHSIZE, add_terrain=add_terrain)

print(f"running on model {model.__class__.__name__}")
optimizer = torch.optim.Adam(model.parameters(), lr=LEARING_RATE)
#train
for epcho in range(EPCHOS):
    loss = model_utils.train(model,train_loader,criterion,optimizer)
    print(f"epcho number - {epcho}, train loss - {loss}")
    #evaluate
    model_utils.evaluate(model, val_loader)
    model_utils.evaluate(model, train_loader ,'Training')



    #test
    model_utils.evaluate(model,test_loader,'Test')

    torch.save(model, f'results/trained_models/{model.__class__.__name__}_Terrain.pth')




