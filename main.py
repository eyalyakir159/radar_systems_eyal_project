import network.basic_linear_network,network.conv_network,network.Doppler_net,network.NASNetMobile,network.unet_model
from utills import data_managment,model_utils
from network import *
import torch
import torch.nn as nn
import torch.nn.functional as F



#hyper paramaters
LEARING_RATE = 0.0002
BATCHSIZE=32
EPCHOS = 200

#network stuff
#model = network.basic_linear_network.SimpleLinearNetwork(10*61*4,3) ##choose model
#model = network.conv_network.CustomCNN()
#model = network.Doppler_net.DopplerNet()
#model = network.NASNetMobile.ModifiedMobileNetV2(3)
model = network.unet_model.UNet(4,3)
criterion = nn.BCEWithLogitsLoss() ## becuse it is normal labling
#criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARING_RATE)

#get data
train_loader,val_loader,test_loader = data_managment.get_data_loader(BATCHSIZE)


#train
for epcho in range(EPCHOS):
    loss = model_utils.train(model,train_loader,criterion,optimizer)
    print(f"epcho number - {epcho}, train loss - {loss}")
    #evaluate
    model_utils.evaluate(model, val_loader)
    #model_utils.evaluate(model, train_loader)



#test
model_utils.evaluate(model,test_loader)



