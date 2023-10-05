import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

def train(model,train_loader,criterion,optimizer):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.train()  # Set the model to training mode
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(train_loader, 0):
        optimizer.zero_grad()  # Zero the parameter gradients
        inputs = inputs.to(device)
        outputs = model(inputs).to(device)  # Forward pass
        loss = criterion(outputs, labels.to(dtype=torch.float32))  # Compute the loss
        loss.backward()  # Backward pass
        optimizer.step()  # Optimize

        running_loss += loss.item()
    return running_loss

def evaluate(model,val_loader,text_from='Validation'):
    # Validation Loop

    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0
    predicted_data = np.zeros((3,3))
    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == torch.argmax(labels, dim=1)).sum().item()
            real,pred = torch.argmax(outputs, dim=1),torch.argmax(labels, dim=1)
            for i in range(len(real)):
                predicted_data[real[i],pred[i]]+=1

    print(f"{text_from} Accuracy: {100 * correct / total}%")
    return predicted_data
