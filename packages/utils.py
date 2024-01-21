from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision
import torch
from torchvision import transforms as T
import torchvision.transforms.functional as TF
import random
import matplotlib.pyplot as plt
from torcheval.metrics import R2Score

def imshow(img, i=0, mean=torch.tensor([0.0], dtype=torch.float32), std=torch.tensor([1], dtype=torch.float32)):
    """
    shows an image on the screen. mean of 0 and variance of 1 will show the images unchanged in the screen
    """
    # undoes the normalization
    unnormalize = T.Normalize((-mean / std).tolist(), (1.0 / std).tolist())
    plt.subplot(1, 10 ,i+1)
    npimg = unnormalize(img).numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))

def validate(model, val_loader, device):
    model.eval()
    criterion = Softmax_Loss()
    # criterion = nn.CrossEntropyLoss()
    correct, total = 0, 0
    loss_step = []
    pred = []
    targets = []
    with torch.no_grad():
        for inp_data, labels in val_loader:
            ### START CODE HERE ### (approx. 10 lines)
            # Move the data to the GPU
            labels = labels.view(labels.shape[0]).to(device)
            inp_data = inp_data.to(device)
            outputs = model(inp_data)
            ages = torch.range(0, outputs.shape[1]-1).to(device)
            val_loss = criterion(outputs, labels)
            total += labels.size(0)
            correct += torch.abs(outputs@ages - labels).sum()
            loss_step.append(val_loss.item())
            targets.extend(list(labels))
            pred.extend(list(outputs@ages))
        # dont forget to take the means here
        val_acc = (correct / total).cpu().numpy()
        val_loss_epoch = torch.tensor(loss_step).mean().numpy()
        val_corr = R2(pred, targets, device)
        # print(torch.Tensor(pred))
        # print(torch.Tensor(targets))
        ### END CODE HERE ###
        return val_acc, val_loss_epoch, val_corr

# Hongli: custom loss function
class Softmax_Loss(nn.Module):
    def __init__(self):
        super(Softmax_Loss, self).__init__()
        # age boundary = 3
        self.b = 3

    def gauss(self, y_i, k, Z):
        sig = 4
        return 1.0 / (np.sqrt(2.0 * np.pi) * sig * Z) * torch.exp(-torch.pow((k - y_i) / sig, 2.0) / 2)

    def forward(self, outputs, targets):
        # the effective range restricted by the boundary
        loss = 0
        for i in range(len(targets)):
            y_i = targets[i]
            output = outputs[i]
            gauss_range = torch.arange(max(y_i - self.b, 0),  min(y_i + self.b + 1, 100)).to('cuda')
            Z = output[gauss_range].sum()
            loss -= (self.gauss(y_i, gauss_range, Z)*torch.log(output[gauss_range])).sum()
        loss = loss/len(targets)
        return loss


def R2(pred, targets, device):
    metric = R2Score(device=device)
    metric.update(torch.Tensor(pred), torch.Tensor(targets))
    return metric.compute()

def train_one_epoch(model, optimizer, train_loader, device):
    model.train()
    criterion = Softmax_Loss()
    # criterion = nn.CrossEntropyLoss()
    loss_step = []
    pred = []
    targets = []
    correct, total = 0, 0
    for (inp_data, labels) in train_loader:
        ### START CODE HERE ### (approx. 14 lines)
        # Move the data to the GPU
        labels = labels.view(labels.shape[0]).to(device)
        inp_data = inp_data.to(device)
        outputs = model(inp_data)
        ages = torch.range(0, outputs.shape[1]-1).to(device)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            total += labels.size(0)
            correct += torch.abs(outputs@ages - labels).sum()
            loss_step.append(loss.item())
            targets.extend(list(labels))
            pred.extend(list(outputs@ages))

    # dont forget the means here
    loss_curr_epoch = np.mean(loss_step)
    train_acc = (correct / total).cpu()
    train_corr = R2(pred, targets, device)
    ### END CODE HERE ###
    return loss_curr_epoch, train_acc, train_corr


def train(model, optimizer, num_epochs, train_loader, val_loader, device):
    best_val_loss = 1000
    best_val_acc = 0
    model = model.to(device)
    dict_log = {"train_acc_epoch":[], "val_acc_epoch":[], "loss_epoch":[], "val_loss":[], "train_corr":[], "val_corr":[]}
    train_acc, _, __ = validate(model, train_loader, device)
    val_acc, _, __ = validate(model, val_loader, device)
    print(f'Init MAE of the model: Train:{train_acc:.3f} \t Val:{val_acc:3f}')
    pbar = tqdm(range(num_epochs))
    for epoch in pbar:
        ### START CODE HERE ### (approx. 2 lines)
        loss_curr_epoch, train_acc, train_corr = train_one_epoch(model, optimizer, train_loader, device)
        val_acc, val_loss, val_corr = validate(model, val_loader, device)
        ### END CODE HERE ###

        # Print epoch results to screen
        msg = (f'Ep {epoch}/{num_epochs}: MAE : Train:{train_acc:.2f} \t Val:\
            {val_acc:.2f} || Loss: Train {loss_curr_epoch:.3f} \t Val {val_loss:.3f}\
            || Corr: Train {train_corr:.3f} \t Val {val_corr:.3f}')
        pbar.set_description(msg)
        # Track stats
        dict_log["train_acc_epoch"].append(train_acc)
        dict_log["val_acc_epoch"].append(val_acc)
        dict_log["loss_epoch"].append(loss_curr_epoch)
        dict_log["val_loss"].append(val_loss)
        dict_log["train_corr"].append(train_corr)
        dict_log["val_corr"].append(val_corr)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                  'epoch': epoch,
                  'model_state_dict': model.state_dict(),
                  'optimizer_state_dict': optimizer.state_dict(),
                  'loss': val_loss,
                  }, f'best_model_min_val_loss.pth')

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                  'epoch': epoch,
                  'model_state_dict': model.state_dict(),
                  'optimizer_state_dict': optimizer.state_dict(),
                  'loss': val_loss,
                  }, f'best_model_max_val_acc.pth')
    return dict_log


def load_model(model, path):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Model {path} is loaded from epoch {checkpoint['epoch']} , loss {checkpoint['loss']}")
    return model

def test_model(model, path, test_loader, device='cuda'):
    model = load_model(model, path)
    model.to("cuda")
    model.eval()
    return validate(model, test_loader, device)


def plot_stats(dict_log, modelname="",baseline=90, title=None):
    fontsize = 14
    plt.subplots_adjust(hspace=0.3)
    plt.subplot(2,1,1)
    ### START CODE HERE ### (approx. 5 lines)
    x_axis = list(range(len(dict_log["val_acc_epoch"])))
    plt.plot(dict_log["train_acc_epoch"], label=f'{modelname} Train accuracy')
    plt.scatter(x_axis, dict_log["train_acc_epoch"])

    plt.plot(dict_log["val_acc_epoch"], label=f'{modelname} Validation accuracy')
    plt.scatter(x_axis, dict_log["val_acc_epoch"])
    ### END CODE HERE ###


    plt.ylabel('Accuracy in %')
    plt.xlabel('Number of Epochs')
    plt.title("Accuracy over epochs", fontsize=fontsize)
    plt.axhline(y=baseline, color='red', label="Acceptable accuracy")
    plt.legend(fontsize=fontsize)


    plt.subplot(2,1,2)
    plt.plot(dict_log["loss_epoch"] , label="Training")

    ### START CODE HERE ### (approx. 3 lines)
    plt.scatter(x_axis, dict_log["loss_epoch"], )
    plt.plot(dict_log["val_loss"] , label='Validation')
    plt.scatter(x_axis, dict_log["val_loss"])
    ### END CODE HERE ###

    plt.ylabel('Loss value')
    plt.xlabel('Number of Epochs')
    plt.title("Loss over epochs", fontsize=fontsize)
    plt.legend(fontsize=fontsize)
    if title is not None:
        plt.savefig(title)
