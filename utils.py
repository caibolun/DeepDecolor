import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torchvision.models as models


def imread(image_name):
    transform = transforms.Compose([transforms.ToTensor()])
    image = Image.open(image_name)
    image = transform(image)
    image = image.unsqueeze(0)
    return image


def imshow(tensor, title=None, cmap=None, filename=None):
    unloader = transforms.ToPILImage()
    image = tensor.clone().cpu()
    image = image.view(image.size(1), image.size(2), image.size(3))
    image = unloader(image)
    plt.ion()
    plt.figure()
    plt.imshow(image, cmap)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)
    if filename is not None:
        image.save(filename)


class L2Loss(nn.Module):
    def __init__(self, target, weight):
        super(L2Loss, self).__init__()
        self.target = target.detach() * weight
        self.weight = weight
        self.criterion = nn.MSELoss()

    def forward(self, input):
        self.loss = self.criterion(input * self.weight, self.target)
        self.output = input
        return self.output

    def backward(self, retain_graph=True):
        self.loss.backward(retain_graph=retain_graph)
        return self.loss


class TileLayer(nn.Module):
    def __init__(self, channel):
        super(TileLayer, self).__init__()
        self.channel=channel

    def forward(self, x):
        y=x.repeat(1, self.channel, 1, 1)
        return y


perceptual_layers_default = ['layer_1', 'layer_2', 'layer_4', 'layer_8', 'layer_12']
perceptual_weights_default = [1, 1, 0.5, 0.25, 0.125]
def DecolorNet(cnn, color_img, use_cuda,
                perceptual_layers = perceptual_layers_default, 
                perceptual_weights = perceptual_weights_default):
    losses = []
    model = nn.Sequential()
    if use_cuda: model = model.cuda()

    i = 1
    for layer in list(cnn):
        name = "layer_" + str(i)
        if isinstance(layer, nn.Conv2d):
            model.add_module("conv_" + str(i), layer)
            if name in perceptual_layers:
                dense_target = model(color_img).clone()
                dense_loss = L2Loss(dense_target, perceptual_weights[perceptual_layers.index(name)])
                model.add_module("dense_loss_" + str(i), dense_loss)
                losses.append(dense_loss)   
        if isinstance(layer, nn.ReLU):
            model.add_module("relu_" + str(i), nn.ReLU(inplace=False))
            if name in perceptual_layers:
                sparse_target = model(color_img).clone()
                sparse_loss = L2Loss(sparse_target, perceptual_weights[perceptual_layers.index(name)])
                model.add_module("sparse_loss_" + str(i), sparse_loss)
                losses.append(sparse_loss)
            i += 1

        if isinstance(layer, nn.MaxPool2d):
            model.add_module("pool_" + str(i), layer)
    
    return model, losses
