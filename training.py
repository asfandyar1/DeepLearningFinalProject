import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import torch
import pickle
import pandas as pd
import os
import math
from models import SqueezeNetFasterRCNN
from dataloader import CarLicensePlatesPickle

import transforms as tf

class Logger:
    def __init__(self, path):
        self.data = {}
        self.vars = {}
        self.path = path + '/log.csv'
    def load(self):
        df = pd.read_csv(self.path, index_col=0, dtype={0:int})
        self.data = df.to_dict()
    def write(self):
        df = pd.DataFrame(self.data)
        df.to_csv(self.path)
    def add(self, step, value_dict):
        for key, val in value_dict.items():
            if key not in self.data:
                self.data[key] = {}
            self.data[key][step] = val.cpu().detach().numpy()
    def getVar(self, var, type_=float):
        if var not in self.data:
            return None, None
        xaxis = list(self.data[var].keys())
        xaxis.sort()
        yaxis = [self.data[var][x] for x in xaxis]
        return xaxis, list(map(type_, yaxis))


class Experiment:
    
    def __init__(self, path, args):
        self.args = args
        self.path = path
        self.model = args['model'](**args['model_args'])
        self.optimizer = args['optimizer'](self.model.parameters(), **args['optimizer_args'])
        self.log = Logger(path)
        self.epoch_n = 0
        self.benchmarks = []
        if not os.path.exists(path):
            print('Create path')
            os.mkdir(path)
    
    @staticmethod
    def load(path):
        with open(path+'/info.pickle', 'rb') as fp:
            info = pickle.load(fp)
            #print(info)
        experiment = Experiment(path, info['args'])
        if os.path.exists(experiment.log.path) and os.path.isfile(experiment.log.path):
            experiment.log.load()
        experiment.epoch_n = info['epoch']
        experiment.benchmarks = info['benchmarks']
        experiment.loadModel()

        return experiment
    
    def loadModel(self):
        if len(self.benchmarks) == 0: return
        self.model.load_state_dict(torch.load(self.benchmarks[-1]))
    
    
    def train(self, train_loader, num_epochs = 5, loss_freq = 50, clip=None):
        list_iter = []
        list_loss = []
        
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print(device)
        self.model.to(device)
        self.model.train() # Set the model in train mode
        total_step = len(train_loader)
        print('Starting Training')
        # Iterate over epochs
        for epoch in range(num_epochs):
            # Iterate the dataset
            for i, (images, labels) in enumerate(train_loader):
                # Get batch of samples and labels
                images = list(image.to(device) for image in images)
                labels = [{k: v.to(device) for k, v in t.items()} for t in labels]
                # Forward pass, returns dictionary with different losses
                outputs = self.model(images, labels) ##CAREFUL THIS MODIFIES LABELS, must copy their values for later access
                losses = sum(loss for loss in outputs.values())
                if not math.isfinite(losses):
                    print("Loss is {}, stopping training".format(losses))
                    print(outputs)
                    return
                # Backward and optimize
                self.optimizer.zero_grad()
                losses.backward()
                if clip != None:
                    torch.nn.utils.clip_grad_value_(self.model.parameters(), clip)
                self.optimizer.step()
                
                if (i+1) % 100 == 0:
                    print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, i+1, total_step, losses))
                if (i+1) % loss_freq == 0:
                    outputs['total_loss'] = losses
                    self.log.add((self.epoch_n+epoch)*total_step+i, outputs)
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, i+1, total_step, losses))
        self.updateResults(num_epochs)
    
    def test_model(self, test_loader, metric_name, metric):
        
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        self.model.eval() # Set the model in evaluation mode
        self.model.to(device)
        # Compute testing accuracy
        with torch.no_grad():
            result = 0
            total = 0
            for images, labels in test_loader:
                images = list(image.to(device) for image in images)
                labels = [{k: v.to(device) for k, v in t.items()} for t in labels]
                # get network predictions
                outputs = self.model(images)
                
                total += len(labels)
                result += (metric(outputs, labels)).sum().item()

        return result / total

    def store(self):
        with open(self.path +'/info.pickle', 'wb') as fp:
            pickle.dump({'args':self.args, 'epoch': self.epoch_n, 'benchmarks': self.benchmarks}, fp)
        
    def updateResults(self, num_epochs):
        model_path = self.path + '/model_{}.ckpt'.format(self.epoch_n)
        self.benchmarks.append(model_path)
        self.epoch_n += num_epochs
        torch.save(self.model.state_dict(), model_path)
        self.store()
        self.log.write()




if __name__ == "__main__":

    arguments = {
        'model' : SqueezeNetFasterRCNN(),
        'model_args' : dict(),
        'optimizer' : optim.Adam,
        'optimizer_args' : dict(lr = 0.0001, weight_decay=1e-4),
    }
    parser = argparse.ArgumentParser()
    parser.add_argument('datapath', help='Path to data stored with pickle')
    parser.add_argument('modelpath', help='Path in which to store the model and logs')
    parser.add_argument('-e', '--epochs', default=6, help='Number of epochs to train')
    parser.add_argument('-l' '--load', default=None, help='Path from which to load an already trained model')
    parser.add_argument('-b', '--batch', default=12, help='Training minibatch size')
    parser.parse_args()
    transform = tf.Compose([tf.SetBoxLabel('vehicle_position', 'vehicle_type'), tf.ToTensor()])
    dataset = CarLicensePlatesPickle(parser.datapath, transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=parser.batch, shuffle=True, collate_fn=tf.collate_fn)
    exp = None
    if parser.load is not None:
        exp = Experiment.load(parser.load)
    else:
        exp = Experiment(parser.modelpath, arguments)
    exp.train(train_loader, num_epochs=parser.epochs, loss_freq=25)
