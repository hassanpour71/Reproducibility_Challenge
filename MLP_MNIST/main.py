'''
Train MLPs for MNIST (adapted from https://github.com/lancopku/meProp/tree/master/src/pytorch) 
'''
from __future__ import division
import sys
from argparse import ArgumentParser

import torch



from collections import OrderedDict


import torch.nn as nn
import torch.nn.functional as F
sys.path.append('..')
from Approximation import approx_Linear

import numpy
from torchvision import datasets, transforms


import time
from statistics import mean


import torch.cuda

import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable

class TestGroup(object):
    '''
    Test groups differ in minibatch size, hidden features, layer number and dropout rate.
    '''

    def __init__(self,
                 args,
                 trnset,
                 mb,
                 hidden,
                 layer,
                 dropout,
                 devset=None,
                 tstset=None,
                 cudatensor=False,
                 file=sys.stdout):
        self.args = args
        self.mb = mb
        self.hidden = hidden
        self.layer = layer
        self.dropout = dropout
        self.file = file
        self.trnset = trnset

        if cudatensor:  # dataset is on GPU
            self.trainloader = torch.utils.data.DataLoader(
                trnset, batch_size=mb, num_workers=0)
            if tstset:
                self.testloader = torch.utils.data.DataLoader(
                    tstset, batch_size=mb, num_workers=0)
            else:
                self.testloader = None
        else:  # dataset is on CPU, using prefetch and pinned memory to shorten the data transfer time
            self.trainloader = torch.utils.data.DataLoader(
                trnset,
                batch_size=mb,
                shuffle=True,
                num_workers=1,
                #num_workers=0,
                pin_memory=True)
            if devset:
                self.devloader = torch.utils.data.DataLoader(
                    devset,
                    batch_size=mb,
                    shuffle=False,
                    num_workers=1,
                    #num_workers=0,
                    pin_memory=True)
            else:
                self.devloader = None
            if tstset:
                self.testloader = torch.utils.data.DataLoader(
                    tstset,
                    batch_size=mb,
                    shuffle=False,
                    num_workers=1,
                    #num_workers=0,
                    pin_memory=True)
            else:
                self.testloader = None
        self.basettime = None
        self.basebtime = None

    def reset(self):
        '''
        Reinit the trainloader at the start of each run,
        so that the traning examples is in the same random order
        '''
        torch.manual_seed(self.args.random_seed)
        self.trainloader = torch.utils.data.DataLoader(
            self.trnset,
            batch_size=self.mb,
            shuffle=True,
            num_workers=1,
            #num_workers=0,
            pin_memory=True)

    def _train(self, model, opt):
        '''
        Train the given model using the given optimizer
        Record the time and loss
        '''
        #compare_grad_vs_approx = True
        compare_grad_vs_approx = False
        model.train()
        ftime = 0
        btime = 0
        utime = 0
        tloss = 0
        
        if compare_grad_vs_approx == True:
            num_approx_layers = 0
            for layer in model.children():
                for sublayer in layer.children():
                    #print(sublayer.__class__.__name__)
                    if sublayer.__class__.__name__ == 'approx_Linear':
                        for n,p in sublayer.named_parameters():
                            #print(p.size())
                            if ('weight' in n):
                                num_approx_layers += 1
            
            
            avg_mean = torch.zeros(num_approx_layers, len(self.trainloader))
            avg_mse = torch.zeros(num_approx_layers, len(self.trainloader))
            avg_std = torch.zeros(num_approx_layers, len(self.trainloader))
        
        for bid, (data, target) in enumerate(self.trainloader):
            data, target = data.to(self.args.device), target.view(-1).to(self.args.device)
            #with torch.autograd.profiler.profile(use_cuda=True) as prof:
            #start = torch.cuda.Event(True)
            #end = torch.cuda.Event(True)

            if compare_grad_vs_approx == True:
                # get gradients with non-approximate calculations:
                acc_grads = []
                for layer in model.children():
                    for sublayer in layer.children():
                        if sublayer.__class__.__name__ == 'approx_Linear':
                            sublayer.eval() 
                
                opt.zero_grad()
                output = model(data)
                loss = F.nll_loss(output, target)
                loss.backward()
                for layer in model.children():
                    for sublayer in layer.children():
                        if sublayer.__class__.__name__ == 'approx_Linear':
                            for n,p in sublayer.named_parameters():
                                if ('weight' in n):
                                    acc_grads.append(p.grad.clone())
                        
                
                approx_grads = []
                model.train()
                opt.zero_grad()
                output = model(data)
                loss = F.nll_loss(output, target)
                loss.backward()
                for layer in model.children():
                    for sublayer in layer.children():
                        if sublayer.__class__.__name__ == 'approx_Linear':
                            for n,p in sublayer.named_parameters():
                                if ('weight' in n):
                                    approx_grads.append(p.grad.clone())

                opt.step()
                tloss += loss.item()
                tloss /= len(self.trainloader)
                #print('approx_grads:')
                #print (approx_grads)
                #print ('acc_grads') 
                #print (acc_grads) 
                #print('mean {}'.format(torch.mean(avg_mean,dim=1)))
                #print('relative MSE {}'.format(torch.mean(avg_mse,dim=1)))
                #print('std {}'.format(torch.mean(avg_std,dim=1)))
                for i, (approx_grad,acc_grad) in enumerate(zip(approx_grads,acc_grads)):
                    #print('index {}'.format(i))
                    #print('mean {}'.format((approx_grad-acc_grad).flatten().mean()))
                    avg_mean[i,bid] = (approx_grad-acc_grad).flatten().mean()
                    #print('relative MSE {}'.format((approx_grad-acc_grad).norm()/acc_grad.norm()))
                    avg_mse[i,bid] = (approx_grad-acc_grad).norm()/acc_grad.norm()
                    #print('std {}'.format((approx_grad-acc_grad).flatten().std()))
                    avg_std[i,bid] = (approx_grad-acc_grad).flatten().std()

            else:    
                #start.record()
                opt.zero_grad()
                #end.record()
                #end.synchronize()
                #utime += start.elapsed_time(end)

                #start.record()
                output = model(data)
                loss = F.nll_loss(output, target)
                #end.record()
                #end.synchronize()
                #ftime += start.elapsed_time(end)

                #start.record()
                loss.backward()
                #end.record()
                #end.synchronize()
                #btime += start.elapsed_time(end)

                #start.record()
                opt.step()
                #end.record()
                #end.synchronize()
                #utime += start.elapsed_time(end)

                tloss += loss.item()
        
                tloss /= len(self.trainloader)
                #print(prof.key_averages())
            
        
        if compare_grad_vs_approx == True:
            print('mean {}'.format(torch.mean(avg_mean,dim=1)))
            print('relative MSE {}'.format(torch.mean(avg_mse,dim=1)))
            print('std {}'.format(torch.mean(avg_std,dim=1)))

        
        return tloss, ftime, btime, utime

    def _evaluate(self, model, loader, name='test'):
        '''
        Use the given model to classify the examples in the given data loader
        Record the loss and accuracy.
        '''
        model.eval()
        test_loss = 0
        correct = 0
        for data, target in loader:
            data, target = Variable(
                #data, requires_grad=False).cuda(), Variable(target).cuda()
                data, requires_grad=False).to(self.args.device), Variable(target).to(self.args.device)
            output = model(data)
            test_loss += F.nll_loss(output, target).item()
            pred = output.data.max(1)[
                1]  # get the index of the max log-probability
            correct += pred.eq(target.data).cpu().sum()

        test_loss = test_loss
        test_loss /= len(
            loader)  # loss function already averages over batch size
        print(
            '{} set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
                name, test_loss, correct,
                len(loader.dataset), 100. * float(correct) / len(loader.dataset)),
            file=self.file,
            flush=True)
        return 100. * float(correct) / len(loader.dataset)

    def run(self, epoch=None):
        '''
        Run a training loop.
        '''
        if epoch is None:
            epoch = self.args.n_epoch
        print(
            'mbsize: {}, hidden size: {}, layer: {}, dropout: {}'.
            format(self.mb, self.hidden, self.layer, self.dropout),
            file=self.file)
        # Init the model, the optimizer and some structures for logging
        self.reset()

        model = MLP(self.hidden, self.layer, self.dropout)
        #print(model)
        model.reset_parameters()
        #model.cuda()
        model.to(self.args.device)
        #model = torch.nn.DataParallel(model)
        opt = optim.Adam(model.parameters())

        acc = 0  # best dev. acc.
        accc = 0  # test acc. at the time of best dev. acc.
        e = -1  # best dev iteration/epoch

        times = []
        losses = []
        ftime = []
        btime = []
        utime = []

        print('Initial evaluation on dev set:')
        self._evaluate(model, self.devloader, 'dev')

        start=time.time() 
         # training loop
        for t in range(epoch):
            print('{}：'.format(t), end='', file=self.file, flush=True)
            # train
            
            #start = torch.cuda.Event(True)
            #end = torch.cuda.Event(True)
            #start.record()
            loss, ft, bt, ut = self._train(model, opt)
            #end.record()
            #end.synchronize()
            #ttime = start.elapsed_time(end)
            print("(wall time: {:.1f} sec) ".format(time.time()-start), end='')
            #times.append(ttime)
            losses.append(loss)
            ftime.append(ft)
            btime.append(bt)
            utime.append(ut)
            # predict
            curacc = self._evaluate(model, self.devloader, 'dev')
            #if curacc > acc:
            #    e = t
            #    acc = curacc
            #    accc = self._evaluate(model, self.testloader, '    test')
        etime = [sum(t) for t in zip(ftime, btime, utime)]
        print('test acc: {:.2f}'.format(self._evaluate(model, self.testloader, '    test')))
        print(
            'best on val set - ${:.2f}|{:.2f} at {}'.format(acc, accc, e),
            file=self.file,
            flush=True)
        print('', file=self.file)

    def _stat(self, name, t, agg=mean):
        return '{:<5}:\t{:8.3f}; {}'.format(
            name, agg(t), ', '.join(['{:8.2f}'.format(x) for x in t]))



class PartDataset(torch.utils.data.Dataset):
    '''
    Partial Dataset:
        Extract the examples from the given dataset,
        starting from the offset.
        Stop if reach the length.
    '''

    def __init__(self, dataset, offset, length):
        self.dataset = dataset
        self.offset = offset
        self.length = length
        super(PartDataset, self).__init__()

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        return self.dataset[i + self.offset]


def get_mnist(datapath='~/datasets/mnist', download=True):
    '''
    The MNIST dataset in PyTorch does not have a development set, and has its own format.
    We use the first 5000 examples from the training dataset as the development dataset. (the same with TensorFlow)
    Assuming 'datapath/processed/training.pt' and 'datapath/processed/test.pt' exist, if download is set to False.
    '''
    trn = datasets.MNIST(
        datapath,
        train=True,
        download=download,
        transform=transforms.ToTensor())
    dev = PartDataset(trn, 0, 5000)
    trnn = PartDataset(trn, 5000, 55000)
    tst = datasets.MNIST(
        datapath, train=False, transform=transforms.ToTensor())
    return trnn, dev, tst

sample_ratio=0.9
minimal_k = 10
sample_ratio_bwd= sample_ratio#None
minimal_k_bwd = 10
sample_ratio_wu= sample_ratio#None
minimal_k_wu = 10

class MLP(nn.Module):
    '''
    A complete network(MLP) for MNSIT classification.
    
    Input feature is 28*28=784
    Output feature is 10
    Hidden features are of hidden size
    
    Activation is ReLU
    '''

    def __init__(self, hidden, layer, dropout=None):
        super(MLP, self).__init__()
        self.layer = layer
        self.dropout = dropout
        self.model = nn.Sequential(self._create(hidden, layer, dropout))

    def _create(self, hidden, layer, dropout=None):
        if layer == 1:
            d = OrderedDict()
            d['linear0'] = approx_Linear(784, 10, sample_ratio=sample_ratio, minimal_k=minimal_k, sample_ratio_bwd=sample_ratio_bwd, minimal_k_bwd = minimal_k_bwd, sample_ratio_wu = sample_ratio_wu, minimal_k_wu = minimal_k_wu)
            return d
        d = OrderedDict()
        for i in range(layer):
            if i == 0:
                d['linear' + str(i)] = approx_Linear(784, hidden, sample_ratio=sample_ratio, minimal_k=minimal_k, sample_ratio_bwd=sample_ratio_bwd, minimal_k_bwd = minimal_k_bwd, sample_ratio_wu = sample_ratio_wu, minimal_k_wu = minimal_k_wu)
                d['relu' + str(i)] = nn.ReLU()
                if dropout:
                    d['dropout' + str(i)] = nn.Dropout(p=dropout)
            elif i == layer - 1:
                d['linear' + str(i)] = approx_Linear(hidden, 10, sample_ratio=sample_ratio, minimal_k=minimal_k, sample_ratio_bwd=sample_ratio_bwd, minimal_k_bwd = minimal_k_bwd, sample_ratio_wu = sample_ratio_wu, minimal_k_wu = minimal_k_wu)
            else:
                d['linear' + str(i)] = approx_Linear(hidden, hidden, sample_ratio=sample_ratio, minimal_k=minimal_k, sample_ratio_bwd=sample_ratio_bwd, minimal_k_bwd = minimal_k_bwd, sample_ratio_wu = sample_ratio_wu, minimal_k_wu = minimal_k_wu)
                d['relu' + str(i)] = nn.ReLU()
                if dropout:
                    d['dropout' + str(i)] = nn.Dropout(p=dropout)
        return d

    def forward(self, x):
        return F.log_softmax(self.model(x.view(-1, 784)), dim=0)

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, type(approx_Linear)):
                m.reset_parameters()



def get_args():
    parser = ArgumentParser()
    parser.add_argument(
        '--n_epoch', type=int, default=20, help='number of training epochs')
    parser.add_argument(
        '--d_hidden', type=int, default=500, help='dimension of hidden layers')
    parser.add_argument(
        '--n_layer',
        type=int,
        default=3,
        help='number of layers, including the output layer')
    parser.add_argument(
        '--d_minibatch', type=int, default=50, help='size of minibatches')
    parser.add_argument(
        '--dropout', type=float, default=0.1, help='dropout rate')
    parser.add_argument(
        '--random_seed', type=int, default=42, help='random seed')
    parser.add_argument(
        '--dev', type=str, default="cuda", help='specify "cuda" or "cpu"')
    parser.set_defaults()
    args = parser.parse_args()
    if args.dev == 'cuda':
        if torch.cuda.is_available():
            print('using cuda')
            args.device = torch.device('cuda')
        else:
            print('requested cuda device but cuda unavailable. using cpu instead')
            args.device = torch.device('cpu')
    else:
        print('using cpu')
        args.device = torch.device('cpu')

    return args


def main():
    args = get_args()
    trn, dev, tst = get_mnist()

    # change the sys.stdout to a file object to write the results to the file
    group = TestGroup(
        args,
        trn,
        args.d_minibatch,
        args.d_hidden,
        args.n_layer,
        args.dropout,
        dev,
        tst,
        cudatensor=False,
        file=sys.stdout)

    # results may be different at each run
    #with torch.autograd.profiler.profile(use_cuda=True) as prof:
    group.run()
    #print(prof.key_averages())

if __name__ == '__main__':
    main()
