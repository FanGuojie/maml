import torch
from torch import nn
from learner import Learner
from torch.nn import functional as F
import gc
from copy import deepcopy


class Meta(nn.Module):
    """
        meaning: meta learner
        :param:
        args
        :return:
        model
    """

    def __init__(self, args,config):
        super(Meta, self).__init__()
        self.nWay = args.nWay
        self.kShot = args.kShot
        self.kQuery = args.kQuery
        self.taskNum = args.taskNum
        self.meta_lr = args.meta_lr
        self.update_lr = args.update_lr
        self.updateStep = args.updateStep
        self.updateTestStep = args.updateTestStep
        self.net = Learner(args.nWay)
        self.config=config
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr= self.meta_lr)
        # optimize all net parameters
        self.loss_func = F.cross_entropy  # the target label is not one-hotted

    def forward(self, xSpt, ySpt, xQry, yQry):
        """
        meaning:
            meta learning
        :param:
            xSpt : [b, surpportSize, ]
        :return:
            accurates
        """
        querySize = xQry.size(1)
        # init loss acc
        losses_q = [0 for _ in range(self.updateStep+1)]
        corrects = [0 for _ in range(self.updateStep+1)]

        for i in range(self.taskNum):

            # this is the loss and accuracy before the first update
            with torch.no_grad():
                self.net.eval()
                logits_q = map(lambda x:  self.net(x), xQry[i])
                loss_q = F.cross_entropy(logits_q, yQry[i])
                losses_q[0] += loss_q

                pred = F.softmax(logits, dim=1).argmax(dim=1)
                correct = torch, eq(pred, yQry[i])
                corrects[0] += correct
            grad=[]
            netCp = deepcopy(self.net)
            optimizerCp = torch.optim.Adam(netCp.parameters(), lr=update_lr)
            for k in range(self.updateStep):
                # 1. run teh i-th task and compute loss for k=1~K-1
                netCp.train()
                logits = map(lambda x:  netCp(x), xSpt[i])
                loss = self.loss_func(logits, ySpt)
                # 2. update weights
                optimizerCp.zero_grad()  # clear gradients for this training step
                loss.backward()  # backpropagation, compute gradients
                optimizerCp.step()  # apply gradients
                # 3. test
                with torch.no_grad():
                    netCp.eval()
                    logits_q = map(lambda x:  self.net(x), xQry[i])
                    loss_q = F.cross_entropy(logits_q, yQry[i])
                    losses_q[k] += loss_q

                    pred = F.softmax(logits, dim=1).argmax(dim=1)
                    correct = torch, eq(pred, yQry[i])
                    corrects[k] += correct

                # free memory
                
                del netCp
                gc.collect()

            # end of all tasks
            # sum over all losses on query set across all tasks
            self.net.train()
            loss_q = losses_q[-1]/taskNum
            self.optimizer.zero_grad()  # clear gradients for this training step
            loss_q.backward()  # backpropagation, compute gradients
            self.optimizer.step()  # apply gradients

            accs = np.array(corrects) / (querySize * taskNum)

            return accs


if __name__ == '__main__':
    import argparse
    arg = argparse.ArgumentParser()
    arg.add_argument('--epoch', type=int, help='epoch numer', default=10000)
    arg.add_argument('--nWay', type=int, help='n way', default=5)
    arg.add_argument('--kShot', type=int,
                     help='k shot for support set', default=1)
    arg.add_argument('--kQuery', type=int,
                     help='k shot for query set', default=15)
    arg.add_argument('--imgSize', type=int, help='image size', default=28)
    arg.add_argument('--taskNum', type=int, help='task numer', default=32)
    arg.add_argument('--meta_lr', type=int,
                     help='meta-levvel outer learning rate', default=1e-3)
    arg.add_argument('--update_lr', type=int,
                     help='task-level inner update learning rate', default=0.4)
    arg.add_argument('--updateStep', type=int,
                     help='task-level inner update steps', default=5)
    arg.add_argument('--updateTestStep', type=int,
                     help='update steps for test', default=10)
    args = arg.parse_args()
    leraner = Meta(args)
