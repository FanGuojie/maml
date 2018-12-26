import gc

import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch import optim
import numpy as np

from innerNet import Learner
from copy import deepcopy


class Meta(nn.Module):
    """
    Meta Learner
    """

    def __init__(self, args):
        """

        :param args:
        """
        super(Meta, self).__init__()

        self.update_lr = args.update_lr
        self.meta_lr = args.meta_lr
        self.nWay = args.nWay
        self.kShot = args.kShot
        self.kQuery = args.kQuery
        self.taskNum = args.taskNum
        self.updateStep = args.updateStep
        self.updateTestStep = args.updateTestStep

        self.net = Learner(args.nWay)
        self.meta_optim = optim.Adam(self.net.parameters(), lr=self.meta_lr)

    def clip_grad_by_norm_(self, grad, max_norm):
        """
        in-place gradient clipping.
        :param grad: list of gradients
        :param max_norm: maximum norm allowable
        :return:
        """

        total_norm = 0
        counter = 0
        for g in grad:
            param_norm = g.data.norm(2)
            total_norm += param_norm.item() ** 2
            counter += 1
        total_norm = total_norm ** (1. / 2)

        clip_coef = max_norm / (total_norm + 1e-6)
        if clip_coef < 1:
            for g in grad:
                g.data.mul_(clip_coef)

        return total_norm / counter

    def forward(self, xSpt, ySpt, xQry, yQry):
        """

        :param xSpt:   [b, setsz, c_, h, w]
        :param ySpt:   [b, setsz]
        :param xQry:   [b, querySize, c_, h, w]
        :param yQry:   [b, querySize]
        :return:
        """
        taskNum, setsz, c_, h, w = xSpt.size()
        querySize = xQry.size(1)

        losses_q = [0 for _ in range(self.updateStep + 1)]  # losses_q[i] is the loss on step i
        corrects = [0 for _ in range(self.updateStep + 1)]

        for i in range(taskNum):

            # 1. run the i-th task and compute loss for k=0
            logits = self.net(xSpt[i], vars=None, bn_training=True)
            loss = F.cross_entropy(logits, ySpt[i])
            grad = torch.autograd.grad(loss, self.net.parameters())
            tmpWeights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, self.net.parameters())))

            # this is the loss and accuracy before first update
            with torch.no_grad():
                # [setsz, nway]
                logits_q = self.net(xQry[i], self.net.parameters(), bn_training=True)
                loss_q = F.cross_entropy(logits_q, yQry[i])
                losses_q[0] += loss_q

                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                correct = torch.eq(pred_q, yQry[i]).sum().item()
                corrects[0] = corrects[0] + correct

            # this is the loss and accuracy after the first update
            with torch.no_grad():
                # [setsz, nway]
                logits_q = self.net(xQry[i], tmpWeights, bn_training=True)
                loss_q = F.cross_entropy(logits_q, yQry[i])
                losses_q[1] += loss_q
                # [setsz]
                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                correct = torch.eq(pred_q, yQry[i]).sum().item()
                corrects[1] = corrects[1] + correct

            for k in range(1, self.updateStep):
                # 1. run the i-th task and compute loss for k=1~K-1
                logits = self.net(xSpt[i], tmpWeights, bn_training=True)
                loss = F.cross_entropy(logits, ySpt[i])
                # 2. compute grad on theta_pi
                grad = torch.autograd.grad(loss, tmpWeights)
                # 3. theta_pi = theta_pi - train_lr * grad
                tmpWeights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, tmpWeights)))

                logits_q = self.net(xQry[i], tmpWeights, bn_training=True)
                # loss_q will be overwritten and just keep the loss_q on last update step.
                loss_q = F.cross_entropy(logits_q, yQry[i])
                losses_q[k + 1] += loss_q

                with torch.no_grad():
                    pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                    correct = torch.eq(pred_q, yQry[i]).sum().item()  # convert to numpy
                    corrects[k + 1] = corrects[k + 1] + correct

        # end of all tasks
        # sum over all losses on query set across all tasks
        loss_q = losses_q[-1] / taskNum

        # optimize theta parameters
        self.meta_optim.zero_grad()
        loss_q.backward()
        # print('meta update')
        # for p in self.net.parameters()[:5]:
        #   print(torch.norm(p).item())
        self.meta_optim.step()

        accs = np.array(corrects) / (querySize * taskNum)
        return accs

    def finetunning(self, xSpt, ySpt, xQry, yQry):
        """

        :param xSpt:   [setsz, c_, h, w]
        :param ySpt:   [setsz]
        :param xQry:   [querySize, c_, h, w]
        :param yQry:   [querySize]
        :return:
        """
        assert len(xSpt.shape) == 4

        querySize = xQry.size(0)

        corrects = [0 for _ in range(self.updateTestStep + 1)]

        # in order to not ruin the state of running_mean/variance and bn_weight/bias
        # we finetunning on the copied model instead of self.net
        net = deepcopy(self.net)

        # 1. run the i-th task and compute loss for k=0
        logits = net(xSpt)
        loss = F.cross_entropy(logits, ySpt)
        grad = torch.autograd.grad(loss, net.parameters())
        tmpWeights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, net.parameters())))

        # this is the loss and accuracy before first update
        with torch.no_grad():
            # [setsz, nway]
            logits_q = net(xQry, net.parameters(), bn_training=True)
            # [setsz]
            pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
            # scalar
            correct = torch.eq(pred_q, yQry).sum().item()
            corrects[0] = corrects[0] + correct

        # this is the loss and accuracy after the first update
        with torch.no_grad():
            # [setsz, nway]
            logits_q = net(xQry, tmpWeights, bn_training=True)
            # [setsz]
            pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
            # scalar
            correct = torch.eq(pred_q, yQry).sum().item()
            corrects[1] = corrects[1] + correct

        for k in range(1, self.updateTestStep):
            # 1. run the i-th task and compute loss for k=1~K-1
            logits = net(xSpt, tmpWeights, bn_training=True)
            loss = F.cross_entropy(logits, ySpt)
            # 2. compute grad on theta_pi
            grad = torch.autograd.grad(loss, tmpWeights)
            # 3. theta_pi = theta_pi - train_lr * grad
            tmpWeights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, tmpWeights)))

            logits_q = net(xQry, tmpWeights, bn_training=True)
            # loss_q will be overwritten and just keep the loss_q on last update step.
            loss_q = F.cross_entropy(logits_q, yQry)

            with torch.no_grad():
                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                correct = torch.eq(pred_q, yQry).sum().item()  # convert to numpy
                corrects[k + 1] = corrects[k + 1] + correct

        del net
        gc.collect();
        # memory recycle
        accs = np.array(corrects) / querySize

        return accs


def main():
    pass


if __name__ == '__main__':
    main()
