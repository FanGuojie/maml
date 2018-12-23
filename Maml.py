import torch
from torch import nn


class Meta(nn.Module):
    """
        meaning: meta learner
        :param:
        args
        :return:
        model
    """
    def __init__(self, args):
        super(Meta, self).__init__()
        self.epoch=args.epoch
        self.nWay=args.nWay
        self.kShot=args.kShot
        self.kQuery=args.kQuery
        self.imgSize=args.imgSize
        self.taskNum=args.taskNum
        self.meta_lr=args.meta_lr
        self.update_lr=args.update_lr
        self.updateStep=args.updateStep
        self.updateTestStep=args.updateTestStep


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
    leraner=Meta(args)