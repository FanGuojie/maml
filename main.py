import argparse
import torch
from Maml import Meta
from initData import Prehandler

def main(args):
    print(args)
    device = torch.device('cuda', 0)
    maml = Meta(args).to(device)
    data = Prehandler(nWay=args.nWay, kShot=args.kShot,
                      kQuery=args.kQuery, imgSize=args.imgSize)


if __name__ == '__main__':
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
    main(args)
