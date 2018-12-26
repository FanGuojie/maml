import torch
import os
from torch import nn
import numpy as np
from initData import Prehandler
import argparse
from tensorboardX import SummaryWriter
from Meta import Meta
name='tmp'
writer = SummaryWriter(name)


def main(args):
    torch.manual_seed(222)
    torch.cuda.manual_seed_all(222)
    np.random.seed(222)

    print(args)

    device = torch.device('cuda', 0)
    device_ids = [0,1,2,3]
    maml = Meta(args,device_ids)
    maml=maml.to(device)
    # maml = nn.DataParallel(maml,device_ids=device_ids)

    data = Prehandler(batchSize=args.taskNum,
                      nWay=args.nWay,
                      kShot=args.kShot,
                      kQuery=args.kQuery,
                      imgSize=args.imgSize)

    for step in range(args.epoch):

        xSpt, ySpt, xQry, yQry = data.next()
        xSpt, ySpt, xQry, yQry = torch.from_numpy(xSpt).to(device), torch.from_numpy(ySpt).to(device), \
                                 torch.from_numpy(xQry).to(
                                     device), torch.from_numpy(yQry).to(device)

        accs = maml(xSpt, ySpt, xQry, yQry)
        if step % 50 == 0:
            print('step:', step, '\ttraining acc:', accs)
            writer.add_scalar('Train/acc', accs[-1], step)
        if step % 500 == 0:
            xSpt, ySpt, xQry, yQry = data.next('test')
            xSpt, ySpt, xQry, yQry = torch.from_numpy(xSpt).to(device), torch.from_numpy(ySpt).to(device), \
                                     torch.from_numpy(xQry).to(
                                         device), torch.from_numpy(yQry).to(device)
            accs = maml(xSpt, ySpt, xQry, yQry,'test')
            print('Test acc:', accs)
            writer.add_scalar('Test/acc', accs[-1], step)
        if step % 5000 == 0 and step != 0:
            torch.save(maml.state_dict(), name+'.pkl')


def load(model):
    model.load_state_dict(torch.load(name+'.pkl'))


# -----多GPU训练的模型读取的代码，multi-gpu training---------
def load_single(network):
    save_path = 'maml.pkl'
    state_dict = torch.load(save_path)
    # create new OrderedDict that does not contain `module.`
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        namekey = k[7:]  # remove `module.`
        new_state_dict[namekey] = v
    # load params
    network.load_state_dict(new_state_dict)
    return network


# ----------单GPU训练读取模型的代码，single gpu training-----------------
def load_multi(network):
    save_path = 'maml.pkl'
    network.load_state_dict(torch.load(save_path))
    return network


if __name__ == '__main__':
    arg = argparse.ArgumentParser()
    arg.add_argument('--epoch', type=int, help='epoch numer', default=15000)
    arg.add_argument('--nWay', type=int, help='n way', default=30)
    arg.add_argument('--kShot', type=int,
                     help='k shot for support set', default=3)
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
