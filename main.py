import torch
import os
import numpy as np
from initData import Prehandler
import argparse
from tensorboardX import SummaryWriter
from Meta import Meta
writer = SummaryWriter('30waystep5_5shot')

def main(args):
    torch.manual_seed(222)
    torch.cuda.manual_seed_all(222)
    np.random.seed(222)

    print(args)

    device = torch.device('cuda', 1)
    maml = Meta(args).to(device)

    tmp = filter(lambda x: x.requires_grad, maml.parameters())
    num = sum(map(lambda x: np.prod(x.shape), tmp))

    data = Prehandler( batchSize=args.taskNum,
                         nWay=args.nWay,
                         kShot=args.kShot,
                         kQuery=args.kQuery,
                         imgSize=args.imgSize)

    for step in range(args.epoch):

        xSpt, ySpt, xQry, yQry = data.next()
        xSpt, ySpt, xQry, yQry = torch.from_numpy(xSpt).to(device), torch.from_numpy(ySpt).to(device), \
            torch.from_numpy(xQry).to(
                device), torch.from_numpy(yQry).to(device)

        # set traning=True to update running_mean, running_variance, bn_weights, bn_bias
        accs = maml(xSpt, ySpt, xQry, yQry)
        if step % 50 == 0:
            print('step:', step, '\ttraining acc:', accs)
            writer.add_scalar('Train/acc', accs[-1], step)
        if step % 500 == 0:
            accs = []
            for _ in range(1000 // args.taskNum):
                # test
                xSpt, ySpt, xQry, yQry = data.next('test')
                xSpt, ySpt, xQry, yQry = torch.from_numpy(xSpt).to(device), torch.from_numpy(ySpt).to(device), \
                    torch.from_numpy(xQry).to(
                        device), torch.from_numpy(yQry).to(device)

                # split to single task each time
                for xSptTmp, ySptTmp, xQryTmp, yQryTmp in zip(xSpt, ySpt, xQry, yQry):
                    test_acc = maml.finetunning(
                        xSptTmp, ySptTmp, xQryTmp, yQryTmp)
                    accs.append(test_acc)

            # [b, update_step+1]
            accs = np.array(accs).mean(axis=0).astype(np.float16)
            print('Test acc:', accs)
            writer.add_scalar('Test/acc', accs[-1], step)


if __name__ == '__main__':
    arg = argparse.ArgumentParser()
    arg.add_argument('--epoch', type=int, help='epoch numer', default=20000)
    arg.add_argument('--nWay', type=int, help='n way', default=30)
    arg.add_argument('--kShot', type=int,
                     help='k shot for support set', default=5)
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
                     help='update steps for test', default=5)
    args = arg.parse_args()

       main(args)
