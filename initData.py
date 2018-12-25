import os
from omniglot import Omniglot
import numpy as np
import torchvision.transforms as transforms
from PIL import Image


class Prehandler(object):
    urls = [
        'https://github.com/brendenlake/omniglot/raw/master/python/images_background.zip',
        'https://github.com/brendenlake/omniglot/raw/master/python/images_evaluation.zip'
    ]
    rawFolder = 'raw'
    processedFloder = 'processed'
    root = 'omniglot'
    processedPath = os.path.join(root, processedFloder)
    rawPath = os.path.join(root, rawFolder)

    def __init__(self, batchSize, nWay, kShot, kQuery, imgSize):
        """
        meaning: init data
        :param:
            nWay: n ways 
            kShot: k shot
            kQuery: num of query set elements
            imgSize: image size
        :return:
            prehandled data
        """
        super(Prehandler, self).__init__()
        self.resize = imgSize
        self.batchSize = batchSize
        self.nWay = nWay
        self.kShot = kShot
        self.kQuery = kQuery
        self.imgSize = imgSize

        if not os.path.exists(os.path.join(self.rawPath, 'images_background.zip')):
            # if not exist,then download data
            self.download()
        if not os.path.exists(os.path.join('omniglot', 'omniglot.npy')):
            self.data = Omniglot(self.processedPath,
                                 transform=transforms.Compose([lambda x: Image.open(x).convert('L'),
                                                               lambda x: x.resize(
                                                                   (imgSize, imgSize)),
                                                               lambda x: np.reshape(
                                                                   x, (imgSize, imgSize, 1)),
                                                               lambda x: np.transpose(
                                                                   x, [2, 0, 1]),
                                                               lambda x: x/255.]))
            temp = dict()  # {label: 20 imgs} 1623 items in total
            for (img, label) in self.data:
                if label in temp.keys():
                    temp[label].append(img)
                else:
                    temp[label] = [img]
            print(len(temp))
            self.data = []
            for label, imgs in temp.items():
                self.data.append(np.array(imgs))
            self.data = np.array(self.data).astype(np.float)

            print(self.data.shape)
            np.save(self.data, os.path.join(self.root, 'omniglot.npy'))
        else:
            print("loading data")
            self.data = np.load(os.path.join(self.root, 'omniglot.npy'))
            print("data shape:\t", self.data.shape)
        self.trainData, self.testData = self.data[:1200], self.data[1200:]
        self.index = {'train': 0, 'test': 0}
        self.datasets = {'train': self.trainData, 'test': self.testData}
        self.datasetsCache = {'train': self.loadData(
            self.trainData), 'test': self.loadData(self.testData)}
        # self.data=np.load(os.path.join('omniglot', 'omniglot.npy'))
        # print(data.shape)

    def loadData(self, data):
        """
        meaning:
            prepare batch data for N-shot learning
        :param:
            data : [classNum,20,84,84,1]
        :return:
            [xSpt,ySpt,xQry,yQry] ready to be fed to network
        """
        surpportSize = self.kShot*self.nWay
        querySize = self.kQuery*self.nWay
        dataCache = []

        for sample in range(10):  # num of episodes

            xSpts, ySpts, xQrys, yQrys = [], [], [], []
            for i in range(self.batchSize):
                xSpt, ySpt, xQry, yQry = [], [], [], []
                selectedClass = np.random.choice(
                    data.shape[0], self.nWay, False)

                for j, curClass in enumerate(selectedClass):
                    selectedImg = np.random.choice(
                        20, self.kShot+self.kQuery, False)

                    #meta-training and meta-test
                    xSpt.append(data[curClass][selectedImg[:self.kShot]])
                    xQry.append(data[curClass][selectedImg[self.kShot:]])
                    ySpt.append([j for _ in range(self.kShot)])
                    yQry.append([j for _ in range(self.kQuery)])

                # shuffle inside a batch
                shuffleIndex = np.random.permutation(self.nWay*self.kShot)
                xSpt = np.array(xSpt).reshape(
                    self.nWay*self.kShot, 1, self.resize, self.resize)
                ySpt = np.array(ySpt).reshape(
                    self.nWay*self.kShot)[shuffleIndex]
                shuffleIndex = np.random.permutation(self.nWay*self.kQuery)
                xQry = np.array(xQry).reshape(
                    self.nWay*self.kQuery, 1, self.resize, self.resize)
                yQry = np.array(yQry).reshape(
                    self.nWay*self.kQuery)[shuffleIndex]

                # append [surpportSize,1,85m85] => [b,setSizem1m84,84]
                xSpts.append(xSpt)
                xQrys.append(xQry)
                ySpts.append(ySpt)
                yQrys.append(yQry)
            # [b, surpportSize, 1, 84 , 84]
            xSpts = np.array(xSpts).astype(np.float32).reshape(
                self.batchSize, surpportSize, 1, self.resize,self.resize)
            ySpts = np.array(ySpts).astype(np.int).reshape(
                self.batchSize, surpportSize)
            # [b, querytSize, 1, 84 , 84]
            xQrys = np.array(xQrys).astype(
                np.float32).reshape(self.batchSize, querySize, 1, self.resize,self.resize)
            yQrys = np.array(yQrys).astype(
                np.int).reshape(self.batchSize, querySize)
            dataCache.append([xSpts, ySpts, xQrys, yQrys])
        return dataCache

    def next(self, mode='train'):
        """
        meaning:
            gets next batch from the dataset with name
        :param:
            mode: 'train' or 'test'
        :return:
            xSpt, ySpt, xQry, yQry
        """
        # update if indexes is larger than cache
        if self.index[mode] >= len(self.datasetsCache[mode]):
            self.index[mode] = 0
            self.datasetsCache[mode] = self.loadData(self.datasets[mode])

        nextBatch = self.datasetsCache[mode][self.index[mode]]
        self.index[mode] += 1
        return nextBatch

    def download(self):
        from six.moves import urllib
        import zipfile
        import errno

        try:
            os.makedirs(os.path.join(root, rawFolder))
            os.makedirs(os.path.join(root, processedFloder))
        except Exception as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise

        for url in self.urls:
            print("downloading : "+url)
            data = urllib.request.urlopen(url)
            filename = url.rpartition('/')[2]
            filePath = os.path.join(rawPath, filename)
            with open(filePath, 'wb') as f:
                f.write(data.read())
            zip = zipfile.ZipFile(filePath, 'r')
            zip.extractall(self.processedPath)
            zip.close()
        print("download finished")


if __name__ == '__main__':
    import argparse
    arg = argparse.ArgumentParser()
    arg.add_argument('--nWay', type=int, help='n way', default=5)
    arg.add_argument('--kShot', type=int,
                     help='k shot for support set', default=1)
    arg.add_argument('--kQuery', type=int,
                     help='k shot for query set', default=15)
    arg.add_argument('--imgSize', type=int, help='image size', default=28)
    args = arg.parse_args()
    data = Prehandler(nWay=args.nWay, kShot=args.kShot,
                      kQuery=args.kQuery, imgSize=args.imgSize)
