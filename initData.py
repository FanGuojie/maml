import os
from omniglot import Omniglot
import numpy as np
import  torchvision.transforms as transforms
from    PIL import Image
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

    def __init__(self, nWay, kShot, kQuery, imgSize):
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
        if not os.path.exists(os.path.join(self.rawPath, 'images_background.zip')):
            # if not exist,then download data
            self.download()
        if not os.path.exists(os.path.join('omniglot','omniglot.npy')):
            self.data=Omniglot(self.processedPath,
                transform=transforms.Compose([lambda x: Image.open(x).convert('L'),
                                                            lambda x: x.resize((imgSize, imgSize)),
                                                            lambda x: np.reshape(x, (imgSize, imgSize, 1)),
                                                            lambda x: np.transpose(x, [2, 0, 1]),
                                                            lambda x: x/255.]))
            temp=dict() #{label: 20 imgs} 1623 items in total
            print("ok?")
            for (img,label) in self.data:
                if label%10==0:
                    print("label:",label)
                if label in temp.keys():
                    temp[label].append(img)
                else:
                    temp[label]=[img]
            print(len(temp))
            self.data=[]
            for label,imgs in temp.items():
                self.data.append(np.array(imgs))
            self.data=np.array(self.data).astype(np.float)

            print(self.data.shape)
            np.save(os.path.join(root,'omniglot.npy'))
        # self.data=np.load(os.path.join('omniglot', 'omniglot.npy'))
        # print(data.shape)


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
            with open(filePath,'wb') as f:
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