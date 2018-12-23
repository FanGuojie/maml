import os


class Prehandler(object):
    urls = [
        'https://github.com/brendenlake/omniglot/raw/master/python/images_background.zip',
        'https://github.com/brendenlake/omniglot/raw/master/python/images_evaluation.zip'
    ]

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
        if not os.path.exists(os.path.join('omniglot', 'omniglot.npy')):
            # if not exist,then download data
            self.download()
            self.data=

    def download(self):
        from six.moves import urllib
        import zipfile
        import errno
        rawFolder = 'raw'
        processedFloder = 'processed'
        root = 'omniglot'
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
            filePath = os.path.join(root, rawFolder, filename)
            with open(filePath,'wb') as f:
                f.write(data.read())
            processedPath = os.path.join(root, processedFloder, filename)
            zip = zipfile.ZipFile(filePath, 'r')
            zip.extractall(processedPath)
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