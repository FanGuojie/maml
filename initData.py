from    omniglot import Omniglot
import  torchvision.transforms as transforms
from    PIL import Image
import  os.path
import  numpy as np


class Prehandler:

    def __init__(self,  batchSize, nWay, kShot, kQuery, imgSize):
        """
        Different from mnistNShot, the
        :param root:
        :param batchSize: task num
        :param nWay:
        :param kShot:
        :param k_qry:
        :param imgSize:
        """
        root='data'
        self.resize = imgSize
        if not os.path.isfile(os.path.join(root, 'omniglot.npy')):
            # if root/data.npy does not exist, just download it
            self.x = Omniglot(root, download=True,
                              transform=transforms.Compose([lambda x: Image.open(x).convert('L'),
                                                            lambda x: x.resize((imgSize, imgSize)),
                                                            lambda x: np.reshape(x, (imgSize, imgSize, 1)),
                                                            lambda x: np.transpose(x, [2, 0, 1]),
                                                            lambda x: x/255.])
                              )

            temp = dict()  # {label:img1, img2..., 20 imgs, label2: img1, img2,... in total, 1623 label}
            for (img, label) in self.x:
                if label in temp.keys():
                    temp[label].append(img)
                else:
                    temp[label] = [img]

            self.x = []
            for label, imgs in temp.items():  # labels info deserted , each label contains 20imgs
                self.x.append(np.array(imgs))

            # as different class may have different number of imgs
            self.x = np.array(self.x).astype(np.float)  # [[20 imgs],..., 1623 classes in total]
            # each character contains 20 imgs
            print('data shape:', self.x.shape)  # [1623, 20, 84, 84, 1]
            temp = []  # Free memory
            # save all dataset into npy file.
            np.save(os.path.join(root, 'omniglot.npy'), self.x)
            print('write into omniglot.npy.')
        else:
            # if data.npy exists, just load it.
            self.x = np.load(os.path.join(root, 'omniglot.npy'))
            print('load from omniglot.npy.')

        # [1623, 20, 84, 84, 1]
        # TODO: can not shuffle here, we must keep training and test set distinct!
        self.x_train, self.x_test = self.x[:1200], self.x[1200:]

        # self.normalization()

        self.batchSize = batchSize
        self.n_cls = self.x.shape[0]  # 1623
        self.nWay = nWay  # n way
        self.kShot = kShot  # k shot
        self.kQuery = kQuery  # k query
        assert (kShot + kQuery) <=20

        # save pointer of current read batch in total cache
        self.indexes = {"train": 0, "test": 0}
        self.datasets = {"train": self.x_train, "test": self.x_test}  # original data cached
        print("data: train", self.x_train.shape, "test", self.x_test.shape)

        self.datasets_cache = {"train": self.load_data_cache(self.datasets["train"]),  # current epoch data cached
                               "test": self.load_data_cache(self.datasets["test"])}


    def load_data_cache(self, data_pack):
        """
        Collects several batches data for N-shot learning
        :param data_pack: [cls_num, 20, 84, 84, 1]
        :return: A list with [support_set_x, support_set_y, target_x, target_y] ready to be fed to our networks
        """
        #  take 5 way 1 shot as example: 5 * 1
        setsz = self.kShot * self.nWay
        querysz = self.kQuery * self.nWay
        data_cache = []

        # print('preload next 50 caches of batchSize of batch.')
        for sample in range(10):  # num of episodes

            x_spts, y_spts, x_qrys, y_qrys = [], [], [], []
            for i in range(self.batchSize):  # one batch means one set

                x_spt, y_spt, x_qry, y_qry = [], [], [], []
                selected_cls = np.random.choice(data_pack.shape[0], self.nWay, False)

                for j, cur_class in enumerate(selected_cls):

                    selected_img = np.random.choice(20, self.kShot + self.kQuery, False)

                    # meta-training and meta-test
                    x_spt.append(data_pack[cur_class][selected_img[:self.kShot]])
                    x_qry.append(data_pack[cur_class][selected_img[self.kShot:]])
                    y_spt.append([j for _ in range(self.kShot)])
                    y_qry.append([j for _ in range(self.kQuery)])

                # shuffle inside a batch
                perm = np.random.permutation(self.nWay * self.kShot)
                x_spt = np.array(x_spt).reshape(self.nWay * self.kShot, 1, self.resize, self.resize)[perm]
                y_spt = np.array(y_spt).reshape(self.nWay * self.kShot)[perm]
                perm = np.random.permutation(self.nWay * self.kQuery)
                x_qry = np.array(x_qry).reshape(self.nWay * self.kQuery, 1, self.resize, self.resize)[perm]
                y_qry = np.array(y_qry).reshape(self.nWay * self.kQuery)[perm]

                # append [sptsz, 1, 84, 84] => [b, setsz, 1, 84, 84]
                x_spts.append(x_spt)
                y_spts.append(y_spt)
                x_qrys.append(x_qry)
                y_qrys.append(y_qry)


            # [b, setsz, 1, 84, 84]
            x_spts = np.array(x_spts).astype(np.float32).reshape(self.batchSize, setsz, 1, self.resize, self.resize)
            y_spts = np.array(y_spts).astype(np.int).reshape(self.batchSize, setsz)
            # [b, qrysz, 1, 84, 84]
            x_qrys = np.array(x_qrys).astype(np.float32).reshape(self.batchSize, querysz, 1, self.resize, self.resize)
            y_qrys = np.array(y_qrys).astype(np.int).reshape(self.batchSize, querysz)

            data_cache.append([x_spts, y_spts, x_qrys, y_qrys])

        return data_cache

    def next(self, mode='train'):
        """
        Gets next batch from the dataset with name.
        :param mode: The name of the splitting (one of "train", "val", "test")
        :return:
        """
        # update cache if indexes is larger cached num
        if self.indexes[mode] >= len(self.datasets_cache[mode]):
            self.indexes[mode] = 0
            self.datasets_cache[mode] = self.load_data_cache(self.datasets[mode])

        next_batch = self.datasets_cache[mode][self.indexes[mode]]
        self.indexes[mode] += 1

        return next_batch





if __name__ == '__main__':

    pass

