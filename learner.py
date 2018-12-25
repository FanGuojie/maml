from torch import nn
class Learner(nn.Module):
    """
    meaning:
        inner learner, cnn+mlp    
    :param:
        nWay: num of output kind
    :return:
        logit: prediction
    """
    def __init__(self, nWay):
        super(Learner, self).__init__()
        self.nWay = nWay
        self.conv1 = nn.Sequential(  # input shape (1, 28, 28)
            nn.Conv2d(
                in_channels=1,  # input height
                out_channels=32,  # n_filters
                kernel_size=5,  # filter size
                stride=1,  # filter movement/step
                padding=2,
                # if want same width and length of this image after Conv2d, padding=(kernel_size-1)/2 if stride=1
            ),  # output shape (32, 28, 28)
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),  # activation
            nn.MaxPool2d(kernel_size=2),  # choose max value in 2x2 area, output shape (16, 14, 14)
        )
        self.conv2 = nn.Sequential(  # input shape (32, 14, 14)
            nn.Conv2d(32, 64, 5, 1, 2),  # output shape (64, 14, 14)
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),  # activation
            nn.MaxPool2d(2),  # output shape (64, 7, 7)
        )
        self.conv3 = nn.Sequential(  # input shape (64, 7, 7)
            nn.Conv2d(64, 128, 5, 1, 2),  # output shape (128, 7, 7)
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),  # activation
        )
        self.conv4 = nn.Sequential(  # input shape (64, 7, 7)
            nn.Conv2d(128, 128, 5, 1, 2),  # output shape (128, 7, 7)
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),  # activation
        )
        self.conv5 = nn.Sequential(  # input shape (128, 7, 7)
            nn.Conv2d(128, 256, 5, 1, 2),  # output shape (256, 7, 7)
            nn.BatchNorm2d(256),
            nn.Dropout(0.8),
            nn.ReLU(inplace=True),  # activation
        )
        # self.conv6 = nn.Sequential(  # input shape (256, 7, 7)
        #     nn.Conv2d(256, 512, 5, 1, 2),  # output shape (512, 7, 7)
        #     nn.BatchNorm2d(512),
        #     nn.Dropout(0.7),
        #     nn.ReLU(inplace=True),  # activation
        # )
        self.out = nn.Sequential(
            nn.Linear(256 * 7 * 7, 100),
            nn.Dropout(0.7),
            nn.ReLU(inplace=True),  # activation
            nn.Linear(100, self.nWay),  # fully connected layer, output nWay classes
        )

    def forward(self, x):
        x = self.conv1(x.float())
        x = self.conv2(x.float())
        x = self.conv3(x.float())
        x = self.conv4(x.float())
        x = self.conv5(x.float())
        # x = self.conv6(x.float())
        x = x.view(x.size(0), -1)  # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        output = self.out(x)
        return output, x  # return x for visualization