from torch.utils.data import Dataset


class Omniplot(Dataset):
    """docstring for Omniplot"""
    urls = [
        'https://github.com/brendenlake/omniglot/raw/master/python/images_background.zip',
        'https://github.com/brendenlake/omniglot/raw/master/python/images_evaluation.zip'
    ]

    def __init__(self):
        
