from torch.utils.data import Dataset
import os

class Omniglot(Dataset):
    """docstring for Omniplot"""

    def __init__(self,path,transform=None):
    	"""
    	meaning:
    		transfer images to npy file
    	params:
    		images path
    		transform: handle image
    	return:
    		omniglot.npy
    	"""
    	self.transform=transform
    	self.items=getItems(path)
    	self.classes=getClasses(self.items)

    def __getitem__(self,index):
    	filename=self.items[index][0]
    	img=str.join('//',[self.items[index][2],filename])

    	target=self.classes[self.items[index][1]]
    	if self.transform is not None:
    		img=self.transform(img)

    	return img,target


def getItems(path):
	items=[]
	for (root,dirs,files) in os.walk(path):
		for f in files:
			if(f.endswith("png")):
				r=root.split('\\')#windows
				# r=root.split('/')#ubuntu
				lr=len(r)
				items.append((f,r[lr-2]+"\\"+r[lr-1],root))
	print("== Found %d items " % len(items)) #32460
	# items[0]:
	# ('0765_17.png', 'Mkhedruli_(Georgian)\\character37', 'omniglot\\processed\\images_background\\Mkhedruli_(Georgian)\\character37')

	return items
 
def getClasses(items):
	classes={}
	for i in items:
		if i[1]not in classes:
			classes[i[1]]=len(classes)
	print("== Found %d classes" % len(classes))
	return classes


if __name__ == '__main__':
	processedFloder = 'processed'
	root = 'omniglot'
	processedPath = os.path.join(root, processedFloder)
	data=Omniglot(processedPath)
