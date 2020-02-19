# Code to produce colored segmentation output in Pytorch
# April 2019
# Kailun Yang
#######################

import numpy as np
import torch
import os
import importlib

from PIL import Image
from argparse import ArgumentParser

from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, CenterCrop, Normalize, Scale, Resize
from torchvision.transforms import ToTensor, ToPILImage

from dataset import cityscapes
from erfnet_pspnet import Net
from transform import Relabel, ToLabel, Colorize

from dataset_loader import *
from collections import OrderedDict , namedtuple

import visdom

NUM_CHANNELS = 3
#NUM_CLASSES = 26

image_transform = ToPILImage()
input_transform_cityscapes = Compose([
    Resize((512,1024*1),Image.BILINEAR),
    ToTensor(),
])

def main(args, get_dataset):

    modelpath = args.loadDir + args.loadModel
    weightspath = args.loadDir + args.loadWeights

    print ("Loading model: " + modelpath)
    print ("Loading weights: " + weightspath)

    NUM_CLASSES=get_dataset.num_labels
    model = Net(NUM_CLASSES, args.em_dim, args.resnet)

    model = torch.nn.DataParallel(model)
    if (not args.cpu):
        model = model.cuda()

    def load_my_state_dict(model, state_dict):  #custom function to load model when not all dict elements
        own_state = model.state_dict()
        
        for a in own_state.keys():
            print(a)
        for a in state_dict.keys():
            print(a)
        print('-----------')
        
        for name, param in state_dict.items():
            if name not in own_state:
                 continue
            own_state[name].copy_(param)
        
        return model

    model = load_my_state_dict(model, torch.load(weightspath))
    print ("Model and weights LOADED successfully")

    model.eval()

    if(not os.path.exists(args.datadir)):
        print ("Error: datadir could not be loaded")
    
    loader = DataLoader(cityscapes(args.datadir, input_transform_cityscapes, subset=args.subset),
        num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False)

    # For visualizer:
    # must launch in other window "python3.6 -m visdom.server -port 8097"
    # and access localhost:8097 to see it
    if (args.visualize):
        vis = visdom.Visdom()
    
    with torch.set_grad_enabled(False):
        for step, (images, filename) in enumerate(loader):
            
            images = images.cuda()

            outputs = model(images, enc=False)
            outputs = outputs['MAP']
      
            label = outputs[0].cpu().max(0)[1].data.byte()
            label_color = Colorize()(label.unsqueeze(0))
            filenameSave = "./save_color/" + filename[0].split("leftImg8bit/")[1]
            os.makedirs(os.path.dirname(filenameSave), exist_ok=True)
            label_save = ToPILImage()(label_color)
            label_save.save(filenameSave) 
            
            label = outputs[1].cpu().max(0)[1].data.byte()
            label_color = Colorize()(label.unsqueeze(0))
            filenameSave = "./save_color/" + filename[1].split("leftImg8bit/")[1]
            os.makedirs(os.path.dirname(filenameSave), exist_ok=True)
            label_save = ToPILImage()(label_color)
            label_save.save(filenameSave) 

            print (step, filenameSave)

class load_data():

	def __init__(self, args):

		## First, a bit of setup
		dinf = namedtuple('dinf' , ['name' , 'n_labels' , 'func' , 'path', 'size'])
		self.metadata = [dinf('IDD', 27, IDD_Dataset , 'idd' , (1024,512)),
					dinf('CS' , 20 , CityscapesDataset , 'cityscapes' , (1024,512)) ,
					dinf('MAP', 26, MapillaryDataset , 'Mapillary', (1024,512)),
					dinf('ADE', 51, ADE20KDataset , 'MITADE', (512,512)),
                                        dinf('IDD20K', 27, IDD20KDataset , 'IDD20K', (1024,512)),
					dinf('CVD' , 12, CamVid, 'CamVid' , (480,360)),
					dinf('SUN', 38, SunRGB, 'sun' , (640,480)),
					dinf('NYU_S' , 14, NYUv2_seg, 'NYUv2_seg' , (320,240)),
					]

		self.num_labels = {entry.name:entry.n_labels for entry in self.metadata if entry.name in args.datasets}

		self.d_func = {entry.name:entry.func for entry in self.metadata}
		basedir = args.basedir
		self.d_path = {entry.name:basedir+entry.path for entry in self.metadata}
		self.d_size = {entry.name:entry.size for entry in self.metadata}

	def __call__(self, name, split='train', num_images=None, mode='labeled', file_path=False):

		transform = self.Img_transform(name, self.d_size[name] , split)
		return self.d_func[name](self.d_path[name] , split, transform, file_path, num_images , mode)

	def Img_transform(self, name, size, split='train'):

		assert (isinstance(size, tuple) and len(size)==2)

		if name in ['CS' , 'IDD' , 'MAP', 'ADE', 'IDD20K']:

			if split=='train':
				t = [
					transforms.Resize(size),
					transforms.RandomCrop((512,512)), 
					transforms.RandomHorizontalFlip(),
					transforms.ToTensor()]
			else:
				t = [transforms.Resize(size),
					transforms.ToTensor()]

			return transforms.Compose(t)

		if split=='train':
			t = [transforms.Resize(size),
				transforms.RandomHorizontalFlip(),
				transforms.ToTensor()]
		else:
			t = [transforms.Resize(size),
				transforms.ToTensor()]

		return transforms.Compose(t)

def parse_args():
    parser = ArgumentParser()

    parser.add_argument('--state')

    parser.add_argument('--loadDir',default="../trained_models/")
    parser.add_argument('--loadWeights', default="erfpspnet.pth")
    parser.add_argument('--loadModel', default="erfnet_pspnet.py") #can be erfnet_pspnet, erfnet_apspnet and other networks
    parser.add_argument('--subset', default="val")  #can be val, test, train, pass, demoSequence

    parser.add_argument('--datadir', default="../dataset/")
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--cpu', action='store_true')

    parser.add_argument('--visualize', action='store_true')

    parser.add_argument('--em-dim', type=int, default=100)
    parser.add_argument('--resnet' , required=False)
    parser.add_argument('--basedir', required=True)
    parser.add_argument('--datasets' , nargs='+', required=True)

    args = parser.parse_args()

    return args

if __name__ == '__main__':

	try:
		args = parse_args()
		get_dataset = load_data(args)
		main(args, get_dataset)
	except KeyboardInterrupt:
		sys.exit(0)
