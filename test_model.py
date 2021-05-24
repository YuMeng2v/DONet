from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F
import numpy as np
import time
import math
from models import *
from PIL import Image
from dataloader import SecenFlowLoader as DA
import copy
parser = argparse.ArgumentParser(description='PSMNet')
parser.add_argument('--datapath', default=r'F:\dataset\SceneFlow\FlyingThings3D_images\FlyingThings3D_subset\val\image_clean\\',
                    help='select model')
parser.add_argument('--loadmodel', default=r'F:\U\毕设\mine\PSM_Net\PSMNet-master\checkpoint_1.tar',
                    help='loading model')
parser.add_argument('--model', default='stackhourglass',
                    help='select model')
parser.add_argument('--maxdisp', default=192,
                    help='maxium disparity')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

if args.model == 'stackhourglass':
    model = stackhourglass(args.maxdisp)
elif args.model == 'basic':
    model = basic(args.maxdisp)
else:
    print('no model')

model = nn.DataParallel(model, device_ids=[0])
model.cuda()

if args.loadmodel is not None:
    pretrained_dict = r"F:\U\毕设\mine\PSM_Net\PSMNet-master\trained\trained_dryad\checkpoint_19.tar"
    state_dict = torch.load(pretrained_dict)['state_dict']
    model = stackhourglass(192)
    new_dict = copy.deepcopy(state_dict)
    for each in list(new_dict.keys()):
        del new_dict[each]
    
    for name in state_dict:
        value = state_dict[name]
        name = name.replace('module.','',1)
        new_dict[name] = value
    #for name in new_dict:
        #name.replace('module','',1)
        #print(name,new_dict[name])
    model.load_state_dict(new_dict)
    

print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

def test(imgL,imgR):
    model.eval()

    if args.cuda:
        pass
        '''
        imgL = imgL.cuda()
        imgR = imgR.cuda()     
        '''
    with torch.no_grad():
        output = model(imgL,imgR)
    output = torch.squeeze(output).data.cpu().numpy()
    return output

def main():
    normal_mean_var = {'mean': [0.485, 0.456, 0.406],
                        'std': [0.229, 0.224, 0.225]}
    infer_transform = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize(**normal_mean_var)])    

    #for inx in range(len(test_left_img)):
        #1*3*256*512
    
    imgL_o = Image.open(r"F:\dataset\DRYAD2\LeftImg\0.png").convert('RGB')
    imgR_o = Image.open(r"F:\dataset\DRYAD2\RightImg\0.png").convert('RGB')

    imgL = infer_transform(imgL_o)
    imgR = infer_transform(imgR_o)         
    
    # pad to width and hight to 16 times
    if imgL.shape[1] % 16 != 0:
        times = imgL.shape[1]//16       
        top_pad = (times+1)*16 -imgL.shape[1]
    else:
        top_pad = 0

    if imgL.shape[2] % 16 != 0:
        times = imgL.shape[2]//16                       
        right_pad = (times+1)*16-imgL.shape[2]
    else:
        right_pad = 0    
    
    imgL = F.pad(imgL,(0,right_pad, top_pad,0)).unsqueeze(0)
    imgR = F.pad(imgR,(0,right_pad, top_pad,0)).unsqueeze(0)
    #左边扩充0列，右边扩充4列，上边扩充top列，下边0列，默认用0扩充，可以指定其他值
    #imgL = imgL.unsqueeze(0)
    #imgR = imgR.unsqueeze(0)
    start_time = time.time()
    pred_disp = test(imgL,imgR)
    print('time = %.2f' %(time.time() - start_time))
    
    if top_pad !=0 and right_pad==0:
        img = pred_disp[top_pad:,:]
        #img = pred_disp[top_pad:,:-right_pad]#
    elif top_pad ==0 and right_pad!=0:
        img = pred_disp[:,:-right_pad]
    else:
        img = pred_disp
    
    img = (img*256).astype('uint16')
    img = Image.fromarray(img)
    img.save('F:\\U\毕设\\mine\\PSM_Net\\PSMNet-master\\trained\\test_dryad_19.png')


    """
    accuracy
    """
if __name__ == '__main__':
    main()
    