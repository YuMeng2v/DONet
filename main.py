from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import time
import math
from dataloader import listflowfile as lt
from dataloader import SecenFlowLoader as DA
from models import *
import copy

parser = argparse.ArgumentParser(description='PSMNet')
parser.add_argument('--maxdisp', type=int ,default=192,
                    help='maxium disparity')
parser.add_argument('--model', default='stackhourglass',
                    help='select model')
parser.add_argument('--datapath', default=r'D:\tar',
                    help='datapath')
parser.add_argument('--epochs', type=int, default=10,
                    help='number of epochs to train')
parser.add_argument('--loadmodel', default= r'C:\Users\Yumeng\Desktop\PSMNet-master-3DCNN\trained\pretrained_sceneflow.tar',
                    help='load model')
parser.add_argument('--savemodel', default='./',
                    help='save model')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

################################################################
#global t_ori1 
t_ori1 = 1 #初始分母
#global t_ori2
t_ori2 = 1 #初始分母，task2,occlusion
################################################################

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

all_left_img, all_right_img, all_left_disp,all_left_occlusion, test_left_img, test_right_img, test_left_disp,test_left_occlusion = lt.dataloader(args.datapath)

TrainImgLoader = torch.utils.data.DataLoader(
         DA.myImageFloder(all_left_img,all_right_img,all_left_disp,all_left_occlusion ,True), 
         batch_size= 1, shuffle= True, num_workers= 1, drop_last=False) #modified, num_workers  = 8 batch = 12

TestImgLoader = torch.utils.data.DataLoader(
         DA.myImageFloder(test_left_img,test_right_img,test_left_disp,test_left_occlusion, False), 
         batch_size= 8, shuffle= False, num_workers= 0, drop_last=False) #num_workers = 4


if args.model == 'stackhourglass':
    model = stackhourglass(args.maxdisp)
elif args.model == 'basic':
    model = basic(args.maxdisp)
else:
    print('no model')

if args.cuda:
    #model = nn.DataParallel(model)#modified
    model.cuda()
    #pass
if args.loadmodel is not None:
    print('Load pretrained model')
    #pretrain_dict = torch.load()
    #model.load_state_dict(pretrain_dict['state_dict'])
    
    state_dict = torch.load(args.loadmodel)['state_dict']
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
    a = new_dict['classif3.2.weight']
    new_dict['classif4.0.weight'] = torch.rand(32,192,3,3)
    new_dict['classif4.2.weight'] = torch.rand(1,32,3,3)
    new_dict["dres5.0.0.weight"] = torch.rand(32,64,3,3,3)
    new_dict["dres5.0.1.weight"] = torch.rand(32)
    new_dict["dres5.2.0.weight"] = torch.rand(1,32,3,3,3)
    new_dict["dres5.2.1.weight"] = torch.rand(1)
    new_dict["dres5.0.1.bias"] =  torch.rand(32)
    new_dict["dres5.2.1.bias"]= torch.rand(1)
    new_dict["dres5.0.1.running_mean"] = torch.rand(32) 
    new_dict["dres5.0.1.running_var"] = torch.rand(32) 
    new_dict["dres5.2.1.running_mean"] = new_dict["dres5.2.1.running_var"] = torch.rand(1)
    model.load_state_dict(new_dict)


print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

optimizer = optim.Adam(model.parameters(), lr=0.00001, betas=(0.9, 0.999))
t_ori1 = 1
t_ori2 = 1
def train(imgL,imgR, disp_L,train_occ):
        model.train()
        #parameters
        disp_cret = 3
        occl_cret = 0.5
        T= N =2

        if args.cuda:
            #modified
            imgL, imgR, disp_true,train_occ = imgL.cuda(), imgR.cuda(), disp_L.cuda(),train_occ.cuda()
            #disp_true = disp_L # modified
            
       #---------
        mask = disp_true < args.maxdisp
        mask.detach_()
        #----
        optimizer.zero_grad()
        back_ok = 1 #是否反向传播
        if args.model == 'stackhourglass':
            output1, output2, output3, output4 = model(imgL,imgR)
            output1 = torch.squeeze(output1,1)
            output2 = torch.squeeze(output2,1)
            output3 = torch.squeeze(output3,1)
            loss1 = 0.5*F.smooth_l1_loss(output1[mask], disp_true[mask], size_average=True) + 0.7*F.smooth_l1_loss(output2[mask], disp_true[mask], size_average=True) + F.smooth_l1_loss(output3[mask], disp_true[mask], size_average=True) 
            #loss2 = F.smooth_l1_loss(output4[mask],train_occ[mask], size_average=True)#not L1_LOSS->
            #new calculation of loss2
            criterion = nn.BCELoss()
            loss2 = criterion(output4[mask],train_occ[mask])
            '''
            criterion = nn.BCELoss()
            try:
                loss2 = criterion(output4[mask],train_occ[mask])
            except:
                loss2 = torch.from_numpy(np.array(1))
                loss2 = loss2.float()
            '''
            #loss2 = F.smooth_l1_loss(output4[mask],train_occ[mask], size_average=True)
            print('loss1:',loss1,'loss2:',loss2)
            global t_ori1
            global t_ori2
            t_temp1 = loss1/t_ori1
            t_temp2 = loss2/t_ori2
            #w1 = N*torch.exp(t_temp1/T)/(torch.exp(t_temp1/T)+torch.exp(t_temp2/T))
            #w2 = N*torch.exp(t_temp2/T)/(torch.exp(t_temp1/T)+torch.exp(t_temp2/T))
            loss = loss1*0.001+loss2**0.999
            if loss1>50 or loss2>50 or np.isnan(np.array(loss1)) or np.isnan(np.array(loss2)):
                print('------------------here----------------')
                back_ok = 0
            #只是中间输出loss，并没有实质性意义，输出还是output3
            t_ori1 = loss1
            t_ori2 = loss2
            temp11 = ((output3-disp_true)<disp_cret)[mask]
            temp22 = torch.sum(temp11)
            temp33 = torch.sum(mask)
            '''
            出现nan
            '''
            #print('temp11:',temp11,'temp22:',temp22,'temp33:',temp33)
            accuracy1 = np.array(temp22.cpu())/np.array(temp33.cpu())
            temp11 = ((output4-train_occ)<occl_cret)[mask]
            temp22 = torch.sum(temp11)
            
            accuracy2 = np.array(temp22.cpu())/np.array(temp33.cpu())
    
        elif args.model == 'basic':
            output,output2 = model(imgL,imgR)
            output = torch.squeeze(output,1)
            loss1 = F.smooth_l1_loss(output[mask], disp_true[mask], size_average=True)
            #accuracy = (torch.sum((output[mask]-disp_true[mask])<accuracy_cret))/output.shape[0]/output.shape[1]
            loss2 = F.smooth_l1_loss(output2[mask],train_occ[mask], size_average=True)
            
            
            accuracy1 = (torch.sum(((output-disp_true)<accuracy_cret)[mask])/torch.sum(mask))
            accuracy2 = (torch.sum(((output2-train_occ)<occl_cret)[mask])/torch.sum(mask))
        if back_ok:
            print('back_ok')
            loss.backward(retain_graph=True)
            optimizer.step()
        else:
            print('back_not_ok')
        
        return loss.data,accuracy1,accuracy2

def test(imgL,imgR,disp_true,test_occ):

        model.eval()
  
        if args.cuda:
            imgL, imgR, disp_true,test_occ = imgL.cuda(), imgR.cuda(), disp_true.cuda(),test_occ.cuda()
        #---------
        mask = disp_true < 192
        #----

        if imgL.shape[2] % 16 != 0:
            times = imgL.shape[2]//16       
            top_pad = (times+1)*16 -imgL.shape[2]
        else:
            top_pad = 0

        if imgL.shape[3] % 16 != 0:
            times = imgL.shape[3]//16                       
            right_pad = (times+1)*16-imgL.shape[3]
        else:
            right_pad = 0  

        imgL = F.pad(imgL,(0,right_pad, top_pad,0))#data augmentation 数据加强
        imgR = F.pad(imgR,(0,right_pad, top_pad,0))

        with torch.no_grad():
            output3,occl = model(imgL,imgR)
            output3 = torch.squeeze(output3)
        
        if top_pad !=0:
            img = output3[:,top_pad:,:]
        else:
            img = output3

        if len(disp_true[mask])==0:
           loss = 0
        else:
           loss = F.l1_loss(img[mask],disp_true[mask]) #torch.mean(torch.abs(img[mask]-disp_true[mask]))  # end-point-error

        return loss.data.cpu()

def adjust_learning_rate(optimizer, epoch):
    lr = 0.00001 #0.001 - > 0.00001
    print(lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def main():
    start_full_time = time.time()
    t_ori1 = 1
    t_ori2 = 1
    for epoch in range(0, args.epochs):
        print('This is %d-th epoch' %(epoch))
        total_train_loss = 0
        adjust_learning_rate(optimizer,epoch)
        total_accuracy1 = 0
        total_accuracy2 = 0 #occlusion
	   ## training ##
        for batch_idx, (imgL_crop, imgR_crop, disp_crop_L,imgL_occ) in enumerate(TrainImgLoader):
            start_time = time.time()
            loss,accuracy1,accuracy2 = train(imgL_crop,imgR_crop, disp_crop_L,imgL_occ)
            
            para_temp = model.to('cpu').state_dict()
            model.to('cuda:0')
            #para_temp_cpu = para_temp
            para_temp_cpu_values =dict(para_temp).values()
            para_temp_cpu_list = list(para_temp_cpu_values)
            para_temp_array = np.array(para_temp_cpu_list[0])
            if np.isnan(np.array(para_temp_array)).any():
                print('---------------参数出现nan----------------')
            print('Iter %d training loss = %.3f ,accuracy_disp = %.3f,accuracy_occl = %.3f ,time = %.2f' %(batch_idx, loss,accuracy1,accuracy2 ,time.time() - start_time))
            
            total_train_loss += loss
            total_accuracy1+= accuracy1
            total_accuracy2 += accuracy2
        print('epoch %d total training loss = %.3f, accuracy_disp = %.3f,accuracy_occl = %.3f ' %(epoch, total_train_loss/len(TrainImgLoader),total_accuracy1/len(TrainImgLoader),total_accuracy2/len(TrainImgLoader)))

	   #SAVE
        savefilename = args.savemodel+'/checkpoint_'+str(epoch)+'.tar'
        torch.save({
		    'epoch': epoch,
		    'state_dict': model.state_dict(),
                    'train_loss': total_train_loss/len(TrainImgLoader),
		}, savefilename)
    print('full training time = %.2f HR' %((time.time() - start_full_time)/3600)) 
    """
	#------------- TEST ------------------------------------------------------------
    #还是算个loss，暂时不用了
    total_test_loss = 0
    total_accuracy = 0
    for batch_idx, (imgL, imgR, disp_L,test_occ) in enumerate(TestImgLoader):
	       test_loss = test(imgL,imgR, disp_L,test_occ)
	       print('Iter %d test loss = %.3f' %(batch_idx, test_loss))
	       total_test_loss += test_loss
    #print('total test loss =  %.3f' %(total_test_loss/len(TestImgLoader)))
	#print('total test loss = %d,len = %d, avg = %.3f' %(total_test_loss,len(TestImgLoader),total_test_loss/len(TestImgLoader)))
	#----------------------------------------------------------------------------------
	#SAVE test information
    savefilename = args.savemodel+'testinformation.tar'
    torch.save({
		    'test_loss': total_test_loss/len(TestImgLoader),
		}, savefilename)
    """

if __name__ == '__main__':
   main()
    
