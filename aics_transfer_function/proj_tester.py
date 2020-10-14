import os
from .options.base_options import BaseOptions
from .models import create_model
from .dataloader.cyclelarge_dataset import cyclelargeDataset
from .dataloader.cyclelargenopad_dataset import cyclelargenopadDataset
from skimage.measure import compare_psnr
from skimage.measure import compare_ssim

import tifffile
from tifffile import imread, imsave
import shutil

from glob import glob
import pdb
import numpy as np
import random

def extract_filename(filename, replace=False, old_name= '', rep_name =''):
    filename_rev = filename[::-1]
    idx = filename_rev.index('/')
    new = filename_rev[0:idx][::-1]
    if replace:
        new = new.replace(old_name, rep_name)
    return new


def arrange(opt,data,output,position):
    data = data[0,0].cpu().numpy()
    za,ya,xa = position
    patch_size = data.shape
    z1 = 0 if za==0 else patch_size[0]//4
    z2 = patch_size[0] if za+patch_size[0]==output.shape[0] else patch_size[0]//4 + patch_size[0]//2
    y1 = 0 if ya==0 else patch_size[1]//4
    y2 = patch_size[1] if ya+patch_size[1]==output.shape[1] else patch_size[1]//4 + patch_size[1]//2
    x1 = 0 if xa==0 else patch_size[2]//4
    x2 = patch_size[2] if xa+patch_size[2]==output.shape[2] else patch_size[2]//4 + patch_size[2]//2

    # zb,yb,xb = za + opt.size_out[0], ya + opt.size_out[1], xa + opt.size_out[2]
    zaa = za+z1; zbb = za + z2
    yaa = ya+y1; ybb = ya + y2
    xaa = xa+x1; xbb = xa + x2
    output[zaa:zbb,yaa:ybb,xaa:xbb] = data[z1:z2, y1:y2, x1:x2]

def imagePSNRAndSSIM(t,v,w=9):
    m = np.max(t)-np.min(t)
    ssim = compare_ssim(t,v,multichannel=False,win_size=w,data_range=m) # if the image has only one channel, multichannel=False, win_size is an odd numher, win_size>=3
    psnr = compare_psnr(t,v,data_range=m)
    return psnr,ssim

def test():
    opt = BaseOptions(isTrain=False).parse()  # get test options
    keep_alnum = lambda s: ''.join(e for e in s if e.isalnum())
    opt.batch_size = 1    # test code only supports batch_size = 1

    if opt.name in ['real2bin','denoise']:
        opt.fpath1, opt.fpath2 = opt.fpath2, opt.fpath1

    fpath1 = opt.fpath1
    fpath2 = opt.fpath2

    if opt.testfile == 'train':
        filenamesA = (sorted(glob(fpath1+'*.tiff'),key=keep_alnum)+sorted(glob(fpath1+'*.tif'),key=keep_alnum))[:opt.train_num]
        filenamesB = (sorted(glob(fpath2+'*.tiff'),key=keep_alnum)+sorted(glob(fpath2+'*.tif'),key=keep_alnum))[:opt.train_num]
    elif opt.testfile == 'all':
        filenamesA = (sorted(glob(fpath1+'*.tiff'),key=keep_alnum)+sorted(glob(fpath1+'*.tif'),key=keep_alnum))[:]
        filenamesB = (sorted(glob(fpath2+'*.tiff'),key=keep_alnum)+sorted(glob(fpath2+'*.tif'),key=keep_alnum))[:]
    elif opt.testfile in ['test','']:
        filenamesA = (sorted(glob(fpath1+'*.tiff'),key=keep_alnum)+sorted(glob(fpath1+'*.tif'),key=keep_alnum))[opt.train_num:]
        filenamesB = (sorted(glob(fpath2+'*.tiff'),key=keep_alnum)+sorted(glob(fpath2+'*.tif'),key=keep_alnum))[opt.train_num:]
    elif os.path.isdir(opt.testfile):
        print(f'load from folder: {opt.testfile}')
        filenamesA = (sorted(glob(opt.testfile+'/*.tiff'),key=keep_alnum)+sorted(glob(opt.testfile+'/*.tif'),key=keep_alnum)+sorted(glob(opt.testfile+'/*source.tif'),key=keep_alnum)+sorted(glob(opt.testfile+'/*source.tiff'),key=keep_alnum))[:]
        filenamesB = filenamesA
        print(filenamesA)
    elif os.path.isfile(opt.testfile):
        filenamesA = [opt.testfile,]
        filenamesB = filenamesA
    else:
        raise ValueError(f'--testfile should be [train|test|all|filename|directory|]. Invalid value: {opt.testfile}')

    if 'nopad' in opt.netG:
        dataset = cyclelargenopadDataset(opt)
    else:
        dataset = cyclelargeDataset(opt,aligned=True)
    opt.size_out = dataset.get_size_out()
    opt.up_scale = dataset.get_up_scale()
    dataset_size = len(dataset)    # get the number of images in the dataset.
    print('The number of training images = %d' % dataset_size)

    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers

    for fileA,fileB in zip(filenamesA,filenamesB):
        dataset.load_from_file([fileA,], [fileB,], num_patch=-1)
        position = dataset.positionB
        positionA = dataset.positionA
        rA = np.zeros(positionA[0]).astype('float32')
        rB = np.zeros(position[0]).astype('float32')
        fA = np.zeros(positionA[0]).astype('float32')
        fB = np.zeros(position[0]).astype('float32')
        fB0 = np.zeros(position[0]).astype('float32')
        recA = np.zeros(positionA[0]).astype('float32')
        recB = np.zeros(position[0]).astype('float32')
        ffB = np.zeros(position[0]).astype('float32')
        rrecB = np.zeros(position[0]).astype('float32')

        print(position)
        for i, data in enumerate(dataset):
            model.set_input(data)  # unpack data from data loader
            if opt.output_path == 'model':
                prefix = opt.continue_from + '/' +  fileA.split('/')[-1] + '_result/'
            elif opt.output_path == 'data':
                prefix = fileA + '_result/'
            else:
                prefix = opt.output_path + '/' +  fileA.split('/')[-1] + '_result/'
            if not os.path.exists(prefix):
                os.makedirs(prefix)
            shutil.copy(opt.config,prefix)

            za,ya,xa = position[i+1]
            print((za,ya,xa))
            zb,yb,xb = za + opt.size_out[0], ya + opt.size_out[1], xa + opt.size_out[2]
            print((zb,yb,xb))

            if opt.model == 'pix2pix':
                rA_i, rB_i, fB_i = model.test()           # run inference
                # psnr_list.append(psnr.psnr_local(rB_i[0,0].cpu().numpy(),fB_i[0,0].cpu().numpy()))
                arrange(opt,rA_i,rA,positionA[i+1])
                arrange(opt,rB_i,rB,position[i+1])
                arrange(opt,fB_i,fB,position[i+1])
            elif  opt.model in ['stn']:
                rA_i, rB_i, fB0_i, fB_i = model.test()           # run inference
                # psnr_list.append(psnr.psnr_local(rB_i[0,0].cpu().numpy(),fB0_i[0,0].cpu().numpy()))
                arrange(opt,rA_i,rA,positionA[i+1])
                arrange(opt,rB_i,rB,position[i+1])
                arrange(opt,fB0_i,fB0,position[i+1])
                arrange(opt,fB_i,fB,position[i+1])
 
        if opt.resizeA == 'upscale':
            from torch import from_numpy
            from  torch.nn.modules.upsampling import Upsample
            from aicsimageio import AICSImage
            input_reader = AICSImage(fileB) #STCZYX
            new_size = input_reader.shape[-3:]
            op = Upsample(size=new_size, mode='trilinear',align_corners=True)
            def resize_(img,op):
                img = np.expand_dims(img,0)
                img = np.expand_dims(img,0)
                img = op(from_numpy(img)).numpy()
                img = np.squeeze(img,axis=0)
                img = np.squeeze(img,axis=0)
                return img
            rB = resize_(rB,op)
            fB = resize_(fB,op)

        ###########################
        # Temp saving script
        filename_ori = extract_filename(fileA, replace=True, old_name='source.tif', rep_name='pred.tif')
        tif = tifffile.TiffWriter(opt.output_path + "/" + filename_ori, bigtiff=True)
        tif.save(fB, compress=9, photometric='minisblack', metadata=None)
        tif.close()
        #imsave(opt.output_path + "/" + filename_ori,fB)
        print(filename_ori + " saved")
        ###########################


if __name__ == '__main__':
    test()


