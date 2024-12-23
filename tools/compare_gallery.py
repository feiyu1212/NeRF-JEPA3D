import os
import cv2
import glob
import numpy as np

def run(cate, src1, src2, target):
    fs = glob.glob(f'/data2/nerf_data/srn_result_gif/{cate}/*/{target}.png')
    img_list = []
    for f in fs[:]:
        try:
            name = f.split('/')[-2]
            img_vit = cv2.imread(f)
            img_src1 = cv2.imread(f'/data2/nerf_data/srn_{cate}_result/srn_{cate}/{src1}/{name}/{target}.png')
            img_src2 = cv2.imread(f'/data2/nerf_data/srn_{cate}_result/srn_{cate}/{src2}/{name}/{target}.png')
            img = np.concatenate([img_vit[:, :128*4], img_src1, img_src2, img_vit[:, -128:]], axis=1)
            cv2.imwrite(f'/data2/nerf_data/gallery/{cate}/{src1}-{src2}-{target}{name}.png', img)
        except:
            pass


for cate in ['cars', 'chairs']:
    for src1 in [72, 88, 104]:
        for src2 in [72, 88, 104]:
            for target in [80, 96, 112, 128, 144, 160, 176]:
                if abs(src1 - target) <= abs(src2 - target):
                    continue
                target = '{:06d}'.format(target)
                print(cate, src1, src2, target)
                run(cate, src1, src2, target)



def run2(src1, src2):
    # Source  /  SoftRas  /  DVR  /  SRN  /  Ours  /  GT
    fs = glob.glob(f'/data2/nerf_data/NMR_result/nmr/{src1}/*/0*.png')
    img_list = []
    for f in fs[:]:
        try:
            name = f.split('/')[-2]
            target = f.split('/')[-1].split('.')[0]
            img_vit = cv2.imread(f'/data2/nerf_data/nmr_result_gif/{name}/{target}.png')
            img_vit = cv2.cvtColor(img_vit, cv2.COLOR_BGR2RGB)
            img_src0 = cv2.imread(f'/data2/nerf_data/NMR_result/nmr/0/{name}/{target}.png')
            img_src1 = cv2.imread(f'/data2/nerf_data/NMR_result/nmr/{src1}/{name}/{target}.png')
            img_src2 = cv2.imread(f'/data2/nerf_data/NMR_result/nmr/{src2}/{name}/{target}.png')
            img = np.concatenate([img_vit[:, :64], img_vit[:, 64*3:64*5], img_src0, img_src1, img_src2, img_vit[:, -64:]], axis=1)
            cv2.imwrite(f'/data2/nerf_data/gallery/nmr/{src1}-{src2}-{target}{name}.png', img)
        except:
            pass



for src1 in [1]:
    for src2 in [2]:
        print(src1,src2)
        run2(src1, src2)









import glob
import numpy as np
fs = glob.glob('/Users/liucancheng/Documents/select_gallery/nmr/*.png')
imgs = [cv2.imread(f) for f in fs]
img = np.concatenate(imgs, axis=0)
cv2.imwrite('/Users/liucancheng/Documents/select_gallery/nmr.png', img)




