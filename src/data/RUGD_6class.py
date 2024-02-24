import argparse
import os.path as osp
import numpy as np
import mmcv
from PIL import Image
import pathlib as pl
from tqdm import tqdm
from multiprocessing import Pool
import os

class ImageProcessor:
    def __init__(self, rudg_dir="/mnt/d/Data/RUGD", annotation_folder="lbls/"):
        self.rudg_dir = rudg_dir
        self.annotation_folder = annotation_folder
        self.CLASSES = ("dirt", "sand", "grass", "tree", "pole", "water", "sky", 
                "vehicle", "container/generic-object", "asphalt", "gravel", 
                "building", "mulch", "rock-bed", "log", "bicycle", "person", 
                "fence", "bush", "sign", "rock", "bridge", "concrete", "picnic-table")
        self.PALETTE = [ [ 108, 64, 20 ], [ 255, 229, 204 ],[ 0, 102, 0 ],[ 0, 255, 0 ],
                [ 0, 153, 153 ],[ 0, 128, 255 ],[ 0, 0, 255 ],[ 255, 255, 0 ],[ 255, 0, 127 ],
                [ 64, 64, 64 ],[ 255, 128, 0 ],[ 255, 0, 0 ],[ 153, 76, 0 ],[ 102, 102, 0 ],
                [ 102, 0, 0 ],[ 0, 255, 128 ],[ 204, 153, 255 ],[ 102, 0, 204 ],[ 255, 153, 204 ],
                [ 0, 102, 102 ],[ 153, 204, 255 ],[ 102, 255, 255 ],[ 101, 101, 11 ],[ 114, 85, 47 ] ]
        self.Groups = [2, 2, 4, 5, 5, 2, 0, 5, 5, 1, 2, 5, 2, 3, 4, 5, 5, 5, 4, 5, 3, 5, 1, 5]
        self.color_id = {tuple(c):i for i, c in enumerate(self.PALETTE)}
        self.color_id[tuple([0, 0, 0])] = 255

    def rgb2mask(self, img):
        h, w, c = img.shape
        out = np.ones((h, w, c)) * 255
        for i in range(h):
            for j in range(w):
                if tuple(img[i, j]) in self.color_id:
                    out[i][j] = self.color_id[tuple(img[i, j])]
                else:
                    print("unknown color, exiting...")
                    exit(0)
        return out

    def raw_to_seq(self, seg):
        h, w = seg.shape
        out = np.zeros((h, w))
        for i in range(len(self.Groups)):
            out[seg==i] = self.Groups[i]
        out[seg==255] = 0
        return out


    def read_save_img(self, l):
        lbl6_path = pl.Path(self.rudg_dir,"6class",l.strip())
        if lbl6_path.exists():
            return
        lbl_fn = pl.Path(self.rudg_dir,self.annotation_folder,l.strip())
        gt_semantic_seg = mmcv.imread(lbl_fn).squeeze().astype(np.uint8)
        gt_semantic_seg[:, :] = gt_semantic_seg[:, :, ::-1]
        grayscale = self.rgb2mask(gt_semantic_seg)
        grayscale = grayscale[:, :, 0]
        orig_lbl_path = pl.Path(self.rudg_dir,"orig",l.strip())
        orig_lbl_path.parents[0].mkdir(parents=True, exist_ok=True)
        lbl6_path.parents[0].mkdir(parents=True, exist_ok=True)
        mmcv.imwrite(grayscale, orig_lbl_path)
        lbl6 = self.raw_to_seq(grayscale)
        mmcv.imwrite(lbl6, lbl6_path)

    def process_images(self, filename):
        with open(osp.join(self.rudg_dir, filename), 'r') as r:
            lines = r.readlines()
            with Pool(os.cpu_count()-2) as p:
                for _ in tqdm(p.imap_unordered(self.read_save_img, [(l) for l in lines]), total=len(lines)):
                    pass
if __name__ == "__main__":
    processor = ImageProcessor()
    processor.process_images('train.txt')
    processor.process_images('val.txt')
    processor.process_images('test.txt')
    print("successful")