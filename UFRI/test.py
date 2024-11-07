import ttach as tta
import multiprocessing.pool as mpp
import multiprocessing as mp
import time
from train import *
import argparse
from pathlib import Path
import cv2
import numpy as np
import torch
import matplotlib.colors as mcolors
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import pdb
import os
import torch
from thop import profile
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(device)

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def label2rgb(mask):
    h, w = mask.shape[0], mask.shape[1]
    mask_rgb = np.zeros(shape=(h, w, 3), dtype=np.uint8)
    #pdb.set_trace()
    mask_convert = mask[np.newaxis, :, :]
    mask_rgb[np.all(mask_convert == 0, axis=0)] = [240, 240, 240]#road
    mask_rgb[np.all(mask_convert == 1, axis=0)] = [248, 214, 236]#commerial
    mask_rgb[np.all(mask_convert == 2, axis=0)] = [231, 241, 209]#education
    mask_rgb[np.all(mask_convert == 3, axis=0)] = [173, 236, 185]#park
    mask_rgb[np.all(mask_convert == 4, axis=0)] = [191, 191, 191]
    mask_rgb[np.all(mask_convert == 5, axis=0)] = [225, 225, 245]#hospital
    mask_rgb[np.all(mask_convert == 6, axis=0)] = [255, 222, 154]
    mask_rgb[np.all(mask_convert == 7, axis=0)] = [32, 164, 48]#greenbelt
    mask_rgb[np.all(mask_convert == 8, axis=0)] = [204, 240, 255]#building
    mask_rgb[np.all(mask_convert == 9, axis=0)] = [60, 120, 200]

    return mask_rgb


def img_writer(inp):
    (mask,  mask_id, rgb) = inp
    if rgb:
        mask_name_tif = mask_id + '.png'
        mask_tif = label2rgb(mask)
        cv2.imwrite(mask_name_tif, mask_tif)
    else:
        mask_png = mask.astype(np.uint8)
        mask_name_png = mask_id + '.png'
        cv2.imwrite(mask_name_png, mask_png)


def get_args():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("-c", "--config_path", default="config/mf2s.py")
    arg("-o", "--output_path", type=Path, help="Path where to save resulting masks.", required=True)
    arg("-t", "--tta", help="Test time augmentation.", default=None, choices=[None, "d4", "lr"])
    arg("--rgb", help="whether output rgb images", action='store_true')
    return parser.parse_args()


def main():
    seed_everything(42)
    args = get_args()
    config = py2cfg(args.config_path)
    
    args.output_path.mkdir(exist_ok=True, parents=True) 
    #pdb.set_trace()
    model = Supervision_Train.load_from_checkpoint(os.path.join(config.weights_path, config.test_weights_name+'.ckpt'), config=config)
   
    model.cuda()

    model.eval()
    evaluator = Evaluator(num_class=config.num_classes)
    evaluator.reset()
    if args.tta == "lr":
        transforms = tta.Compose(
            [
                tta.HorizontalFlip(),
                tta.VerticalFlip()
            ]
        )
        model = tta.SegmentationTTAWrapper(model, transforms)
    elif args.tta == "d4":
        transforms = tta.Compose(
            [
                tta.HorizontalFlip(),
                tta.VerticalFlip(),
                tta.Rotate90(angles=[90]),
                tta.Scale(scales=[0.5, 0.75, 1.0, 1.25, 1.5], interpolation='bicubic', align_corners=False)
            ]
        )
        model = tta.SegmentationTTAWrapper(model, transforms)

    test_dataset = config.test_dataset

    with torch.no_grad():
        test_loader = DataLoader(
            test_dataset,
            batch_size=1,
            num_workers=1,
            pin_memory=True,
            drop_last=False,
        )
        results = []
        y_true = torch.empty(1)
        y_pred = torch.empty(1)

        for input in tqdm(test_loader):
            # raw_prediction NxCxHxW
            raw_predictions = model(input['img'].to(device))#to(device)

            image_ids = input["img_id"]
            masks_true = input['gt_semantic_seg']

            raw_predictions = nn.Softmax(dim=1)(raw_predictions)
            predictions = raw_predictions.argmax(dim=1)

            for i in range(raw_predictions.shape[0]):
                mask = predictions[i].cpu().numpy()
                evaluator.add_batch(pre_image=mask, gt_image=masks_true[i].cpu().numpy())
                mask_name = image_ids[i]
                results.append((mask, str(args.output_path / mask_name), args.rgb))
                y_pred = torch.cat((y_pred,predictions[i].view(-1).cpu()),0)
                y_true = torch.cat((y_true,masks_true[i].view(-1).cpu()),0)


    iou_per_class = evaluator.Intersection_over_Union()
    f1_per_class = evaluator.F1()
    OA = evaluator.OA()
    for class_name, class_iou, class_f1 in zip(config.classes, iou_per_class, f1_per_class):
        print('F1_{}:{}, IOU_{}:{}'.format(class_name, class_f1, class_name, class_iou))
    print('F1:{}, mIOU:{}, OA:{}'.format(np.nanmean(f1_per_class[:-1]), np.nanmean(iou_per_class[:-1]), OA))

    t0 = time.time()
    mpp.Pool(processes=mp.cpu_count()).map(img_writer, results)
    t1 = time.time()
    img_write_time = t1 - t0
    print('images writing spends: {} s'.format(img_write_time))
  
    # input1 = torch.randn(4, 3, 512, 512).to(device)
    # flops, params = profile(model, inputs=(input1, ))
    # print('FLOPs = ' + str(flops/1000**3) + 'G')
    # print('Params = ' + str(params/1000**2) + 'M')
    class_names = ['Road', 'Commercial', 'Education', 'Park', 'Industrial', 'Hospital', 'Water', 'Greenbelt', 'Buliding', 'Landuse']
    confusion_mat = confusion_matrix(y_pred[1:],y_true[1:])
    confusion_mat_percent = confusion_mat.astype('float') / confusion_mat.sum(axis=1)[:, np.newaxis] 
    plt.figure(figsize=(12, 10))
    norm = mcolors.Normalize(vmin=0, vmax=1)
    plt.imshow(confusion_mat_percent, interpolation='nearest', cmap=plt.cm.Blues)
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    thresh = confusion_mat_percent.max() / 2 + 0.3

    for i in range(confusion_mat_percent.shape[0]):
        for j in range(confusion_mat_percent.shape[1]):
            plt.text(j, i, f'{confusion_mat_percent[i, j]:.2f}',  
                    horizontalalignment="center",
                    color="white" if confusion_mat_percent[i, j] > thresh else "black")
   
    plt.savefig(str(args.output_path / "confusion.png"))

if __name__ == "__main__":
    main()
