import numpy as np
import pandas as pd
import os
import json
import random
from nuscenes import NuScenes
from PIL import Image, ImageDraw
from typing import List

def visualize_crits(paths: List[str], detectors: List[str], sample_token: str, crit_weight: str = 'C'):
    """
    Return image visualizing LIDAR prediction CRIT values (overall of for a specific criticality weight)
    params:
        paths: list of paths to METRIC_SAMPLES (sample directory) for each object detector result dir
        detectors: names of detectors to compare results as name of directory where their respective results are stored in /results
        sample_token: sample to analyze
        crit_weight: which crit weight to visualize: one of c(rit)/t/r/d where 'c' is the overall crit value
    returns:
        Image (PIL) composite_img
    """    

    paths = [path + sample_token for path in paths]
    imgs = []
    size = (500, 500)

    if crit_weight == 'C':
        for p in paths:
            imgs.append(Image.open(p+'/LIDAR_CRIT.png').resize(size))
    elif crit_weight == 'T':
        for p in paths:
            imgs.append(Image.open(p+'/LIDAR_CRIT_T.png').resize(size))
    elif crit_weight == 'R':
        for p in paths:
            imgs.append(Image.open(p+'/LIDAR_CRIT_R.png').resize(size))
    elif crit_weight == 'D':
        for p in paths:
            imgs.append(Image.open(p+'/LIDAR_CRIT_D.png').resize(size))   
    else:
        print('crit_weight not recognized in visualization_helpers.py') 
        exit()
    
    composite_img = Image.new('RGB', (len(detectors)*imgs[0].width, imgs[0].height)) # len(detectors)x1 grid
    for i in range(len(imgs)):
        # compose grid
        composite_img.paste(imgs[i], (imgs[0].width*i, 0))
  


    drw =  ImageDraw.Draw(composite_img)
    for i in range(len(imgs)):
        # render detector names in frame
        drw.text((20 + i*imgs[0].width, 40), detectors[i], fill=(0, 0, 0))

    drw.text((40,20), "Parameters: " + crit_weight, fill=(255, 0, 0))

    
    return composite_img
    

def visualize_table(paths: List[str], detectors: List[str], sample_token: str, fname: str):

    dfs = [pd.read_csv(path + sample_token + fname, sep=" ; ", header=None) for path in paths]
    #data.columns = ["a", "b", "c", "etc."]
    return dfs
    