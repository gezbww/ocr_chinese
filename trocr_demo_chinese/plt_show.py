import matplotlib.pyplot as plt
import multiprocessing
from multiprocessing.pool import ThreadPool
from glob import glob
from resnet18 import res18
#import tensorflow as tf
from PIL import Image
from tqdm import tqdm
path=
path1=glob(path)
print(path1)

#pool = multiprocessing.Pool(10)



for i in tqdm(path1[1000:]):
    #print(i)
    res18(i)
