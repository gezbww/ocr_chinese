import easyocr
import os
from PIL import Image
import numpy as np
import time
import torch
import argparse
from glob import glob
from sklearn.model_selection import train_test_split
#from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from dataset import decode_text
from tqdm import tqdm
from datasets import load_metric

cer_metric = load_metric("./cer.py")


def compute_metrics(pred_str, label_str):
    """
    计算cer,acc
    :param pred:
    :return:
    """
    cer = cer_metric.compute(predictions=pred_str, references=label_str)
    acc = [pred == label for pred, label in zip(pred_str, label_str)]
    acc = sum(acc) / (len(acc) + 0.000001)
    return {"cer": cer, "acc": acc}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='trocr 模型评估')
    parser.add_argument('--cust_data_init_weights_path', default='', type=str,
                        help="初始化训练权重，用于自己数据集上fine-tune权重")
    parser.add_argument('--CUDA_VISIBLE_DEVICES', default='-1', type=str, help="GPU设置")
    parser.add_argument('--dataset_path', default='', type=str,
                        help="img path")
    parser.add_argument('--random_state', default=None, type=int, help="用于训练集划分的随机数")

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.CUDA_VISIBLE_DEVICES
    test_paths = glob(args.dataset_path)
    import pandas as pd
    df_test = pd.read_table("",header=None, sep='\t')
    df_test.columns = ["text"]
    pred_str, label_str = [], []
    reader = easyocr.Reader(['ch_sim', 'en'], gpu=True)
    for p in tqdm(test_paths):
        img = Image.open(p).convert('RGB')
        #txt_p = os.path.splitext(p)[0] + '.txt'

        #with open(txt_p) as f:
        #    label = f.read().strip()
        index=test_paths.index(p)
        label=df_test['text'][index]
       # easyocr.

        # need to run only once to load model into memory
        result = reader.readtext(p, detail=0)
        print(result)
        pred_str.append(result)
        label_str.append(label)

    res = compute_metrics(pred_str, label_str)
    print(res)

#设置识别中英文两种语言

