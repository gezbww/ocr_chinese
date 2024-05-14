import os
from PIL import Image
import numpy as np
import time
import torch
import argparse
from glob import glob
from sklearn.model_selection import train_test_split
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from dataset import decode_text
from tqdm import tqdm
from datasets import load_metric
from torchvision import models


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
    os.environ["CUDA_VISIBLE_DEVICES"] = -1
    #processor = TrOCRProcessor.from_pretrained(args.cust_data_init_weights_path)
    #vocab = processor.tokenizer.get_vocab()
    model=models.resnet18(pretrained=True)
    vocab=model.
    vocab_inp = {vocab[key]: key for key in vocab}
    model = VisionEncoderDecoderModel.from_pretrained(args.cust_data_init_weights_path)
    model.eval()
    model.cuda()

    vocab = processor.tokenizer.get_vocab()
    vocab_inp = {vocab[key]: key for key in vocab}
    import pandas as pd
    df_test = pd.read_table("", sep='\t')
    df_test.columns = ["text"]
    pred_str, label_str = [], []
    for p in tqdm(test_paths):
        img = Image.open(p).convert('RGB')
        #txt_p = os.path.splitext(p)[0] + '.txt'

        #with open(txt_p) as f:
        #    label = f.read().strip()
        index=test_paths.index(p)
        label=df_test['text'][index]
        pixel_values = processor([img], return_tensors="pt").pixel_values

        with torch.no_grad():
            generated_ids = model.generate(pixel_values[:, :, :].cuda())

        generated_text = decode_text(generated_ids[0].cpu().numpy(), vocab, vocab_inp)
        pred_str.append(generated_text)
        label_str.append(label)

    res = compute_metrics(pred_str, label_str)
    print(res)
