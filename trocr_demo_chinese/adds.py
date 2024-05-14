from add_token import gen_vocabs
from PIL import Image
import time
import torch
import argparse
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from dataset import decode_text
import os
import json

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='trocr fine-tune训练')
    parser.add_argument('--cust_data_init_weights_path', default='./cust-data/weights', type=str,
                        help="初始化训练权重，用于自己数据集上fine-tune权重")
    parser.add_argument('--CUDA_VISIBLE_DEVICES', default='-1', type=str, help="GPU设置")
    parser.add_argument('--test_img',
                        default='', type=str,
                        help="img path")

    args = parser.parse_args()

    processor = TrOCRProcessor.from_pretrained(args.cust_data_init_weights_path)
    vocab = gen_vocabs('./cust-data/scut_chars_list.txt',processor).get_vocab()
    print(vocab)
    print(len(vocab))
    vocab_inp = {vocab[key]: key for key in vocab}
    model = VisionEncoderDecoderModel.from_pretrained(args.cust_data_init_weights_path)
    model.config.vocab_size = len(vocab)
    with open(os.path.join(args.cust_data_init_weights_path, "vocab1.json"), "w", encoding='utf-8') as f:
        f.write(json.dumps(vocab, ensure_ascii=False))