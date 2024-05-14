from transformers import TrOCRProcessor
from transformers import AutoTokenizer,AutoModel
import pandas as pd
def gen_vocabs(path,processor):
    #processor = TrOCRProcessor.from_pretrained('./cust-data/weights')
    tokenizer = processor.tokenizer
    #processor = TrOCRProcessor.from_pretrained(args.cust_data_init_weights_path)
    vocab = processor.tokenizer.get_vocab()
    print(len(vocab))
    tokenizer.add_tokens("赅")

    #path='./cust-data/scut_chars_list.txt'
    #path='./cust-data/scut_chars_list.txt'
    data=pd.read_fwf(path,encoding='utf-8')

    for i in data.iloc[:,0]:
        tokenizer.add_tokens(i)
    ts=tokenizer.get_vocab()
    print("-------------------------")
    print(len(ts))
    return tokenizer