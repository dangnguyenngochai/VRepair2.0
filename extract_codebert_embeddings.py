from transformers import AutoTokenizer, AutoModel
import transformers
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import shutil

PATH = '/home/lgm/VRepair2.0/param_sweep_tgt/00_parameter_sweep/'
def read_opennmt_vocab(vocab_file=None):
    if vocab_file is None:
        vul_vocab = []
        with open(PATH+'data.vocab.src') as file:
            data = file.readlines()
            for line in data:
                line = line.strip().split('\t')
                vul_vocab.append(line[0])
        with open(PATH+'data.vocab.tgt') as file:
            data = file.readlines()
            for line in data:
                line = line.strip().split('\t')
                vul_vocab.append(line[0])
        vul_vocab = list(set(vul_vocab))
        with open(PATH+'vocab.txt', 'w+') as file:
            file.write('\n'.join(vul_vocab))
        return vul_vocab

    elif vocab_file and type(vocab_file) is str:
        with open(vocab_file, 'r') as file:
            data = file.read().splitlines()
            return data

def updating_embedding_file_v1(tokenizer, model, vul_vocab=[]):
    if len(vul_vocab)==0:
        vul_vocab = read_opennmt_vocab()
        vocab_size = len(vul_vocab)
        print("Total vocab size", len(vul_vocab))
        # vocab_dict = {}
        vocab_df = pd.DataFrame()
        vocab_df['tok'] = vul_vocab[:vocab_size]
    else:
        vocab_size = len(vul_vocab)
        vocab_df = pd.DataFrame()
        vocab_df['tok'] = vul_vocab

    with open('codebert_embeddings.txt', 'a') as ebf:
        vul_embs = torch.Tensor(size=(vocab_size, 768))
        for idx, vocab in tqdm(enumerate(vul_vocab[:vocab_size])):
            code_tokens=tokenizer.tokenize(vocab)
            tokens_ids=tokenizer.convert_tokens_to_ids(code_tokens)
            context_embeddings = model(torch.tensor(tokens_ids)[None,:])[0]
            context_embeddings = torch.mean(context_embeddings, dim=1).ravel()
            vul_embs[idx,:] = context_embeddings
            del code_tokens
            del tokens_ids
            del context_embeddings

        vul_embs = vul_embs.detach().numpy()
        vul_df = pd.DataFrame(vul_embs)
        df = pd.concat([vocab_df, vul_df], axis=1)
        df.to_csv(ebf, sep=' ', header=False, index=False)
        del vul_embs
        del vul_df
        del df

def fetch_vocab_chunks(vul_vocab, chunk_size=3000):
    for chunk_start_idx in range(0, len(vul_vocab), chunk_size):
        chunk_vocab = vul_vocab[chunk_start_idx:chunk_start_idx+chunk_size]
        yield chunk_vocab

if __name__=="__main__":
    start_it = 0
    tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
    model = AutoModel.from_pretrained("microsoft/codebert-base")
    print("#############\n\n")
    print("Start: Extracting emb")
    vul_vocab = read_opennmt_vocab()
    emb_file = open('codebert_embeddings.txt', 'a+')
    emb_file.close()

    for it, chunk_vocab in enumerate(fetch_vocab_chunks(vul_vocab)):
        chunk_vocab_df = pd.DataFrame(chunk_vocab)
        print('Iteration ', it)
        if it < start_it:
            print("SKIP")
            continue
        shutil.copyfile('codebert_embeddings.txt',f'codebert_embeddings_{it}_bk.txt'.format(it))       
        updating_embedding_file_v1(tokenizer, model, vul_vocab=chunk_vocab)
        del chunk_vocab_df
    print("Done: Extracting emb")

  