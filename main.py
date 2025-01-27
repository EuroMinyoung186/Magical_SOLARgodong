import random
import argparse
import numpy as np
import pandas as pd

from tqdm import tqdm
from model.model import get_llm_model,get_sentence_transformer
from prompt.prompt_format import get_query_prompt, get_rerank_prompt, Prompter
from sentence_transformers import util
from utils.util import visualize



def concat_tag(df):
    df = df.dropna(subset=['감정', '상황', '상황 유형', '의도'])
    name_list = df.columns.tolist()

    tags = [name for name in name_list[1:-9] if type(name) == str]
    df_llm = df[['감정', '상황', '상황 유형', '의도']]
    df_tag = df[tags]

    db_llm = [df_llm.iloc[i] for i in range(len(df_llm))]
    db_llm = [item for items in db_llm for item in items]
    db_tag = [df_tag.iloc[i] for i in range(len(df_tag))]
    db_tag2 = []
    for idx in range(len(db_tag)):
        imsi = []
        for name in db_tag[idx]:
            if type(name) is str:
                imsi.append(name)
        db_tag2.append(imsi)
    meme_prompt = [' '.join(db_tag2[i]) for i in range(len(db_tag2))]


    return db_llm, meme_prompt, meme_list

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--user_input', type=str, help="사용자 쿼리를 입력해주세요.")
    parser.add_argument('--filename', type=str, help="csv 파일일 경로를 입력해주세요요")

    args = parser.parse_args()

    prompt_dict = get_query_prompt()
    prompter = Prompter()
    prompt_dict['input'] = args.user_input
    llm, tokenizer = get_llm_model()
    sbert = get_sentence_transformer()
    df = pd.read_csv(args.filename)
    
    db_llm, meme_prompt = concat_tag(df)
    meme_list = df['Image URL'].tolist()
    inputs = tokenizer(prompter.generate_prompt(prompt_dict['instruction'], prompt_dict['input']), return_tensors='pt').to('cuda')
    outputs = llm.generate(**inputs, max_new_tokens = 200)
    llm_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    query_output = []

    outputs = llm_output.split(f"{args.user_input}")[1].split("### 입력 :")[0]
    query_output.append(outputs.split('감정 : ')[1].split('\n')[0])
    query_output.append(outputs.split('상황 : ')[1].split('\n')[0])
    query_output.append(outputs.split('문장의 유형 : ')[1].split('\n')[0])
    query_output.append(outputs.split('의도 : ')[1].split('\n')[0])
    query_output.append(outputs.split('최종 상황 : ')[1].split('\n')[0])

    query_embeddings = sbert.encode(query_output)

    score = 0.

    for idx in [0,1,2,3]:
      score += util.cos_sim(query_embeddings[idx], db_llm[idx::4])
    score = score.reshape(-1)

    index_list = enumerate(list(score))
    sorted_list = sorted(index_list, key=lambda x: x[1])
    top = sorted_list[-15:]
    llm_prompt = get_rerank_prompt(args.user_input, meme_prompt, top)

    inputs = tokenizer(llm_prompt, return_tensors = 'pt')
    outputs = llm.generate(**inputs, max_new_tokens = 8)
    llm_output = tokenizer.decode(outputs[0], skip_special_tokens = True)

    visualize(meme_list, [int(llm_output.split('IDX ')[-1])])


if __name__ == "__main__":
    main()
