import re
import torch
import random
import numpy as np
import pandas as pd

from tqdm import tqdm
from sentence_transformers import util
from model.model import get_sentence_transformer, get_llm_model
from prompt.prompt_format import get_document_prompt, Prompter

def remove_gif(df):
    df = df[df['Image URL'].apply(lambda x: isinstance(x, str) and 'gif' not in x)]
    df = df.reset_index(drop = True)

    return df

def remove_similar_tag(filename, model_name = 'snunlp/KR-SBERT-V40K-klueNLI-augSTS', threshold = 0.8):
    model = get_sentence_transformer(model_name = model_name)

    delete_tag = []
    sim_change_tag = {}


    tag_data = df['tag_data']
    tag_data = list(tag_data)
    tag_tensor = model.encode(tag_data)
    sim_list = util.cos_sim(tag_tensor, tag_tensor)

    diag_size = sim_list.size(0)
    high = (sim_list >= threshold) & (~torch.eye(diag_size, dtype=torch.bool))

    batch = high.size(0)
    non_korean_pattern = re.compile(r'[^ㄱ-ㅎㅏ-ㅣ가-힣\s]')
    for i in tqdm(range(batch), total = batch):

        if bool(non_korean_pattern.search(tag_data[i])):
            continue

        for j in range(i+1, batch):
            if high[i, j]:
                if tag_data[i] not in sim_change_tag.keys():
                    sim_change_tag[tag_data[i]] = []
                sim_change_tag[tag_data[i]].append(tag_data[j])
                delete_tag.append(tag_data[j])

        for key, value in tqdm(sim_change_tag.items(), total = len(sim_change_tag)):
            for unit in value:
                df.replace(unit, key, inplace=True)

    return df

def remove_same_word(row):
    
    seen = set()
    new_row = []
    for value in row:
        if value in seen:
            new_row.append(np.nan)
        else:
            seen.add(value)
            new_row.append(value)

    return new_row

def divide_tag_to_five_sentimental(df):
    total_check = []
    for i in tqdm(range(len(df))):
        check = df.iloc[i].tolist()
        tmp = check[-8]
        check = check[1:-9]
        check = [x for x in check if type(x) == str]
        if type(tmp) == str and len(check) < 3:
            check.append(''.join(tmp.split()))
        check = ' '.join(check)
        total_check.append(check)
    return total_check

def llm_preprocessing(df):
    total_check = divide_tag_to_five_sentimental(df)
    prompt_dict = get_document_prompt()
    prompter = Prompter()
    model, tokenizer = get_llm_model()

    for i in tqdm(range(total_check, len(total_check))):
        

        prompt_dict['input'] = total_check[i]
        inputs = tokenizer(prompter.generate_prompt(prompt_dict['instruction'], prompt_dict['input']), return_tensors='pt').to('cuda')
        outputs = model.generate(**inputs, max_new_tokens = 200)
        llm_output = tokenizer.decode(outputs[0], skip_special_tokens=True)

        try:
            outputs = llm_output.split(f"{total_check[i]}")[1].split("### 입력 :")[0]
            print(f'\n{total_check[i]}')
        except:
            print('err')
        try:
            df.loc[i, '감정'] = outputs.split('감정 : ')[1].split('\n')[0]
            print(df.loc[i, '감정'])
        except:
            df.loc[i, '감정'] = np.nan
            print('err')

        try:
            df.loc[i, '상황'] = outputs.split('상황 : ')[1].split('\n')[0]
            print(df.loc[i, '상황'])
        except:
            df.loc[i, '상황'] = np.nan
            print('err')

        try:

            df.loc[i, '상황 유형'] = outputs.split('상황 유형 : ')[1].split('\n')[0]
            print(df.loc[i, '상황 유형'])
        except:
            df.loc[i, '상황 유형'] = np.nan
            print('err')

        try:
            df.loc[i, '의도'] = outputs.split('의도 : ')[1].split('\n')[0]
            print(df.loc[i, '의도'])
        except:
            df.loc[i, '의도'] = np.nan
            print('err')

        try:
            df.loc[i, '최종 상황'] = outputs.split('최종 상황 : ')[1].split('\n')[0]
            print(df.loc[i, '최종 상황'])
        except:
            df.loc[i, '최종 상황'] = np.nan
            print('err')

        return df
     

if __name__ == '__main__':
    """
    filename : file format should be 'csv'

    """
    filename = ""
    output_filename = ""

    df = pd.read_csv(filename)
    df = remove_gif(df)
    df = remove_similar_tag(df)
    df = df.apply(remove_same_word, axis=1, result_type='broadcast')
    df = llm_preprocessing(df)
    df.to_csv(output_filename, index=False)
