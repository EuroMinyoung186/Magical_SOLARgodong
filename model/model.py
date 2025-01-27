import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BartForConditionalGeneration, AutoModelForCasualLM
from sentence_transformers import SentenceTransformer, util

def get_summarization_model(model_name = "alaggung/bart-r3f"):
    model = BartForConditionalGeneration.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

def get_llm_model(model_name = 'beomi/OPEN-SOLAR-KO-10.7B', device='auto'):

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        load_in_4bit = True,
        torch_dtype = torch.float16,
        device_map = device
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

def get_sentence_transformer(model_name = 'snunlp/KR-SBERT-V40K-klueNLI-augSTS'):
    model = SentenceTransformer(model_name)
    return model

