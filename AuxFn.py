import os
import requests
import fitz
from tqdm.auto import tqdm
from spacy.lang.en import English
import re
from sentence_transformers import SentenceTransformer
import pandas as pd
import random
import numpy as np
import torch
from sentence_transformers import util
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.utils import is_flash_attn_2_available
from huggingface_hub import login


def install_link(url, name): # url is the url of the pdf and name is the name link you want to sace 
    
    pdf_path = name
    if not os.path.exists(pdf_path):
        os.makedirs(pdf_path)
        url = url
        r = requests.get(url, stream=True)

        if r.status_code == 200:
            with open(os.path.join(pdf_path, f"{name}.pdf"), "wb") as f:
                for chunk in r:
                    f.write(chunk)
            print("PDF downloaded successfully!")
        else:
            print("Failed to download PDF.")
    else:
        print("PDF already exists.")

def text_formatter(text: str) -> str:

  cleaned_text = text.replace("\n", " ").strip()
  return cleaned_text

def open_and_read_pdf(pdf_path: str)-> list[dict]:

  doc = fitz.open(pdf_path)
  pages_and_texts = []
  for page_number, page in tqdm(enumerate(doc)):
    page_text = page.get_text()
    cleaned_text = text_formatter(page_text)
    pages_and_texts.append({"page_number": page_number, "text": cleaned_text})

  return pages_and_texts

def chunking_text(pdf_path):
    pages_and_texts = open_and_read_pdf(pdf_path)
    nlp = English()
    nlp.add_pipe('sentencizer')

    max_len_size = 0

    for item in tqdm(pages_and_texts):
        doc = nlp(item["text"])
        sents = [text_formatter(sent.text) for sent in doc.sents]
        item["sents"] = sents
        max_len_size = max(max_len_size, len(sents))
    
    for item in tqdm(pages_and_texts):
        sents = item["sents"]
        sentence_groups = []
        chunks = []
        count = 0
        n = len(sents)

        for i in range(n):

            chunks.extend(sents[i].split())
            count += len(sents[i].split())
            if count >= 150:
                sentence_groups.append(" ".join(chunks))
                count = 0
                chunks = []

        item["sentence_chunks"] = sentence_groups

# Now store these sentence chunks as dictionaries of sentences and their attributes

    pages_and_chunks = []

    for item in tqdm(pages_and_texts):

        sentence_groups = item["sentence_chunks"]
        for sentence_chunk in sentence_groups:
            chunk_dict = {}
            chunk_dict["page_number"] = item["page_number"]
            chunk_dict["sentence_chunk"] = sentence_chunk
            chunk_dict["chunk_char_count"] = len(sentence_chunk)
            chunk_dict["chunk_word_count"] = len([word for word in sentence_chunk.split(" ")])
            chunk_dict["chunk_token_count"] = len(sentence_chunk) / 4 # 1 token = ~4 characters
            pages_and_chunks.append(chunk_dict)

    return pages_and_chunks
   
def convert_embedding(pages_and_chunks):
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embedding_model = SentenceTransformer(model_name_or_path="sentence-transformers/all-MiniLM-L6-v2", device = "cpu")
    embedding_model.to(device)
    for chunk in tqdm(pages_and_chunks):
        chunk["embedding"] = embedding_model.encode(chunk["sentence_chunk"])
    
    return pages_and_chunks

def store_embeddings(df_path, pages_and_chunks):

    
    text_and_embeddings_df = pd.DataFrame(pages_and_chunks)
    path = df_path
    text_and_embeddings_df.to_csv(path)

def Query_and_search_results(query, records, embeddings):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    embedding_model = SentenceTransformer(model_name_or_path="sentence-transformers/all-MiniLM-L6-v2", device = "cpu")
    embedding_model.to(device)
    
    ## Some extra loop for preprocessing - extract the 5 main topics necessary to obtain the output
    query_embedding = embedding_model.encode(query, convert_to_tensor=True)
    cos_scores = util.cos_sim(query_embedding, embeddings)
    # Alternatively could use dot product similarity
    top_results_cos_product = torch.topk(cos_scores, k=10)
    return top_results_cos_product

def llm_model_loading(model_id="google/gemma-7b-it" ):
    login("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") # huggingface tokem id
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_id)
    llm = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=model_id, 
                                                torch_dtype=torch.float16)
    return tokenizer, llm

def generating_response(query, records, tokenizer, llm):
    

    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    prompt = f"Analyze the problem: '{query}'. Identify 5 key topics that must be addressed to solve this problem. Return the topics as a structured dictionary in the format {{1: 'Topic 1', 2: 'Topic 2', ..., 5: 'Topic 5'}}."
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = llm.generate(inputs.input_ids, max_length=150, num_return_sequences=1, temperature=0.7)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    try:
        # Evaluate the dictionary-like response
        topics_dict = eval(response)
        print("Extracted Topics Dictionary:\n", topics_dict)
    except Exception as e:
        print("Error parsing the response. Make sure the model's output is formatted as a dictionary.")

    response_top_k = []
    text_and_embedding_df = pd.read_csv(records)
    text_and_embedding_df["embedding"] = text_and_embedding_df["embedding"].apply(lambda x: np.fromstring(x.strip("[]"), sep=" "))
    text_and_embedding_dict = text_and_embedding_df.to_dict(orient="records")
    embeddings = torch.tensor(np.array(text_and_embedding_df["embedding"].tolist()), dtype=torch.float32).to(device)

    for _, q in topics_dict:
        r =  Query_and_search_results(q, records,embeddings)
        response_top_k.append[r]
    
    return response_top_k, text_and_embedding_dict