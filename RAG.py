from AuxFn import *


def main():

    url, name = "https://teaching.eng.cam.ac.uk/sites/teaching.eng.cam.ac.uk/files/Documents/Databooks/Maths%202017.pdf", "Math"
    install_link(url, name)
    pdf_path = name + '/' + name +'.pdf'
    df_path = name + "/" + "store.csv"
    if not os.path.exists(df_path):  
        pages_and_chunks = chunking_text(pdf_path)
        pages_and_chunks = convert_embedding(pages_and_chunks) # adds chunk embeddings to the dictionary   
    
        store_embeddings(df_path=df_path, pages_and_chunks=pages_and_chunks)

    tokenizer, llm = llm_model_loading(model_id="google/gemma-7b-it")
    Query = "Butterworth"
    response_top_k, text_and_embedding_dict = generating_response(Query, df_path, tokenizer, llm)

    information = []
    for top_k in response_top_k:
        top_k[0].cpu().np()
        a = list(zip(top_k[0].cpu().numpy()[0], top_k[1].cpu().numpy()[0]))
        for score, idx in a:
            information.append(text_and_embedding_dict[idx]["sentence_chunk"])
    
    Final_feed_to_transformer = Query

    for i in information:
        Final_feed_to_transformer += (i + "\n")
    
    output = llm.generate(Final_feed_to_transformer.input_ids, max_length=50, num_return_sequences=1)
    # Decode the output
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    print(response)

    return response

if __name__ == "__main__":
    main()








            









