import os
from langchain.llms import LlamaCpp, HuggingFacePipeline
from langchain import PromptTemplate, LLMChain
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import gradio as gr
from torch import cuda, bfloat16
import transformers
import sys

# Global Variables
MODEL_ID = 'meta-llama/Llama-2-7b-chat-hf'
DEVICE = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'
BNB_CONFIG = transformers.BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=bfloat16
)


def install_dependencies():
    pass  # Placeholder for the installation of dependencies


def load_hf_model_and_tokenizer():
    hf_auth = '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'
    model_config = transformers.AutoConfig.from_pretrained(
        MODEL_ID,
        cache_dir='/workspace',
        use_auth_token=hf_auth
    )

    model = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        config=model_config,
        quantization_config=BNB_CONFIG,
        device_map='auto',
        cache_dir='/workspace',
        use_auth_token=hf_auth
    )
    model.eval()
    print(f"Model loaded on {DEVICE}")

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        MODEL_ID,
        cache_dir='/workspace',
        use_auth_token=hf_auth
    )
    return model, tokenizer


def generate_text(model, tokenizer):
    return transformers.pipeline(
        model=model, tokenizer=tokenizer,
        return_full_text=True,
        task='text-generation',
        temperature=0.1,
        max_new_tokens=512,
        repetition_penalty=1.1
    )


def load_from_directory(directory_path):
    pdf_files = [f for f in os.listdir(directory_path) if f.endswith('.pdf')]
    docs = []
    for pdf_file in pdf_files:
        try:
            loader = UnstructuredPDFLoader(os.path.join(directory_path, pdf_file))
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
            documents = loader.load()
            splitted_docs = text_splitter.split_documents(documents)
            docs.extend(splitted_docs)
        except:
            print("Skipping: ", os.path.join(directory_path, pdf_file))
    print("Loaded and split all documents")
    return docs


def save_db(docs):
    hf_embedding = HuggingFaceInstructEmbeddings()
    db = FAISS.from_documents(docs, hf_embedding)
    db.save_local("vector_db")
    db = FAISS.load_local("vector_db/", embeddings=hf_embedding)
    return db


def get_search(db, query):
    search = db.similarity_search(query, k=2)
    return search


def main():
    install_dependencies()
    model, tokenizer = load_hf_model_and_tokenizer()
    llm = generate_text(model, tokenizer)
    
    docs = load_from_directory('./docs')
    db = save_db(docs)
    
    query = "what is waymo dataset?"  # This can be modified to use sys.argv[1] for CLI
    search = get_search(db, query)
    
    template = '''Context: {context}
Based on Context provide me answer for following question
Question: {question}
Tell me the information about the fact. The answer should be from context only
do not use general knowledge to answer the query'''
    
    prompt = PromptTemplate(input_variables=["context", "question"], template=template)
    final_prompt = prompt.format(question=query, context=search)
    response = llm(final_prompt)
    print("Response:", response)


if __name__ == '__main__':
    main()
