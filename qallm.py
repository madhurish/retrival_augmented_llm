# -*- coding: utf-8 -*-
import os
import sys
from langchain.llms import LlamaCpp, HuggingFacePipeline
from langchain import PromptTemplate, LLMChain
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.document_loaders import UnstructuredPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import gradio as gr
from torch import cuda, bfloat16
import transformers


class LLM_PDF_QA:
    def __init__(self):
        self.install_dependencies()
        self.load_dependencies()
        self.model, self.tokenizer = self.load_hf_model_and_tokenizer()
        self.generate_text=transformers.pipeline(
            model=self.model, tokenizer=self.tokenizer,
            return_full_text=True,  # langchain expects the full text
            task='text-generation',
            temperature=0.1,
            max_new_tokens=512,
            repetition_penalty=1.1
        )
        self.llm = HuggingFacePipeline(pipeline=self.generate_text)
        docs = self.load_from_directory('./docs')
        self.db = self.save_db(docs)

    @staticmethod
    def install_dependencies():
        # Install required packages
        # Note: In the given environment, the `!pip install` commands will not work
        # They are kept for reference
        pass

    def load_dependencies(self):
        self.model_id = 'meta-llama/Llama-2-13b-chat-hf'
        self.device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'
        self.bnb_config = transformers.BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=bfloat16
        )

    def load_hf_model_and_tokenizer(self):
        hf_auth = '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'
        model_config = transformers.AutoConfig.from_pretrained(
            self.model_id,
            cache_dir='/workspace',
            use_auth_token=hf_auth
        )

        model = transformers.AutoModelForCausalLM.from_pretrained(
            self.model_id,
            trust_remote_code=True,
            config=model_config,
            quantization_config=self.bnb_config,
            device_map='auto',
            cache_dir='/workspace',
            use_auth_token=hf_auth
        )
        model.eval()
        print(f"Model loaded on {self.device}")

        tokenizer = transformers.AutoTokenizer.from_pretrained(
            self.model_id,
            cache_dir='/workspace',
            use_auth_token=hf_auth
        )
        return model, tokenizer

    

    @staticmethod
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
        txt_files = [f for f in os.listdir(directory_path) if f.endswith('.txt')]
        for text_file in txt_files:
            try:
                loader = TextLoader(os.path.join(directory_path, text_file))
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
                documents = loader.load()
                splitted_docs = text_splitter.split_documents(documents)
                docs.extend(splitted_docs)
            except:
                print("Skipping: ", os.path.join(directory_path, text_file))
        print("Loaded and split all documents")
        return docs
    @staticmethod
    def delete_files_in_directory(directory_path):
        try:
            files = os.listdir(directory_path)
            for file in files:
                file_path = os.path.join(directory_path, file)
                if os.path.isfile(file_path):
                    os.remove(file_path)
            print("All files deleted successfully.")
        except OSError:
            print("Error occurred while deleting files.")
    @staticmethod
    def delete_file_in_directory(directory_path,file_name):
        file_path = os.path.join(directory_path, file_name)
        if os.path.isfile(file_path):
            os.remove(file_path)
            return 200
        else:
            return 400
    
    @staticmethod
    def concatenate_files_from_directory(directory_path):
        """Reads all files from the given directory and concatenates their content into a single string."""
        concatenated_string = ""
        
        # List all files in the directory
        files = [f for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f))]
        
        # Loop through each file and read its content
        for file in files:
            with open(os.path.join(directory_path, file), 'r') as f:
                concatenated_string += f.read()
        
        return concatenated_string

        

    def save_db(self,docs):
        hf_embedding = HuggingFaceInstructEmbeddings()
        self.db = FAISS.from_documents(docs, hf_embedding)
        self.db.save_local("vector_db")
        self.db = FAISS.load_local("vector_db/", embeddings=hf_embedding)
        return self.db

    def get_search(self, db, query):
        search = db.similarity_search(query, k=2)
        return search
    
    def answer(self,query):
        ans=self.llm(prompt=query)
        return ans
    

    def run(self, query):
        template = """Question: {question}

Answer:"""

        qa_prompt = PromptTemplate(template=template, input_variables=["question"])
        llm_chain = LLMChain(prompt=qa_prompt, llm=self.llm, verbose=True)
        search = self.get_search(self.db, query)
        template = '''Context: {context}

Based on Context provide me answer for following question
Question: {question}

Tell me the information about the fact. The answer should be from context only
do not use general knowledge to answer the query. If context is irrelevant inform that context is not relevant'''
        prompt = PromptTemplate(input_variables=["context", "question"], template=template)
        final_prompt = prompt.format(question=query, context=search)
        response = llm_chain(final_prompt)
        print(response["text"])
        return response["text"]
    
    def classify(self, query):
        template = """Question: {question}

Answer:"""

        qa_prompt = PromptTemplate(template=template, input_variables=["question"])
        llm_chain = LLMChain(prompt=qa_prompt, llm=self.llm, verbose=True)
        template = '''Message: {context}

Based on Message provided categorize it to one of the following: "URGENT", "IMPORTANT", "NOT IMPORTANT", "SPAM"

clssify the message as one of the 4 given categories and only respond with one word answer.'''
        prompt = PromptTemplate(input_variables=["context"], template=template)
        final_prompt = prompt.format(context=query)
        response = llm_chain(final_prompt)
        print(response["text"])
        return response["text"]
    def answer(self, query):
        template = """Question: {question}

Answer:"""

        qa_prompt = PromptTemplate(template=template, input_variables=["question"])
        llm_chain = LLMChain(prompt=qa_prompt, llm=self.llm, verbose=True)
        search = self.concatenate_files_from_directory('./chats')
        template = '''Context: {context}

Based on Context provide me answer for following question
Question: {question}
use messages and metadata to answer the question, use minimum words to answer the question
'''
        prompt = PromptTemplate(input_variables=["context", "question"], template=template)
        final_prompt = prompt.format(question=query, context=search)
        if search!="":
            response = llm_chain(final_prompt)
            print(response["text"])
            return response["text"]
        return "Upload messages to infer"

if __name__ == '__main__':
    
    if len(sys.argv) < 2:
        print("Usage: python script_name.py <query>")
        sys.exit(1)

    query = sys.argv[1]
    obj = LLM_PDF_QA()
    obj.run(query)
