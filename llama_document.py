from langchain.llms import LlamaCpp
from langchain import PromptTemplate, LLMChain
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import gradio as gr

from langchain.llms import LlamaCpp
from langchain import PromptTemplate, LLMChain
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import (
    StreamingStdOutCallbackHandler
)
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


print("Entered main")
path="/Volumes/T7/llama-2-7b-chat.ggmlv3.q4_0.bin"
# path="/Volumes/T7/LLaMA-2-7B-32K.ggmlv3.q4_0.bin"

# path="/Volumes/T7/rp-chat-3b-v1-ggml-model-q4_0.bin"

template = """Question: {question}

Answer:"""

prompt = PromptTemplate(template=template, input_variables=["question"])
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

n_gpu_layers = 1  # Metal set to 1 is enough.
n_batch = 1024  # Should be between 1 and n_ctx, consider the amount of RAM of your Apple Silicon Chip.

# Make sure the model path is correct for your system!
llm = LlamaCpp(
    model_path=path,
    n_gpu_layers=n_gpu_layers,
    n_batch=n_batch,
    n_ctx=2048,
    f16_kv=True,  # MUST set to True, otherwise you will run into problem after a couple of calls
    callback_manager=callback_manager,
    verbose=True,
)



print("Loaded model in memory")

llm_chain = LLMChain(prompt=prompt, llm=llm, verbose=True)

# loader = UnstructuredFileLoader("AI discovers over 20K taxable French swimming pools.docx")
loader = UnstructuredPDFLoader("waymo.pdf")
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
docs = text_splitter.split_documents(documents)
print("splitted docs")
# embedding engine
hf_embedding = HuggingFaceInstructEmbeddings()

db = FAISS.from_documents(docs, hf_embedding)

# save embeddings in local directory
db.save_local("faiss_AiArticle")

# load from local
db = FAISS.load_local("faiss_AiArticle/", embeddings=hf_embedding)

query = "what is waymo dataset?"
search = db.similarity_search(query, k=2)
print(search)
template = '''Context: {context}

Based on Context provide me answer for following question
Question: {question}

Tell me the information about the fact. The answer should be from context only
do not use general knowledge to answer the query'''

prompt = PromptTemplate(input_variables=["context", "question"], template= template)
final_prompt = prompt.format(question=query, context=search)

# response = llm(final_prompt)
response = llm_chain(final_prompt)
print("Response:", response)