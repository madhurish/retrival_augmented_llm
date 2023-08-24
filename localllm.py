from langchain.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import gradio as gr

def main():
    print("Entered main")
    path="/Volumes/T7/llama-2-7b-chat.ggmlv3.q4_0.bin"

    n_gpu_layers = 1  # Metal set to 1 is enough.
    n_batch = 1024  # Should be between 1 and n_ctx, consider the amount of RAM of your Apple Silicon Chip.
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

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

    # Gradio logic here
    
    prompt = """
    Question: how tall is the eiffel tower?
    """
    print("here")
    llm(prompt)
    

if __name__=="__main__":
    print("Hello")
    main()