import gradio as gr
import os
import spaces
from transformers import GemmaTokenizer, AutoModelForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from threading import Thread

# Set an environment variable
HF_TOKEN = os.environ.get("HF_TOKEN", None)

# Custom HTML for the header and footer
DESCRIPTION = '''
<div style="text-align: center;">
    <h1 style="font-size: 32px; font-weight: bold; color: #1565c0;">DeepSeek-R1-Distill-Qwen-32B-bnb-4bit</h1>
    <p style="font-size: 16px; color: #555;">Developed by <a href="https://ruslanmv.com/" target="_blank" style="color: #1565c0; text-decoration: none;">RuslanMV</a></p>
</div>
'''

FOOTER = '''
<div style="text-align: center; margin-top: 20px; padding: 10px; background-color: #f5f5f5; border-radius: 8px;">
    <p style="font-size: 14px; color: #777;">Powered by Gradio and Hugging Face Transformers</p>
</div>
'''

PLACEHOLDER = '''
<div style="padding: 30px; text-align: center; display: flex; flex-direction: column; align-items: center;">
    <h1 style="font-size: 28px; margin-bottom: 2px; opacity: 0.55;">DeepSeek-R1-Distill-Qwen-32B-bnb-4bit</h1>
    <p style="font-size: 18px; margin-bottom: 2px; opacity: 0.65;">Ask me anything...</p>
</div>
'''

# Custom CSS for better styling
css = """
h1 {
    text-align: center;
    display: block;
    font-weight: bold;
    color: #1565c0;
}
#duplicate-button {
    margin: auto;
    color: white;
    background: #1565c0;
    border-radius: 100vh;
}
.chatbot {
    border-radius: 8px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}
.accordion {
    background-color: #f5f5f5;
    border-radius: 8px;
    padding: 10px;
}
"""

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("unsloth/DeepSeek-R1-Distill-Qwen-32B-bnb-4bit")
tokenizer.chat_template = "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% set ns = namespace(is_first=false, is_tool=false, is_output_first=true, system_prompt='') %}{%- for message in messages %}{%- if message['role'] == 'system' %}{% set ns.system_prompt = message['content'] %}{%- endif %}{%- endfor %}{{bos_token}}{{ns.system_prompt}}{%- for message in messages %}{%- if message['role'] == 'user' %}{%- set ns.is_tool = false -%}{{'<|user|>' + message['content']}}{%- endif %}{%- if message['role'] == 'assistant' and message['content'] is none %}{%- set ns.is_tool = false -%}{%- for tool in message['tool_calls']%}{%- if not ns.is_first %}{{'<|assistant|>' + tool['type'] + ':' + tool['function']['name'] + '\\n' + '```json' + '\\n' + tool['function']['arguments'] + '\\n' + '```' + '}}\\n'}}{%- set ns.is_first = true -%}{%- else %}{{'\\n' + '<|assistant|>' + tool['type'] + ':' + tool['function']['name'] + '\\n' + '```json' + '\\n' + tool['function']['arguments'] + '\\n' + '```' + '}}\\n'}}{{'}}\\n'}}{%- endif %}{%- endfor %}{%- endif %}{%- if message['role'] == 'assistant' and message['content'] is not none %}{%- if ns.is_tool %}{{'<|assistant|>' + message['content'] + '}}\\n'}}{%- set ns.is_tool = false -%}{%- else %}{% set content = message['content'] %}{% if '</think>' in content %}{% set content = content.split('</think>')[-1] %}{% endif %}{{'<|assistant|>' + content + '}}\\n'}}{%- endif %}{%- endif %}{%- if message['role'] == 'tool' %}{%- set ns.is_tool = true -%}{%- if ns.is_output_first %}{{'<|tool|>' + message['content'] + '}}\\n'}}{%- set ns.is_output_first = false %}{%- else %}{{'\\n<|tool|>' + message['content'] + '}}\\n'}}{%- endif %}{%- endif %}{%- endfor -%}{% if ns.is_tool %}{{'<|assistant|>'}}{% endif %}{% if add_generation_prompt and not ns.is_tool %}{{'<|assistant|>'}}{% endif %}"

model = AutoModelForCausalLM.from_pretrained("unsloth/DeepSeek-R1-Distill-Qwen-32B-bnb-4bit", device_map="auto")
terminators = [
    tokenizer.eos_token_id,
]

@spaces.GPU(duration=120)
def chat_llama3_8b(message: str, history: list, temperature: float, max_new_tokens: int) -> str:
    """
    Generate a streaming response using the llama3-8b model.
    Args:
        message (str): The input message.
        history (list): The conversation history used by ChatInterface.
        temperature (float): The temperature for generating the response.
        max_new_tokens (int): The maximum number of new tokens to generate.
    Returns:
        str: The generated response.
    """
    conversation = []
    for user, assistant in history:
        conversation.extend([{"role": "user", "content": user}, {"role": "assistant", "content": assistant}])
    conversation.append({"role": "user", "content": message})

    input_ids = tokenizer.apply_chat_template(conversation, return_tensors="pt", add_generation_prompt=True).to(model.device)
    streamer = TextIteratorStreamer(tokenizer, timeout=10.0, skip_prompt=True, skip_special_tokens=True)

    generate_kwargs = dict(
        input_ids=input_ids,
        streamer=streamer,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        eos_token_id=terminators,
    )
    if temperature == 0:
        generate_kwargs['do_sample'] = False

    t = Thread(target=model.generate, kwargs=generate_kwargs)
    t.start()

    outputs = []
    for text in streamer:
        if "<think>" in text:
            text = text.replace("<think>", "[think]").strip()
        if "</think>" in text:
            text = text.replace("</think>", "[/think]").strip()
        outputs.append(text)
        yield "".join(outputs)

# Gradio block
chatbot = gr.Chatbot(height=450, placeholder=PLACEHOLDER, label='Chat with DeepSeek-R1')

with gr.Blocks(fill_height=True, css=css) as demo:
    gr.Markdown(DESCRIPTION)
    gr.ChatInterface(
        fn=chat_llama3_8b,
        chatbot=chatbot,
        fill_height=True,
        additional_inputs_accordion=gr.Accordion(label="⚙️ Parameters", open=False, render=False),
        additional_inputs=[
            gr.Slider(minimum=0, maximum=1, step=0.1, value=0.5, label="Temperature", render=False),
            gr.Slider(minimum=128, maximum=4096, step=1, value=1024, label="Max new tokens", render=False),
        ],

        
        examples=[
            ['Write a short poem about a lonely robot finding a friend.'],
            ['Explain quantum mechanics as if I’m a beginner in high school physics.'],
            ['If you have three apples and cut each into four pieces, how many pieces do you have?'],
            ['Make up a funny conversation between a cat and a goldfish.'],
            ['Convince me that dragons could exist in some form.'],
            ['What is the square root of 3,456 rounded to two decimal places?'],
            ['If humans had three arms, how would it change sports like basketball?'],
            ['Do you think artificial intelligence can ever truly be creative? Why or why not?'],
            ['Imagine a futuristic city powered entirely by renewable energy. What would it look like?'],
            ['Write a sentence where every word starts with the letter "S".'],
            ['Describe a traditional dish from Japan and how it is made.'],
            ['Is it ethical to use cloning to bring back extinct species? Why or why not?'],
            ['Write a Python function to reverse a string.'],
            ['Give me a motivational speech for finishing a challenging project.'],
            ['If dogs ruled the world, what laws would they make?']
        ]
        
        
        ,
        cache_examples=False,
    )
    gr.Markdown(FOOTER)

if __name__ == "__main__":
    demo.launch()