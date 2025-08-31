from semantic_kernel import (
    Kernel,
    __version__,
)
import asyncio
import torch
from transformers import (
    AutoTokenizer,
    logging,
)

# Configuration for conversational history management
from semantic_kernel.prompt_template import PromptTemplateConfig
from semantic_kernel.prompt_template.input_variable import InputVariable
from semantic_kernel.contents import ChatHistory
from semantic_kernel.functions import KernelArguments

import sys

logging.set_verbosity_info()

torch.set_float32_matmul_precision("high")

# Configuration settings for executing models distributed via the Hugging Face platform
from semantic_kernel.connectors.ai.hugging_face import (
    HuggingFaceTextCompletion,
    HuggingFacePromptExecutionSettings,
)

print(f"The version of semantic-kernel: {__version__}")


"""
Kernel instantiation and initialisation
"""
kernel = Kernel()

"""
Model configuration and parameter specification
"""
# Designation of the text generation model architecture
text_service_id = "google/gemma-3-270m-it"

# Tokeniser configuration to establish padding and end-of-sequence token definitions
tokenizer = AutoTokenizer.from_pretrained(
    text_service_id,
    use_fast=True,
)

# Runtime configuration parameters for model initialisation
model_kwargs = dict(
    attn_implementation="flash_attention_2",
    torch_dtype="auto",
)

pipeline_kwargs = dict(
    # max_tokens=200, # Note: Specifying max_tokens within pipeline_kwargs results in kernel.invoke method failure
    do_sample=True,
    max_new_tokens=100,
    num_return_sequences=1,
    pad_token_id=tokenizer.pad_token_id,
    eos_token_id=tokenizer.eos_token_id,
    repetition_penalty=1.5,
    temperature=0.7,
    top_p=0.95,
    top_k=50,
)

hf_text_service = HuggingFaceTextCompletion(
    service_id=text_service_id,
    ai_model_id=text_service_id,
    task="text-generation",
    device=0,
    model_kwargs=model_kwargs,
    pipeline_kwargs=pipeline_kwargs,
)

kernel.add_service(
    service=hf_text_service,
)

execution_settings = HuggingFacePromptExecutionSettings(
    service_id=text_service_id,
    do_sample=True,  # サンプリングを有効にする（必須）
    temperature=0.7,  # 生成のランダム性を制御
    top_p=0.95,       # Nucleus sampling
    top_k=50,         # Top-K sampling
    max_tokens=100,   # 最大生成トークン数
    extension_data=pipeline_kwargs,
)

chat_prompt_template = """{% for message in history %}
{{ message.role }}: {{ message.content }}
{% endfor %}

User: {{ user_input }}

ChatBot: """

system_prompt = """You are an advanced conversational agent capable of engaging with diverse thematic discourse.
Please adhere to explicit instructions when provided. In instances where an optimal response cannot be formulated, kindly indicate "I do not possess sufficient information to provide a comprehensive answer."
"""

user_input = """Please provide comprehensive information regarding wildlife and natural fauna.
"""

prompt_template_config = PromptTemplateConfig(
    template=chat_prompt_template,
    name="chat",
    template_format="jinja2",
    input_variables=[
        InputVariable(
            name="user_input",
            description="The user input",
            is_required=True,
        ),
        InputVariable(
            name="history",
            description="The conversation history",
            is_required=True,
            # allow_dangerously_set_content is required when updating sk from 1.31.0 to 1.36.0
            allow_dangerously_set_content=True,
        ),
    ],
    execution_settings=execution_settings,
)

chat_function = kernel.add_function(
    function_name="chat",
    plugin_name="chatPlugin",
    prompt_template_config=prompt_template_config,
    # execution_settings=execution_settings,
)

chat_history = ChatHistory()
chat_history.add_system_message(system_prompt)

context = KernelArguments(
    user_input=user_input,
    history=chat_history,
)


async def main():
    response = await kernel.invoke(
        function=chat_function,
        arguments=context,
    )
    print(response)


# Run the main function
if __name__ == "__main__":
    if "ipykernel" in sys.modules:
        # For Jupyter Notebook and Interactive Window
        import nest_asyncio
        nest_asyncio.apply()
        asyncio.run(main())
    else:
        # For Python and ipython scripts
        asyncio.run(main())
