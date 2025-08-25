from semantic_kernel import Kernel
import asyncio
import torch
from transformers import AutoTokenizer

# 会話履歴関連の設定
from semantic_kernel.memory import (
    SemanticTextMemory,
    VolatileMemoryStore,
)
from semantic_kernel.core_plugins import TextMemoryPlugin
from semantic_kernel.prompt_template import PromptTemplateConfig
from semantic_kernel.prompt_template.input_variable import InputVariable
from semantic_kernel.contents import ChatHistory
from semantic_kernel.functions import KernelArguments

# Huggingface上で提供されているモデルを動かすための設定
from semantic_kernel.connectors.ai.hugging_face import (
    HuggingFaceTextCompletion,
    HuggingFaceTextEmbedding,
    HuggingFacePromptExecutionSettings,
)

kernel = Kernel()  # create a Kernel object
text_service_id = "google/gemma-3-270m-it"  # specify the LLM to use for text generation

# padとeosのトークンを定義するためにトークナイザーを設定
tokenizer = AutoTokenizer.from_pretrained(
    text_service_id,
    use_fast=True,
)

# Define model init arguments
model_kwargs = dict(
    attn_implementation="flash_attention_2",  # Use "flash_attention_2" when running on Ampere or newer GPU
    torch_dtype=torch.bfloat16,  # What torch dtype to use, defaults to auto
    # device_map="cuda:0",  # Let torch decide how to load the model
)


# Let us add this LLM to our kernel object
# Since we are using Hugging Face model, we have imported and will use HuggingFaceTextCompletion class
# Below we have added text generation model to our kernel
kernel.add_service(
    service=HuggingFaceTextCompletion(
        service_id=text_service_id,
        ai_model_id=text_service_id,
        task="text-generation",
        device=0,  # GPUの場合は0以上（0で`cuda:0`指定になる），CPUの場合は-1
        model_kwargs=model_kwargs,
    ),
)

# execution settings for AI model
execution_settings = HuggingFacePromptExecutionSettings(
    service_id=text_service_id,
    ai_model_id=text_service_id,
    max_tokens=200,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.pad_token_id,
    max_new_tokens=100,
)

jinja2_prompt = """ChatBot can have a conversation with you about any topic.
It can give explicit instructions or say 'I don't know' if it does not have an answer.

{% for message in history %}
{{ message.role }}: {{ message.content }}
{% endfor %}

User: {{ user_input }}

ChatBot: """

prompt_template_config = PromptTemplateConfig(
    template=jinja2_prompt,
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
        ),
    ],
    execution_settings=execution_settings,
)

chat_function = kernel.add_function(
    function_name="chat",
    plugin_name="chatPlugin",
    prompt_template_config=prompt_template_config,
)

chat_history = ChatHistory()
chat_history.add_system_message("You are a helpful chatbot.")


# Convert ChatHistory to dictionary format for Jinja2 compatibility
def convert_chat_history_to_dict(chat_history):
    """Convert ChatHistory object to list of dictionaries for Jinja2 template"""
    history_list = []
    for message in chat_history:
        # Extract role and content from ChatHistory message
        role = (
            message.role.value if hasattr(message.role, "value") else str(message.role)
        )
        content = str(message.content)
        history_list.append({"role": role, "content": content})
    return history_list


history_dict = convert_chat_history_to_dict(chat_history)

context = KernelArguments(
    user_input="Tell me about wild animals",
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
    # asyncio.run(main())
    await main()
