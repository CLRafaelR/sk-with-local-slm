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
    attn_implementation="eager",  # Use "flash_attention_2" when running on Ampere or newer GPU
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
# Next we have added an embedding model from HF to our kernel
embed_service_id = "sentence-transformers/all-MiniLM-L6-v2"
embedding_svc = HuggingFaceTextEmbedding(
    service_id=embed_service_id,
    ai_model_id=embed_service_id,
)
kernel.add_service(
    service=embedding_svc,
)
# Next we are adding volatile memory plugin to our kernel
memory = SemanticTextMemory(
    storage=VolatileMemoryStore(),
    embeddings_generator=embedding_svc,
)
kernel.add_plugin(TextMemoryPlugin(memory), "TextMemoryPlugin")


async def setup_memory():
    # let us create a collection to store 5 pieces of information in memory plugin
    # this is infomration about 5 animals
    collection_id = "generic"
    await memory.save_information(
        collection=collection_id,
        id="info1",
        text="Sharks are fish.",
    )
    await memory.save_information(
        collection=collection_id,
        id="info2",
        text="Whales are mammals.",
    )
    await memory.save_information(
        collection=collection_id,
        id="info3",
        text="Penguins are birds.",
    )
    await memory.save_information(
        collection=collection_id,
        id="info4",
        text="Dolphins are mammals.",
    )
    await memory.save_information(
        collection=collection_id,
        id="info5",
        text="Flies are insects.",
    )
    return collection_id


# Define prompt function using SK prompt template language
my_prompt = """I know these animal facts:
- {{recall 'fact about sharks'}}
- {{recall 'fact about whales'}}
- {{recall 'fact about penguins'}}
- {{recall 'fact about dolphins'}}
- {{recall 'fact about flies'}}
Now, tell me something about: {{$request}}"""

# execution settings for AI model
execution_settings = HuggingFacePromptExecutionSettings(
    service_id=text_service_id,
    ai_model_id=text_service_id,
    max_tokens=200,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.pad_token_id,
    max_new_tokens=100,
)

# prompt template configurations
prompt_template_config = PromptTemplateConfig(
    template=my_prompt,
    name="text_complete",
    template_format="semantic-kernel",
    execution_settings=execution_settings,
)
# let the semantic function to the kernel
# this function uses above prompt and model
my_function = kernel.add_function(
    function_name="text_complete",
    plugin_name="TextCompletionPlugin",
    prompt_template_config=prompt_template_config,
)


async def run_memory_demo():
    collection_id = await setup_memory()

    output = await kernel.invoke(
        my_function,
        request="What are whales?",
    )

    output = str(output).strip()
    query_result1 = await memory.search(
        collection=collection_id,
        query="What are sharks?",
        limit=1,
        min_relevance_score=0.3,
    )
    print(f"The queried result for 'What are sharks?' is {query_result1[0].text}")
    print(f"{text_service_id} completed prompt with: '{output}'")


prompt = """ChatBot can have a conversation with you about any topic.
It can give explicit instructions or say 'I don't know' if it does not have an answer.

{{$history}}

User: {{$user_input}}

ChatBot: """

prompt_template_config = PromptTemplateConfig(
    template=prompt,
    name="chat",
    template_format="semantic-kernel",
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

context = KernelArguments(
    user_input="Tell me about wild animals",
    history=chat_history,
    # history=history_string,
)


async def main():
    # Run memory demo first
    # await run_memory_demo()

    # Then run chat
    response = await kernel.invoke(
        function=chat_function,
        arguments=context,
    )
    print(response)


# Run the main function
if __name__ == "__main__":
    # asyncio.run(main())
    await main()
