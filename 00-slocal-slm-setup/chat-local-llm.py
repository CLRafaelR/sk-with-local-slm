import asyncio

from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.function_choice_behavior import (
    FunctionChoiceBehavior,
)
from semantic_kernel.connectors.ai.open_ai import AzureChatPromptExecutionSettings
from semantic_kernel.contents import ChatHistory
from semantic_kernel.functions import KernelArguments

from semantic_kernel.connectors.ai.hugging_face import HuggingFaceTextCompletion

from openai import AsyncOpenAI

from transformers import (
    BitsAndBytesConfig,
)

import torch


openAIClient: AsyncOpenAI = AsyncOpenAI(
    api_key="fake-key",  # This cannot be an empty string, use a fake key
    base_url="http://localhost:1234/v1",
)

MODEL_ID = "google/gemma-3-270m-it"

# Define model init arguments
model_kwargs = dict(
    attn_implementation="eager",  # Use "flash_attention_2" when running on Ampere or newer GPU
    torch_dtype=torch.bfloat16,  # What torch dtype to use, defaults to auto
    # device_map="auto",  # Let torch decide how to load the model
)

# BitsAndBytesConfig: Enables 4-bit quantization to reduce model size/memory usage
# model_kwargs["quantization_config"] = BitsAndBytesConfig(
# load_in_4bit=True,
# bnb_4bit_use_double_quant=True,
# bnb_4bit_quant_type="nf4",
# bnb_4bit_compute_dtype=model_kwargs["torch_dtype"],
# bnb_4bit_quant_storage=model_kwargs["torch_dtype"],
# )

pipeline_kwargs = {
    "max_new_tokens": 300,
    "framework": "pt",
}

service_id = "my-service-id"


async def main() -> None:
    kernel = Kernel()
    kernel.add_service(
        service=HuggingFaceTextCompletion(
            ai_model_id=MODEL_ID,
            # device=None,
            # device_map="cuda",
            service_id=service_id,
            # task="text2text-generation",
            model_kwargs=model_kwargs,
            # pipeline_kwargs=pipeline_kwargs,
        )
    )

    # kernel.add_service(
    #     service=AzureChatCompletion(
    #         service_id=service_id,
    #         api_key=app_settings.OPENAI_COMPLETION_API_KEY,
    #         deployment_name=app_settings.OPENAI_COMPLETION_DEPLOYMENT_NAME,
    #         endpoint=app_settings.OPENAI_COMPLETION_ENDPOINT,
    #     )
    # )

    execution_config = kernel.get_service(
        service_id
    ).instantiate_prompt_execution_settings(
        service_id=service_id,
        max_tokens=100,
        temperature=0,
        seed=42,
    )

    if isinstance(execution_config, AzureChatPromptExecutionSettings):
        execution_config.function_choice_behavior = FunctionChoiceBehavior.Auto(
            auto_invoke=True
        )

    service = kernel.get_service(service_id=service_id)

    if not isinstance(service, HuggingFaceTextCompletion):
        raise Exception("Invalid Value")

    history = ChatHistory()
    history.add_user_message("hello")

    hf_complete = kernel.create_function_from_prompt(
        prompt=prompt,
        plugin_name="Generate_Capital_City_Completion",
        function_name="generate_city_completion_opt",
        execution_settings=execution_config,
    )

    response = await kernel.invoke(hf_complete, input="Paris")

    settings = kernel.get_prompt_execution_settings_from_service_id(
        service_id=service_id
    )

    result = await service.get_chat_message_contents(
        chat_history=history,
        settings=settings,
        kernel=kernel,
        arguments=KernelArguments(settings=settings),
    )

    if not result:
        raise Exception("result is None")

    print(result[0].content)


if __name__ == "__main__":
    # asyncio.run(main())
    await main()
