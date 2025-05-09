from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatResult, ChatGeneration
import requests
import os

class OpenAIChatCustom(BaseChatModel):
    def __init__(self, deployment_name, api_key, endpoint, api_version="2023-05-15"):
        self.deployment_name = deployment_name
        self.api_key = api_key
        self.endpoint = endpoint
        self.api_version = api_version

    def _convert_messages(self, messages):
        converted = []
        for m in messages:
            if isinstance(m, HumanMessage):
                role = "user"
            elif isinstance(m, AIMessage):
                role = "assistant"
            elif isinstance(m, SystemMessage):
                role = "system"
            else:
                continue
            converted.append({"role": role, "content": m.content})
        return converted

    def _call(self, messages, stop=None):
        url = f"{self.endpoint}/openai/deployments/{self.deployment_name}/chat/completions?api-version={self.api_version}"
        headers = {
            "Content-Type": "application/json",
            "api-key": self.api_key
        }
        data = {
            "messages": self._convert_messages(messages),
            "temperature": 0.7,
            "max_tokens": 500
        }
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        result_text = response.json()["choices"][0]["message"]["content"]
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=result_text))])

    @property
    def _llm_type(self) -> str:
        return "openai-custom"
