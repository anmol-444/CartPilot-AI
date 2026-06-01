from langchain_google_genai import ChatGoogleGenerativeAI


class LlmClient:
    """Google Gemini client — wraps ChatGoogleGenerativeAI creation.

    Created once in di.py and injected into LLM components.
    """

    def __init__(self, api_key: str, model: str = "gemini-2.0-flash") -> None:
        self.chat_openai = ChatGoogleGenerativeAI(
            model=model,
            api_key=api_key,
            temperature=0.3,
        )
