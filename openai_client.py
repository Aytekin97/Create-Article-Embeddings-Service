from openai import OpenAI
from config import settings


class OpenAiClient:
    def __init__(self) -> None:
        self.client = OpenAI(api_key=settings.openai_api_key)
        self.model = settings.openai_model

    def query_gpt(self, messages, response_format):
        completion = self.client.beta.chat.completions.parse(
            model=self.model, messages=messages, response_format=response_format
        )
        return completion.choices[0].message.parsed
    
    def generate_embeddings(self, text_list):
        try:
            response = self.client.embeddings.create(
                model="text-embedding-3-large", input=text_list
            )
            #embeddings = [item['embedding'] for item in response['data']]
            return response
        except Exception as e:
            print(f"An error occurred: {e}")
            return []
    