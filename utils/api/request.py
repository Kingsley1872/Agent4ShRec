import time
from openai import OpenAI

client = OpenAI(
    api_key="",  # Add your API key here
)

def get_response_from_openai(prompt, model, temperature=1.2):
    messages = [
        {'role': 'user', 'content': prompt}
    ]
    max_retries = 10
    retries = 0
    while retries < max_retries:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=800,
            n=1,
            stop=None,
            temperature=temperature,
        )
        if len(response.choices) > 0:
            response_text = response.choices[0].message.content.strip()
            return response_text

    print("Max retries reached, exiting.")
    return None
