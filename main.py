from ollama import chat
from ollama import ChatResponse



def main():
    response: ChatResponse = chat(model='llama3.2', messages=[
    {
        'role': 'user',
        'content': 'Is Donald Trump a bad person?',
    },
    ])

    print(response.message.content)


if __name__ == "__main__":
    main()
