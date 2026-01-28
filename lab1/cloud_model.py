import os
from ollama import Client, ChatResponse

client = Client(
    host="https://ollama.com",
    headers={'Authorization': 'Bearer ' + os.environ.get('OLLAMA_API_KEY')}
)

messages = [
  {
    'role': 'user',
    'content': 'Why do cats like to sit in boxes?',
  },
]

#Get user input
while(True):
    print("Do you want a streamed response or wait untill all is loaded?")
    print("1: Streamed")
    print("2: Wait")
    streaming = input("-->: ")
    if streaming == "1":
        streaming = True
        break
    elif streaming == "2":
        streaming = False
        break
    else:
        print("Please enter 1 or 2!")

# Either print stream or wait for full response
if streaming:
    for part in client.chat('gpt-oss:120b', messages=messages, stream=True):
        print(part['message']['content'], end='', flush=True)
else:
    response: ChatResponse = client.chat(model='gpt-oss:120b', messages=messages)
    print(response.message.content)