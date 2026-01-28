# DV1729
Course repo for DV1712/DV1729 lp3 VT26 Applied generative AI

## Installing the tools
You will need a code editor. Visual studio code is a great alternative: https://code.visualstudio.com/

We will use Python for labs and assignments. It is recommended to run Python using a virtual environment. This way you will not have to manually handle different system-wide Python-installations.

The example below will use uv, a very fast manager for both packages and projects (https://docs.astral.sh/uv/). Installation instructions: https://docs.astral.sh/uv/getting-started/installation/. You can however choose another tool to install virtual environments.

You also need to install Ollama (https://docs.ollama.com/quickstart/), and download the models you want to work with. Using a terminal of your choice, run:
``` bash
ollama pull llama3.2
```

If you want to clone this repository, you can use uv to initialize a project with the same dependenies. After downloading the repo, just run:

``` bash
uv sync
```

If you cloned the repository, skip to the example below. If you want to start from scratch, create a folder in which you want to work with the lab. Using a terminal of your choice, navigate to that folder and run the following commands:

Create a uv project in that folder

``` bash
uv init --name NameYourProject
```

Create a virtual environment for Python
``` bash
uv venv
```
Activate the virtual environment
Linux/Mac
``` bash
source ./.venv/bin/activate
```
Windows
``` cmd
.venv\Scripts\activate.bat
```
Install Ollama wrappers for Python
``` bash
uv add ollama
```
### Example
Now try a simple example. Edit the main.py file or create a new .py file.
``` python
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
```
Run the python program using uv
``` bash
uv run main.py
```
