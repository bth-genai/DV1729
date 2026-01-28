import re
from ollama import chat
from ollama import ChatResponse

#Number of rounds to loop chat/response
MAX_ROUNDS = 10
#Models in discussion
models = ["llama3.2", "deepseek-r1"]

#Function for removing "think section" in LLM-answers (change RE as needed)
def removeThinkSection(text):
    responseString = re.sub(r'<think>.+?</think>', '', text, flags=re.DOTALL)
    return (responseString)

def main():
    print("***This application lets you start a discussion between two AI-LLMs, alternating between their answers-***\n")
    
    #Get initial question from user
    question = input("Please state your discussion question: ")
    
    #Get starting model
    modelToGoFirst = 0
    while(modelToGoFirst != 1 and modelToGoFirst != 2):
        print("Which model should go first?\n1. Llama 3.2\n2. Deepseek R1")
        string = input("-->: ")
        try:
            modelToGoFirst = int(string)
        except ValueError:
            # Handle the exception
            print('Please enter an integer!')
 
    #Round 1
    response1: ChatResponse = chat(model=models[modelToGoFirst-1], messages=[
    {
        'role': 'user',
        'content': question,
    }], think=False)
    responseString = response1.message.content
    print("\n\n[" + models[modelToGoFirst-1] + "]: " + responseString)
    
    #Round 2
    response2: ChatResponse = chat(model=models[modelToGoFirst%2], messages=[
    {
        'role': 'user',
        'content': "I asked someone this question: '" + question + "'. Their answer was: '"
        + responseString + "'. End your response with a sharp follow-up question to counterpoint their answer.",
    }])
    responseString = response2.message.content
    print("\n\n[" + models[modelToGoFirst%2] + "]: " + responseString)
    question = responseString
    
    #Chat/response-loop
    MAX_ROUNDS = 0
    while(MAX_ROUNDS < 5 or MAX_ROUNDS > 12):
        string = input("How many rounds shall we alternate? Choose 5-12: ")
        try:
            MAX_ROUNDS = int(string)
        except ValueError:
            # Handle the exception
            print('Please enter an integer!')

        for i in range(MAX_ROUNDS):
            modelToGoNext = 0
            while(modelToGoNext != 1 and modelToGoNext != 2):
                print("Which model should go next?\n1. Llama 3.2\n2. Deepseek R1")
                string = input("-->: ")
                try:
                    modelToGoNext = int(string)
                except ValueError:
                    # Handle the exception
                    print('Please enter an integer!')
            
            temperature = -1.0
            while(temperature < 0 or temperature > 1.5):
                string = input("Change the temperature, i.e. how 'creative' the answers can be? "\
                    "Higher = more creative. Choose 0 - 1.5: ")
                try:
                    temperature = float(string)
                except ValueError:
                    # Handle the exception
                    print('Please enter a float!')
            
            response: ChatResponse = chat(model=models[modelToGoNext-1], messages=[
                {
                    'role': 'system',
                    'content': 'You are a discussion partner. Please respond to questions and statements." \
                        Always end your response with a sharp follow-up question to counterpoint their argument.',
                },
                {
                    'role': 'user',
                    'content': question,
                },
            ],
            options={
                'temperature': temperature
                }
            )
            responseString = response.message.content
            print("\n\n[" + models[modelToGoNext-1] + "]: " + responseString)
            question = responseString


if __name__ == "__main__":
    main()
