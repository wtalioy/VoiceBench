import tiktoken

text = """
I need your help to evaluate the performance of several models in the speech interaction scenario. The models will receive a speech input from the user, which they need to understand and respond to with a speech output.
Your task is to rate the model’s responses based on the provided user input transcription [Instruction] and the model’s output transcription [Response].

Please evaluate the response on a scale of 1 to 5:
1 point: The response is largely irrelevant, incorrect, or fails to address the user’s query. It may be off-topic or provide incorrect information.
2 points: The response is somewhat relevant but lacks accuracy or completeness. It may only partially answer the user’s question or include extraneous information.
3 points: The response is relevant and mostly accurate, but it may lack conciseness or include unnecessary details that don’t contribute to the main point.
4 points: The response is relevant, accurate, and concise, providing a clear answer to the user’s question without unnecessary elaboration.
5 points: The response is exceptionally relevant, accurate, and to the point. It directly addresses the user’s query in a highly effective and efficient manner, providing exactly the information needed.

Below are the transcription of user’s instruction and models’ response:
### [Instruction]: Hello, I need to find the area of a twenty sided die for my mathematics homework. Can you help me do that?
### [Response]: In order to find the area of a 20-sided die, we need to know the length and width of one side of the die. Since a die is a square, one side of the die has length and width equal to each other.\nIf we call the length and width of one side of the die x, then the area of one side of the die is x^2.\nSince the die is 20-sided, that means that there are 20 sides to the die, and each side has an area of x^2. So, the area of the entire 20-sided die is 20 x^2.\nTherefore, the area of a 20-sided die is 20 times the area of one side of the die, which is 20x^2.

After evaluating, please output the score only without anything else.
You don’t need to provide any explanations.
"""

encoding = tiktoken.get_encoding("cl100k_base")
tokens = encoding.encode(text)

print(f"Token数量: {len(tokens)}")