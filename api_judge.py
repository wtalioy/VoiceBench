from tqdm import tqdm
import multiprocessing
import json
from openai import OpenAI
from src.api import generate_text_chat
from argparse import ArgumentParser
import os
import httpx

client = OpenAI(
    api_key=os.getenv('OPENAI_API_KEY'),
    base_url="https://172.203.11.191:3826/v1",
    http_client=httpx.Client()
)

meta_prompt_open = """
I need your help to evaluate the performance of several models in the speech interaction scenario. The models will receive a speech input from the user, which they need to understand and respond to with a speech output.
Your task is to rate the model’s responses based on the provided user input transcription [Instruction] and the model’s output transcription [Response].

Please evaluate the response on a scale of 1 to 5:
1 point: The response is largely irrelevant, incorrect, or fails to address the user’s query. It may be off-topic or provide incorrect information.
2 points: The response is somewhat relevant but lacks accuracy or completeness. It may only partially answer the user’s question or include extraneous information.
3 points: The response is relevant and mostly accurate, but it may lack conciseness or include unnecessary details that don’t contribute to the main point.
4 points: The response is relevant, accurate, and concise, providing a clear answer to the user’s question without unnecessary elaboration.
5 points: The response is exceptionally relevant, accurate, and to the point. It directly addresses the user’s query in a highly effective and efficient manner, providing exactly the information needed.

Below are the transcription of user’s instruction and models’ response:
### [Instruction]: {prompt}
### [Response]: {response}

After evaluating, please output the score only without anything else.
You don’t need to provide any explanations.
"""

meta_prompt_qa = """
### Question
{prompt}

### Reference answer
{reference}

### Candidate answer
{response}

Is the candidate answer correct based on the question and reference answer? 
Please only output a single "Yes" or "No". Do not output anything else.
""".strip()


def generate(item):
    if "reference" in item:
        prompt = meta_prompt_qa.replace('{prompt}', item['prompt']).replace('{reference}', item['reference']).replace('{response}', item['response'])
    else:
        prompt = meta_prompt_open.replace("{prompt}", item['prompt']).replace('{response}', item['response'])
    rtn = [
        item.message.content.strip() for item in generate_text_chat(
            client=client,
            model='gpt-4o-mini',
            messages=[{"role": "system",
                       "content": "You are a helpful assistant who tries to help answer the user's question."},
                      {"role": "user", "content": prompt}],
            max_tokens=1024,
            frequency_penalty=0,
            presence_penalty=0,
            stop=None,
            temperature=0.5, top_p=0.95, n=1
        ).choices
    ]
    item['score'] = rtn
    return item


def main():
    parser = ArgumentParser()
    parser.add_argument('--src-file', required=True)
    args = parser.parse_args()

    # read source data
    data = []
    with open(args.src_file, 'r') as f:
        for line in f:
            json_obj = json.loads(line.strip())  # Convert JSON string to dictionary
            data.append(json_obj)

    # evaluate data
    with multiprocessing.Pool(4) as pool:
        scores = list(tqdm(pool.imap(generate, data), total=len(data)))

    # save results
    os.makedirs('result', exist_ok=True)
    src_file = args.src_file.split('/')[-1]
    tgt_file = 'result/' + src_file
    with open(tgt_file, "w") as file:
        for d in scores:
            file.write(json.dumps(d) + "\n")

    print(f"Results saved to {tgt_file}")

if __name__ == '__main__':
    main()
