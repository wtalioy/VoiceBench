from openai import OpenAI
import httpx
import io
import soundfile as sf
import base64
from argparse import ArgumentParser
from datasets import load_dataset, Audio
import json
from tqdm import tqdm
from loguru import logger
from src.models.base import VoiceAssistant

openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"

class LocalAssistant(VoiceAssistant):
    def __init__(self):
        self.client = OpenAI(
            api_key=openai_api_key,
            base_url=openai_api_base,
            http_client=httpx.Client()
        )

        models = self.client.models.list()
        self.model_name = models.data[0].id

    def generate_audio(
        self,
        audio,
        max_new_tokens=2048,
    ):
        # Write the audio data to an in-memory buffer in WAV format
        buffer = io.BytesIO()
        sf.write(buffer, audio['array'], audio['sampling_rate'], format='WAV')
        buffer.seek(0)  # Reset buffer position to the beginning

        # Read buffer as bytes and encode in base64
        wav_data = buffer.read()
        encoded_string = base64.b64encode(wav_data).decode('utf-8')

        completion = self.client.chat.completions.create(
            model=self.model_name,
            max_tokens=max_new_tokens,
            messages=[
                {"role": "system", "content": "You are a helpful assistant who tries to help answer the user's question."},
                {"role": "user", "content": [{"type": "input_audio", "input_audio": {"data": encoded_string, "format": 'wav'}}]},
            ]
        )

        token_usage = {
            "prompt_tokens": completion.usage.prompt_tokens,
            "completion_tokens": completion.usage.completion_tokens,
            "total_tokens": completion.usage.total_tokens
        }
        
        return completion.choices[0].message.content, token_usage


def main():
    parser = ArgumentParser()
    parser.add_argument('--data', type=str, default='alpacaeval')
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('--modality', type=str, default='audio', choices=['audio', 'text', 'ttft'])
    args = parser.parse_args()

    # load data
    data = load_dataset('hlt-lab/voicebench', args.data, split=args.split)
    data = data.cast_column("audio", Audio(sampling_rate=16_000))

    # load model
    model = LocalAssistant()

    # inference
    results = []
    for item in tqdm(data, total=len(data)):
        tmp = {k: v for k, v in item.items() if k != 'audio'}
        if args.modality == 'text':
            response = model.generate_text(item['prompt'])
            token_usage = None
        elif args.modality == 'audio':
            response, token_usage = model.generate_audio(item['audio'])
        elif args.modality == 'ttft':
            response = model.generate_ttft(item['audio'])
            token_usage = None
        else:
            raise NotImplementedError
        logger.info(item['prompt'])
        logger.info(response)
        if token_usage:
            logger.info(f"Token usage: Prompt={token_usage['prompt_tokens']}, Completion={token_usage['completion_tokens']}, Total={token_usage['total_tokens']}")
        logger.info('====================================')
        tmp['response'] = response
        if token_usage:
            tmp['token_usage'] = token_usage
        results.append(tmp)

    # save results
    model_name = model.model_name.split('/')[-1]
    output_file = f'{model_name}-{args.data}-{args.split}-{args.modality}.jsonl'
    with open(output_file, 'w') as f:
        for record in results:
            json_line = json.dumps(record)  # Convert dictionary to JSON string
            f.write(json_line + '\n')


if __name__ == '__main__':
    main()