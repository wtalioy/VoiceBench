from .base import VoiceAssistant
import io
import base64
import os
from openai import OpenAI
import soundfile as sf

class QwenOmniAssistant(VoiceAssistant):
    def __init__(self):
        self.client = OpenAI(
            api_key=os.getenv("API_KEY"),
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        self.model_name = "qwen-omni-turbo"
    
    def generate_audio(
        self,
        audio,
        max_new_tokens=2048,
    ):
        # Write the audio data to an in-memory buffer in WAV format
        buffer = io.BytesIO()
        sf.write(buffer, audio['array'], audio['sampling_rate'], format='WAV')
        buffer.seek(0)

        # Read buffer as bytes and encode in base64
        wav_data = buffer.read()
        encoded_string = base64.b64encode(wav_data).decode('utf-8')

        completion = self.client.chat.completions.create(
            model=self.model_name,
            max_tokens=max_new_tokens,
            messages=[
                {
                    "role": "system",
                    "content": [{"type": "text", "text": "You are a helpful assistant."}],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_audio",
                            "input_audio": {
                                "data": f"data:;base64,{encoded_string}",
                                "format": "wav",
                            },
                        }
                    ]
                }
            ],
            modalities=["text"],
            stream=True,
            stream_options={"include_usage": True},
        )
        token_usage = {
            "prompt_tokens": completion.usage.prompt_tokens,
            "completion_tokens": completion.usage.completion_tokens,
            "total_tokens": completion.usage.total_tokens
        }
        return completion.choices[0].message.content, token_usage