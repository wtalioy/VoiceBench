# VoiceBench

This repo contains the code and data of:
[VoiceBench: Benchmarking LLM-Based Voice Assistants](https://arxiv.org/abs/2410.17196)

## News
* **`2024.12.10`** Added a curated list of awesome voice assistants.
* **`2024.11.24`** Expanded the test samples in VoiceBench to include `mmsu`, covering 12 diverse domains from `mmlu-pro`.
* **`2024.11.12`** Updated the VoiceBench Leaderboard to include: 1) Mini-Omni2, GPT-4o-Audio, and Whisper-v3+GPT-4o, and 2) multiple-choice QA from OpenBookQA.
* **`2024.10.30`** Expanded the test samples in VoiceBench to include: 1) the complete set of open-ended QA from `alpacaeval`, and 2) multiple-choice QA from `openbookqa`.

## Leaderboard

| Rank | Model                         | AlpacaEval | CommonEval | SD-QA | OpenBookQA | IFEval | AdvBench | Overall |
|:----:|-------------------------------|:----------:|:----------:|:-----:|:----------:|:------:|:--------:|:-------:|
|  1   | Whisper-v3-large+GPT-4o       |    4.80    |    4.47    | 75.77 |   92.97    | 76.51  |  98.27   |  88.15  |
|  2   | GPT-4o-Audio                  |    4.78    |    4.49    | 75.50 |   89.23    | 76.02  |  98.65   |  87.45  |
|  3   | Whisper-v3-large+LLaMA-3.1-8B |    4.53    |    4.04    | 70.43 |   81.54    | 69.53  |  98.08   |  81.83  |
|  4   | Whisper-v3-turbo+LLaMA-3.1-8B |    4.55    |    4.02    | 58.23 |   72.09    | 71.12  |  98.46   |  78.55  |
|  5   | Whisper-v3-turbo+LLaMA-3.2-3B |    4.45    |    3.82    | 49.28 |   60.66    | 69.71  |  98.08   |  73.86  |
|  6   | DiVA                          |    3.67    |    3.54    | 57.05 |   25.49    | 39.15  |  98.27   |  60.69  |
|  7   | GLM-4-Voice                   |    3.97    |    3.42    | 36.98 |   53.41    | 25.92  |  88.08   |  58.70  |
|  8   | Qwen2-Audio                   |    3.74    |    3.43    | 35.71 |   49.45    | 26.33  |  96.73   |  58.62  |
|  9   | LLaMA-Omni                    |    3.70    |    3.46    | 39.69 |   27.47    | 14.87  |  11.35   |  39.44  |
|  10  | VITA                          |    3.38    |    2.15    | 27.94 |   29.01    | 22.82  |  26.73   |  36.18  |
|  11  | Mini-Omni2                    |    2.32    |    2.18    | 9.31  |   26.59    | 11.56  |  57.50   |  32.49  |
|  12  | Mini-Omni                     |    1.95    |    2.02    | 13.92 |   26.59    | 13.58  |  37.12   |  28.44  |
|  13  | Moshi                         |    2.01    |    1.60    | 15.64 |   25.93    | 10.12  |  44.23   |  28.02  |


We encourage you to submit new voice assistant results directly through the issue tracker. The ranking list will be updated accordingly.

## Setup
```shell
conda create -n voicebench python=3.10
conda activate voicebench
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121
pip install xformers==0.0.23 --no-deps
```

## Dataset

The data used in this project is available at [VoiceBench Dataset](https://huggingface.co/datasets/hlt-lab/voicebench) hosted on Hugging Face.

You can access it directly via the link and integrate it into your project by using the Hugging Face `datasets` library.

### How to Use the Dataset

To load the dataset in your Python environment:

```python
from datasets import load_dataset

# Load the VoiceBench dataset
# Available subset: alpacaeval, commoneval, sd-qa, ifeval, advbench, ...
dataset = load_dataset("hlt-lab/voicebench", 'alpacaeval')
```

### Available Data

| Subset          | # Samples | Audio Source |       Task Type       |
|-----------------|:---------:|:------------:|:---------------------:|
| alpacaeval      |    199    |  Google TTS  |     Open-Ended QA     |
| alpacaeval_full |    636    |  Google TTS  |     Open-Ended QA     |
| commoneval      |    200    |    Human     |     Open-Ended QA     |
| openbookqa      |    455    |  Google TTS  |  Multiple-Choice QA   |
| mmsu            |    3,074  |  Google TTS  |  Multiple-Choice QA   |
| sd-qa           |    553    |    Human     |  Reference-Based QA   |
| mtbench         |    46     |  Google TTS  |     Multi-Turn QA     |
| ifeval          |    345    |  Google TTS  | Instruction Following |
| advbench        |    520    |  Google TTS  |        Safety         |


**PS**: `alpacaeval` contains `helpful_base` and `vicuna` data, while `alpacaeval_full` is constructed with the complete data.


## Evaluation
### Step 1: Get the Voice Assistant's Response
To obtain the responses from the voice assistant model, run the following command:
```shell
python main.py --model naive --data alpacaeval --split test --modality audio
```

**Supported Arguments:**
- `--model`: Specifies the model to use for generating responses. Replace `naive` with the model you want to test (e.g., `qwen2`, `diva`).
- `--data`: Selects the subset of the dataset. Replace `alpacaeval` with other subsets like `commoneval`, `sd-qa`, etc., depending on your evaluation needs.
- `--split`: Chooses the data split to evaluate.
    - For most datasets (`alpacaeval`, `commoneval`, `ifeval`, `advbench`), use `test` as the value.
    - For the `sd-qa` subset, you should provide a region code instead of `test`, such as `aus` for Australia, `usa` for the United States, etc.
- `--modality`: Use `audio` for spoken instructions, `text` for text-based instructions.

This will generate the output and save it to a file named naive-alpacaeval-test-audio.jsonl.

### Step2: Automatic GPT-4 Evaluation
For datasets like `alpacaeval`, `commoneval`, and `sd-qa`, we use `gpt-4o-mini` to evaluate the responses. Run the following command to get the GPT score:
```shell
python api_judge.py --src_file naive-alpacaeval-test-audio.jsonl
```
The GPT evaluation scores will be saved to `result-naive-alpacaeval-test-audio.jsonl`.

**Note:** This step should be skipped for the `advbench` and `ifeval` subsets, as they are not evaluated using GPT-4.

### Step3: Get the Final Results
To generate the final evaluation results, run:
```shell
python evaluate.py --src_file result-naive-alpacaeval-test-audio.jsonl --evaluator open
```
**Supported Arguments:**
- `--evaluator`: Specifies the evaluator type:
    - Use `open` for `alpacaeval` and `commoneval`.
    - Use `qa` for `sd-qa`.
    - Use `ifeval` for `ifeval`.
    - Use `harm` for `advbench`.
    - Use `mcq` for `openbookqa`.

## Awesome Voice Assistants
| Title                                                                                                                                                                                                                              |    Date    |                        Code                        |
|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:----------:|:--------------------------------------------------:|
| [**GLM-4-Voice: Towards Intelligent and Human-Like End-to-End Spoken Chatbot**](https://arxiv.org/abs/2412.02612) &nbsp; ![Star](https://img.shields.io/github/stars/THUDM/GLM-4-Voice.svg?style=social&label=Star)                | 2024-12-03 |   [Github](https://github.com/THUDM/GLM-4-Voice)   |
| [**Advancing Speech Language Models by Scaling Supervised Fine-Tuning with Over 60,000 Hours of Synthetic Speech Dialogue Data**](https://arxiv.org/abs/2412.01078)                                                                | 2024-12-02 |                         --                         |
| [**SALMONN-omni: A Codec-free LLM for Full-duplex Speech Understanding and Generation**](https://arxiv.org/abs/2411.18138)                                                                                                         | 2024-11-27 |                         --                         |
| [**Freeze-Omni: A Smart and Low Latency Speech-to-speech Dialogue Model with Frozen LLM**](https://arxiv.org/abs/2411.00774) &nbsp; ![Star](https://img.shields.io/github/stars/VITA-MLLM/Freeze-Omni.svg?style=social&label=Star) | 2024-11-01 | [Github](https://github.com/VITA-MLLM/Freeze-Omni) |
| [**OmniFlatten: An End-to-end GPT Model for Seamless Voice Conversation**](https://arxiv.org/abs/2410.17799)                                                                                                                       | 2024-10-23 |                         --                         |
| [**Ichigo: Mixed-Modal Early-Fusion Realtime Voice Assistant**](https://arxiv.org/abs/2410.15316)                                                                                                                                  | 2024-10-20 |                         --                         |
| [**Mini-Omni2: Towards Open-source GPT-4o with Vision, Speech and Duplex Capabilities**](https://arxiv.org/abs/2410.11190) &nbsp; ![Star](https://img.shields.io/github/stars/gpt-omni/mini-omni2.svg?style=social&label=Star)     | 2024-10-15 |  [Github](https://github.com/gpt-omni/mini-omni2)  |
| [**Distilling an End-to-End Voice Assistant Without Instruction Training Data**](https://arxiv.org/abs/2410.02678)                                                                                                                 | 2024-10-03 |                         --                         |
| [**Moshi: a Speech-Text Foundation Model for Real-Time Dialogue**](https://arxiv.org/abs/2410.00037) &nbsp; ![Star](https://img.shields.io/github/stars/kyutai-labs/moshi.svg?style=social&label=Star)                             | 2024-09-17 |   [Github](https://github.com/kyutai-labs/moshi)   |
| [**LLaMA-Omni: Seamless Speech Interaction with Large Language Models**](https://arxiv.org/abs/2409.06666) &nbsp; ![Star](https://img.shields.io/github/stars/ictnlp/LLaMA-Omni.svg?style=social&label=Star)                       | 2024-09-10 |   [Github](https://github.com/ictnlp/LLaMA-Omni)   |
| [**Mini-Omni: Language Models Can Hear, Talk While Thinking in Streaming**](https://arxiv.org/abs/2408.16725) &nbsp; ![Star](https://img.shields.io/github/stars/gpt-omni/mini-omni.svg?style=social&label=Star)                   | 2024-08-29 |  [Github](https://github.com/gpt-omni/mini-omni)   |
| [**VITA: Towards Open-Source Interactive Omni Multimodal LLM**](https://arxiv.org/abs/2408.05211) &nbsp; ![Star](https://img.shields.io/github/stars/VITA-MLLM/VITA.svg?style=social&label=Star)                                   | 2024-08-09 |    [Github](https://github.com/VITA-MLLM/VITA)     |
| [**Qwen2-Audio Technical Report**](https://arxiv.org/abs/2407.10759) &nbsp; ![Star](https://img.shields.io/github/stars/QwenLM/Qwen2-Audio.svg?style=social&label=Star)                                                            | 2024-07-15 |  [Github](https://github.com/QwenLM/Qwen2-Audio)   |

## Citation
If you use the VoiceBench in your research, please cite the following paper:
```
@article{chen2024voicebench,
  title={VoiceBench: Benchmarking LLM-Based Voice Assistants},
  author={Chen, Yiming and Yue, Xianghu and Zhang, Chen and Gao, Xiaoxue and Tan, Robby T. and Li, Haizhou},
  journal={arXiv preprint arXiv:2410.17196},
  year={2024}
}
```
