# VoiceBench

This repo contains the code and data of:
[VoiceBench: Benchmarking LLM-Based Voice Assistants](https://arxiv.org/abs/2410.17196)

## News
* **`2024.10.30`** We expanded the test samples in VoiceBench to include: 1) the complete set of open-ended QA from `alpacaeval`, and 2) multiple-choice QA from `openbookqa`.

## Leaderboard

| Rank | Model                 |    AlpacaEval   |    CommonEval   |    SD-QA   |    IFEval   |   AdvBench   |   Overall   |
|------|-----------------------|:---------------:|:---------------:|:----------:|:-----------:|:------------:|:-----------:|
| 1    | Whisper-v3+LLaMA-3.1   | 4.68       | 4.04       | 70.43 | 69.53  | 98.08    | 74.50   |
| 2    | DiVA                   | 3.86       | 3.54       | 57.05 | 39.15  | 98.27    | 64.02   |
| 3    | Qwen2-Audio            | 3.89       | 3.43       | 35.71 | 26.33  | 96.73    | 59.83   |
| 4    | Mini-Omni              | 2.25       | 2.02       | 13.92 | 13.58  | 37.12    | 41.56   |
| 5    | LLaMA-Omni             | 3.95       | 3.46       | 39.69 | 14.87  | 11.35    | 40.21   |
| 6    | VITA                   | 3.78       | 2.15       | 27.94 | 22.82  | 26.73    | 39.33   |

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
# Available subset: alpacaeval, commoneval, sd-qa, ifeval, advbench
dataset = load_dataset("hlt-lab/voicebench", 'alpacaeval')
```

### Available Data

| Subset          | # Samples | Audio Source |       Task Type       |
|-----------------|:---------:|:------------:|:---------------------:|
| alpacaeval      |    199    |  Google TTS  |     Open-Ended QA     |
| alpacaeval_full |    636    |  Google TTS  |     Open-Ended QA     |
| commoneval      |    200    |    Human     |     Open-Ended QA     |
| openbookqa      |    455    |  Google TTS  |  Multiple-Choice QA   |
| sd-qa           |    553    |    Human     |  Reference-Based QA   |
| ifeval          |    345    |  Google TTS  | Instruction Following |
| advbench        |    520    |  Google TTS  |        Safety         |


**PS**: `alpacaeval` (used in our paper) contains `helpful_base` and `vicuna` data, while `alpacaeval_full` is constructed with the complete data.


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

## Citation
If you use the VoiceBench dataset in your research, please cite the following paper:
```
@article{chen2024voicebench,
  title={VoiceBench: Benchmarking LLM-Based Voice Assistants},
  author={Chen, Yiming and Yue, Xianghu and Zhang, Chen and Gao, Xiaoxue and Tan, Robby T. and Li, Haizhou},
  journal={arXiv preprint arXiv:2410.17196},
  year={2024}
}
```
