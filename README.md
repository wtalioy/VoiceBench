# VoiceBench

This repo contains the data of:
[VoiceBench: Benchmarking LLM-Based Voice Assistants](https://arxiv.org/abs/2410.17196)

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