from .diva import DiVAAssistant
from .qwen2 import Qwen2Assistant
from .naive import NaiveAssistant

model_cls_mapping = {
    'qwen2': Qwen2Assistant,
    'diva': DiVAAssistant,
    'naive': NaiveAssistant,
}
