from .diva import DiVAAssistant
from .qwen2 import Qwen2Assistant
from .naive import NaiveAssistant
from .mini_omni import MiniOmniAssistant
from .mini_omni2 import MiniOmni2Assistant

model_cls_mapping = {
    'qwen2': Qwen2Assistant,
    'diva': DiVAAssistant,
    'naive': NaiveAssistant,
    'mini_omni': MiniOmniAssistant,
    'mini_omni2': MiniOmni2Assistant,
}
