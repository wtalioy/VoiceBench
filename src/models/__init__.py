from .diva import DiVAAssistant
from .qwen2 import Qwen2Assistant
from .naive import NaiveAssistant
from .mini_omni import MiniOmniAssistant
from .mini_omni2 import MiniOmni2Assistant
from .gpt4o import GPT4oAssistant
from .naive2 import Naive2Assistant
from .moshi import MoshiAssistant

model_cls_mapping = {
    'qwen2': Qwen2Assistant,
    'diva': DiVAAssistant,
    'naive': NaiveAssistant,
    'naive2': Naive2Assistant,
    'mini_omni': MiniOmniAssistant,
    'mini_omni2': MiniOmni2Assistant,
    'gpt4o': GPT4oAssistant,
    'moshi': MoshiAssistant,
}
