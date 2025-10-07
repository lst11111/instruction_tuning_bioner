from dataclasses import dataclass
from typing import Dict

import yaml
class Hypernum:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    @classmethod
    def from_yaml(cls, file_path):
        with open(file_path, 'r') as f:
            params = yaml.safe_load(f)
        return cls(**params)
@dataclass
class Template:
    template_name:str
    system_format: str
    user_format: str
    assistant_format: str
    stop_word: str
    # stop_token_id: int


template_dict: Dict[str, Template] = dict()


def register_template(template_name, system_format, user_format, assistant_format, stop_word=None):
    template_dict[template_name] = Template(
        template_name=template_name,
        system_format=system_format,
        user_format=user_format,
        assistant_format=assistant_format,
        stop_word=stop_word,
        # stop_token_id=stop_token_id
    )




register_template(
    template_name='qwen',
    system_format='<|im_start|>system\n{content}<|im_end|>\n',
    user_format='<|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n',
    assistant_format='{content}<|endoftext|>\n',
    stop_word='<|endoftext|>'
)

register_template(
    template_name='ner',
    system_format='<|im_start|>system\n{content}<|im_end|>\n',##指令的设计
    user_format='<|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n',##输入文本的设计
    assistant_format='{content}<|endoftext|>\n',  ##输出的设计
    stop_word='<|endoftext|>'
)

register_template(
    template_name='llama',
    system_format='<<SYS>>\n{content}\n<</SYS>>\n\n',
    user_format='[INST]{content}[/INST]',
    assistant_format='{content} </s>',
    stop_word='</s>'
)