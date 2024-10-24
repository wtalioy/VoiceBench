import torch


class VoiceAssistant:
    @torch.no_grad()
    def generate_audio(
        self,
        audio,
    ):
        raise NotImplementedError

    @torch.no_grad()
    def generate_text(
        self,
        text,
    ):
        raise NotImplementedError
