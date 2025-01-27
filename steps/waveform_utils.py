import torch
from audiocraft.models import EncodecModel

class InferenceDecodeWrapper:
    def __init__(self, checkpoint_path, device):
        self.device = device
        self.model = EncodecModel.get_pretrained('encodec_24khz').to(self.device)  # Initialize pretrained model
        self.load_checkpoint(checkpoint_path)

    def load_checkpoint(self, checkpoint_path):
        """
        Load the model checkpoint for decoding.
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model"], strict=False)
        self.model.eval()

    def decode_tokens_to_waveform(self, tokens):
        """
        Decode tokens into waveform using the Encodec decoder.
        Args:
            tokens (torch.Tensor): Input tokens of shape [B, K, T].
        Returns:
            torch.Tensor: Decoded waveform.
        """
        # Ensure correct input shape for decoder
        if tokens.ndim == 3:
            tokens = tokens.permute(1, 0, 2)  # Convert [B, K, T] -> [K, B, T]
        else:
            raise ValueError(f"Invalid tokens shape: {tokens.shape}")

        # Decode tokens into waveform
        with torch.no_grad():
            quantized_embeddings = self.model.quantizer.decode(tokens)
            waveform = self.model.decoder(quantized_embeddings)
        return waveform
    