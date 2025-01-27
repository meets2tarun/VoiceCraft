import torchaudio
import torch
from speechbrain.pretrained import SpeakerRecognition
from scipy.spatial.distance import cosine

# Load pre-trained ECAPA-TDNN model from SpeechBrain
recognizer = SpeakerRecognition.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb", savedir="pretrained_model"
)

# Preprocess audio with fixed duration
def preprocess_audio(file_path, sample_rate=16000, fixed_duration=5):
    """
    Preprocess audio by resampling and ensuring fixed duration.
    """
    audio, sr = torchaudio.load(file_path)
    if sr != sample_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=sample_rate)
        audio = resampler(audio)
    audio = audio.mean(dim=0)  # Convert stereo to mono

    # Adjust audio length to fixed_duration (in seconds)
    max_length = int(sample_rate * fixed_duration)  # e.g., 5 seconds
    if audio.size(0) > max_length:
        audio = audio[:max_length]  # Truncate if too long
    elif audio.size(0) < max_length:
        padding = max_length - audio.size(0)
        audio = torch.nn.functional.pad(audio, (0, padding))  # Pad if too short

    return audio

# Extract speaker embeddings
def extract_embedding(audio_path, recognizer):
    """
    Extracts the speaker embedding from the audio tensor.
    Ensures the embedding is flattened into a 1-D vector.
    """
    audio = preprocess_audio(audio_path)
    embedding = recognizer.encode_batch(audio).squeeze(0).detach().numpy()
    return embedding.flatten()  # Flatten ensures it is 1-D

# Calculate cosine similarity
def calculate_cosine_similarity(embedding1, embedding2):
    """
    Computes cosine similarity between two embeddings.
    """
    return 1 - cosine(embedding1, embedding2)

# Paths to Hindi audio files
audio_file_1 = "logs/ivr/e830M/output/prompts/31.wav"  # Replace with actual file path
audio_file_2 = "logs/ivr/e830M/output/samples_enhprompts/31.wav"  # Replace with actual file path

# Extract embeddings and compute similarity
embedding_1 = extract_embedding(audio_file_1, recognizer)
embedding_2 = extract_embedding(audio_file_2, recognizer)

# Ensure embeddings have the same shape
assert embedding_1.shape == embedding_2.shape, "Embeddings have mismatched shapes!"

similarity = calculate_cosine_similarity(embedding_1, embedding_2)

# Display results
print(f"Cosine Similarity: {similarity}")
if similarity > 0.5:
    print("Speaker Verified: The voices match.")
else:
    print("Speaker Verification Failed: The voices do not match.")