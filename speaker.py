import torchaudio
from torch.nn.functional import pad
from speechbrain.inference import EncoderClassifier
from scipy.spatial.distance import cosine

# Preprocess audio
def preprocess_audio(file_path, sample_rate=16000):
    audio, sr = torchaudio.load(file_path)
    if sr != sample_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=sample_rate)
        audio = resampler(audio)
    audio = audio.mean(dim=0)  # Convert stereo to mono
    max_length = 16000 * 10  # Example: 10 seconds max length
    if audio.size(0) < max_length:
        padding = max_length - audio.size(0)
        audio = pad(audio, (0, padding))
    return audio

# Extract embeddings
def extract_embedding(audio_tensor, model):
    """
    Extracts the speaker embedding from the audio tensor.
    Ensures the embedding is flattened into a 1-D vector.
    """
    audio_tensor = audio_tensor.unsqueeze(0)  # Add batch dimension
    embedding = model.encode_batch(audio_tensor)
    return embedding.squeeze().detach().numpy().flatten()  # Ensure it's 1-D

# Calculate cosine similarity
def calculate_cosine_similarity(embedding1, embedding2):
    return 1 - cosine(embedding1, embedding2)

# Load the model
model = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-xvect-voxceleb", savedir="pretrained_model"
)

# Paths to audio files
audio_file_1 = "logs/ivr/e830M/output/prompts/0.wav"  # Replace with your audio file
audio_file_2 = "logs/ivr/e830M/output/samples_enhprompts/40.wav"  # Replace with your audio file

# Preprocess and extract embeddings
audio_1 = preprocess_audio(audio_file_1)
audio_2 = preprocess_audio(audio_file_2)

embedding_1 = extract_embedding(audio_1, model)
embedding_2 = extract_embedding(audio_2, model)

# Compute cosine similarity
similarity = calculate_cosine_similarity(embedding_1, embedding_2)

# Display results
print(f"Cosine Similarity: {similarity}")
if similarity > 0.5:  # Adjust threshold if needed
    print("Speaker Verified: The voices match.")
else:
    print("Speaker Verification Failed: The voices do not match.")