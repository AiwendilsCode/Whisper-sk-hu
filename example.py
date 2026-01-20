from safetensors.torch import load_file
import torch
from transformers import WhisperForConditionalGeneration
from transformers import WhisperProcessor
import librosa

processor = WhisperProcessor.from_pretrained("openai/whisper-base")

def load_whisper_model(device: str, model_path: str):
    """
    Load a Whisper model from a safetensors file.
    
    Args:
        device: "cuda" or "cpu" - the device to load the model on
        model_path: Path to the safetensors file
        
    Returns:
        The loaded model
    """
    
    if device not in ["cuda", "cpu"]:
        raise ValueError("Device must be 'cuda' or 'cpu'")
    
    # Load the model state dict from safetensors
    state_dict = load_file(model_path)
    
    # Load the Whisper model architecture and state dict
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base")
    # Load with strict=False to allow missing keys like proj_out.weight
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    
    return model

def load_whisper_base_model(device: str):
    """
    Load the base Whisper model from Hugging Face.
    
    Args:
        device: "cuda" or "cpu" - the device to load the model on
        
    Returns:
        The loaded base model
    """
    
    if device not in ["cuda", "cpu"]:
        raise ValueError("Device must be 'cuda' or 'cpu'")
    
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base")
    model.to(device)
    model.eval()
    
    return model

def transcribe_audio(model, audio, language: str):
    """
    Transcribe audio using the Whisper model.
    
    Args:
        model: The loaded Whisper model
        audio: Audio input (waveform tensor or path)
        language: Language code for transcription (e.g., "en" for English)
        
    Returns:
        Transcribed text as a string
    """
    
    # Process audio input
    inputs = processor(audio, sampling_rate=16000, language=language, return_tensors="pt")
    inputs = {key: value.to(model.device) for key, value in inputs.items()}
    
    # Generate transcription
    with torch.no_grad():
        generated_ids = model.generate(**inputs)
    
    # Decode to text
    transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    return transcription

if __name__ == "__main__":
    
    WHISPER_MODEL_PATH = "path/to/whisper_base_model.safetensors"
    LANGUAGE = "hu"  # Example: Hungarian | "sk" for Slovak
    AUDIO_FILE_PATH = "path/to/hungarian_audio.wav"
    
    # Load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_whisper_model(device, WHISPER_MODEL_PATH)
    
    # Example 1: Transcribe Hungarian audio
    hungarian_audio_path = AUDIO_FILE_PATH
    hungarian_waveform, sr = librosa.load(hungarian_audio_path, sr=16000)
    hungarian_text = transcribe_audio(model, hungarian_waveform, language=LANGUAGE)
    print(f"{LANGUAGE} transcription: {hungarian_text}")