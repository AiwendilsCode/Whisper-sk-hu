import pyaudio
import numpy as np
from collections import deque
import threading
import time
import torch
import torchaudio

def list_audio_input_devices():
    """List all available audio input devices"""
    print("\nAvailable audio devices:")
    for i in range(pyaudio.PyAudio().get_device_count()):
        info = pyaudio.PyAudio().get_device_info_by_index(i)
        if info['maxInputChannels'] > 0:
            print(f"  [{i}] {info['name']} (inputs: {info['maxInputChannels']})")
            
def list_audio_output_devices():
    """List all available audio output devices"""
    print("\nAvailable audio output devices:")
    for i in range(pyaudio.PyAudio().get_device_count()):
        info = pyaudio.PyAudio().get_device_info_by_index(i)
        if info['maxOutputChannels'] > 0:
            print(f"  [{i}] {info['name']} (outputs: {info['maxOutputChannels']})")

class MicrophoneCapture:
    """Capture audio from microphone in real-time"""
    
    def __init__(self, sample_rate=16000, chunk_size=2048, device_index=None):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.audio = pyaudio.PyAudio()
        self.stream = None
        self.is_recording = False
        self.audio_buffer = deque(maxlen=sample_rate * 30)  # 30 sec buffer
        self.device_index = device_index
        
        # List available devices
        #self._list_devices()
        #list_audio_devices()
        #list_audio_output_devices()
    
    def start(self):
        """Start microphone stream"""
        self.stream = self.audio.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size,
            input_device_index=self.device_index
        )
        self.is_recording = True
        threading.Thread(target=self._read_audio, daemon=True).start()
        device_info = self.audio.get_device_info_by_index(self.device_index or self.audio.get_default_input_device_info()['index'])
        print(f"Microphone started: {device_info['name']}")
        
    def _read_audio(self):
        """Background thread to capture audio"""
        while self.is_recording:
            try:
                data = self.stream.read(self.chunk_size, exception_on_overflow=False)
                audio_chunk = np.frombuffer(data, dtype=np.float32)
                # Normalize to [-1, 1] range if needed
                audio_chunk = np.clip(audio_chunk, -1.0, 1.0)
                self.audio_buffer.extend(audio_chunk)
            except Exception as e:
                print(f"Error reading audio: {e}")
                
    def get_audio_chunk(self):
        """Get current audio buffer without clearing"""
        if len(self.audio_buffer) == 0:
            return np.array([])
        current_audio = np.array(list(self.audio_buffer))
        self.flush_buffer()
        return current_audio

    def get_accumulated_audio(self, duration_seconds):
        """Get audio accumulated over specified duration"""
        target_samples = int(self.sample_rate * duration_seconds)
        accumulated = []
        
        while len(accumulated) < target_samples and self.is_recording:
            if self.audio_buffer:
                accumulated.extend(list(self.audio_buffer))
                self.flush_buffer()
            time.sleep(0.05)
        
        return np.array(accumulated[:target_samples])
    
    def stop(self):
        """Stop microphone stream"""
        self.is_recording = False
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self.audio.terminate()
        print("Microphone stopped")
        
    def flush_buffer(self):
        """Clear the audio buffer"""
        self.audio_buffer.clear()
        
def start_recording():
    mic = MicrophoneCapture(sample_rate=16000)
    mic.start()
    
    return mic

if __name__ == "__main__":
    mic = start_recording()
    print("Recording for 30 seconds...")
    time.sleep(30)  # Record for 30 seconds
    audio_data = mic.get_audio_chunk()
    mic.stop()
    print(f"Recorded {len(audio_data)} samples.")

    # Save audio to file
    torchaudio.save('recording.wav', torch.from_numpy(audio_data).unsqueeze(0), mic.sample_rate)
    
    print("Audio saved to recording.wav")