# Whisper-sk-hu
Whisper base model pretrained for the Slovak and Hungarian languages

## Deploy

1. Download the model from https://huggingface.co/Felagund
2. Download requirements.txt

'''
pip install -r requirements.txt
'''
4. Record audio of hungarian or slovak speech
5. Update variables WHISPER_MODEL_PATH, LANGUAGE, AUDIO_FILE_PATH
6. Run example.py
'''
python example.py
'''
