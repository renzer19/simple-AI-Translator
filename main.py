import gradio as gr
import assemblyai as aai
from translate import Translator
from elevenlabs import VoiceSettings
from elevenlabs.client import ElevenLabs
import uuid
from pathlib import Path

def voice_to_voice(audio_file):
    #transcript audio
    transcription_respone = audio_transcrip(audio_file)
    if transcription_respone.status == aai.TranscriptStatus.error:
        raise gr.Error(transcription_respone.error)
    else:
        text = transcription_respone.text
        text_translate(text)
        
    es_translation, tr_translation, ja_translation = text_translate(text)
    
    es_audi_path = text_to_speech(es_translation)
    tr_audi_path = text_to_speech(tr_translation)
    ja_audi_path = text_to_speech(ja_translation)
    
    es_path = Path(es_audi_path)
    tr_path = Path(tr_audi_path)
    ja_path = Path(ja_audi_path)
    
    return es_path, tr_path,ja_path
    
def audio_transcrip(audio_file):
    aai.settings.api_key = ""
    transcribber = aai.Transcriber()
    transcrip =  transcribber.transcribe(audio_file)
    return transcrip
    
def text_translate(text):
    translator_es = Translator(from_lang="en", to_lang="es")
    es_text = translator_es.translate(text)
    
    translator_tr = Translator(from_lang="en", to_lang="tr")
    tr_text = translator_tr.translate(text)
    
    translator_ja = Translator(from_lang="en", to_lang="ja")
    ja_text = translator_ja.translate(text)
    
    return es_text, tr_text, ja_text

def text_to_speech(text):
    client = ElevenLabs(
        api_key="",
    )
    response = client.text_to_speech.convert(
        voice_id="pNInz6obpgDQGcFmaJgB", 
        optimize_streaming_latency="0",
        output_format="mp3_22050_32",
        text=text,
        model_id="eleven_multilingual_v2", 
        voice_settings=VoiceSettings(
            stability=0.5,
            similarity_boost=0.8,
            style=0.5,
            use_speaker_boost=True,
        ),
    )
    save_file_path = f"{uuid.uuid4()}.mp3"

    with open(save_file_path, "wb") as f:
        for chunk in response:
            if chunk:
                f.write(chunk)
    print(f"{save_file_path}: A new audio file was saved successfully!")
    return save_file_path

audio_input = gr.Audio(
    sources = ["microphone"],
    type = "filepath"
)

demo = gr.Interface(
    fn = voice_to_voice,
    inputs = audio_input,
    outputs = [gr.Audio(label="Spanish"), gr.Audio(label="Turkish"), gr.Audio(label="Japanese")]
)

if __name__ == '__main__':
    demo.launch(share="true")
