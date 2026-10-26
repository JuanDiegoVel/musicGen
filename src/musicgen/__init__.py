
import torch
from scipy.io.wavfile import write
from transformers import AutoProcessor, MusicgenForConditionalGeneration

MODEL_ID = "facebook/musicgen-small"
DEVICE = "cpu"

PROMPT = "Create a smooth, atmospheric melodic trap song with a late-night Toronto vibe, blending emotional R&B melodies with modern hip-hop production. Use warm ambient synths, moody piano chords, deep 808 bass, crisp hi-hats, subtle vocal samples, and spacious reverb. Feature melodic rap verses that switch between confident, rhythmic flows and vulnerable, emotional singing. The lyrics should be in English, exploring fame, money, complicated relationships, loyalty, heartbreak, ambition, trust issues, and late-night thoughts. Include a catchy, unforgettable chorus, intimate vocals, tasteful autotune, layered harmonies, and natural flow switches. The mood should feel luxurious, introspective, seductive, and slightly melancholic, like driving through Toronto at 2 AM. Use polished, minimalist production with a deep bassline and a cinematic atmosphere. Tempo: 95–105 BPM. Make it sound modern, smooth, catchy, and emotionally captivating."

print("Cargando MusicGen...")

processor = AutoProcessor.from_pretrained(MODEL_ID)
model = MusicgenForConditionalGeneration.from_pretrained(MODEL_ID)

model = model.to(DEVICE)
model.eval()

print("Preparando prompt...")

inputs = processor(
    text=[PROMPT],
    padding=True,
    return_tensors="pt"
).to(DEVICE)

print("Generando música...")

with torch.inference_mode():
    audio = model.generate(
        **inputs,
        do_sample=True,
        guidance_scale=3.0,
        max_new_tokens=1000
    )

print("Guardando archivo de audio...")

sample_rate = model.config.audio_encoder.sampling_rate
waveform = audio[0, 0].cpu().float().numpy()

# Convertir el audio a formato WAV de 16 bits
waveform = waveform.clip(-1.0, 1.0)
waveform = (waveform * 32767).astype("int16")

write("musica.wav", sample_rate, waveform)

print("¡Música generada! Archivo guardado como musica.wav")

