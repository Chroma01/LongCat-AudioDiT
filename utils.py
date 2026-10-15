import re
import librosa
import torch

def get_dbfs(y, max_possible=1):
    max_sample = torch.max(torch.abs(y))
    return 20 * torch.log10(max_sample / max_possible + 1e-8)

def normalize_volume(y, target_dbfs=-6.0):
    current_dbfs = get_dbfs(y)
    
    gain_db = target_dbfs - current_dbfs
    gain = 10 ** (gain_db / 20)
    
    y_norm = y * gain
    y_norm = torch.clip(y_norm, -1.0, 1.0)

    return y_norm

def load_audio(wavpath, sr):
    audio, _ = librosa.load(wavpath, sr=sr, mono=True)
    audio = torch.from_numpy(audio).unsqueeze(0)
    audio = normalize_volume(audio, -6)
    return audio

def normalize_text(text):
    text = text.lower()
    text = re.sub(r'["“”‘’]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text

def approx_duration_from_text(text, max_duration=30.0):
    EN_DUR_PER_CHAR = 0.082
    ZH_DUR_PER_CHAR = 0.21
    text = re.sub(r"\s+", "", text)
    num_zh = num_en = num_other = 0
    for c in text:
        if "\u4e00" <= c <= "\u9fff":
            num_zh += 1
        elif c.isalpha():
            num_en += 1
        else:
            num_other += 1
    if num_zh > num_en:
        num_zh += num_other
    else:
        num_en += num_other
    return min(max_duration, num_zh * ZH_DUR_PER_CHAR + num_en * EN_DUR_PER_CHAR)