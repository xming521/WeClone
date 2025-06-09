import os

import soundfile as sf
import torch
from SparkTTS import SparkTTS

model = SparkTTS("weclone-audio/pretrained_models/Spark-TTS-0.5B", "cuda")


with torch.no_grad():
    wav = model.inference(
        text="晚上好啊,小可爱们，该睡觉了哦",
        prompt_speech_path=os.path.join(os.path.dirname(__file__), "sample.wav"),
        prompt_text="对，这就是我万人敬仰的太乙真人，虽然有点婴儿肥，但也掩不住我逼人的帅气。",
    )
    sf.write(os.path.join(os.path.dirname(__file__), "output.wav"), wav, samplerate=16000)
    print("生成成功！")
