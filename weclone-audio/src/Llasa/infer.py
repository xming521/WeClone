import os

import soundfile as sf
from text_to_speech import TextToSpeech

sample_audio_text = "对，这就是我万人敬仰的太乙真人，虽然有点婴儿肥，但也掩不住我逼人的帅气。"  # 示例音频文本
sample_audio_path = os.path.join(os.path.dirname(__file__), "sample.wav")  # 示例音频路径
tts = TextToSpeech(sample_audio_path, sample_audio_text)
target_text = "晚上好啊"  # 生成目标文本
result = tts.infer(target_text)
sf.write(os.path.join(os.path.dirname(__file__), "output.wav"), result[1], result[0])  # 保存生成音频
