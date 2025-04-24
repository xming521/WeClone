# WeClone-audio 模块

WeClone-audio 是一个使用微信语音消息克隆声音的模块，使用模型实现高质量语音合成。
### 显存需求
**Spark-TTS** 推荐
- **0.5B 模型**: 约 4GB 显存

**Llasa** （已弃用）
- **3B 模型**: 约 16GB 显存
- **1B 模型**: 约 9GB 显存  




## 1. 导出微信语音数据

### 1.1 准备工作
- 使用 [PyWxDump](https://github.com/xaoyaoo/PyWxDump) 提取微信聊天记录
- 下载软件并解密数据库
- 点击聊天备份，导出类型选择"解密文件"

### 1.2 环境配置
语音导出仅支持Windows环境
WeClone Audio使用uv作为包管理器。 
```bash
# 为 PyWxDump 创建 Python 环境和安装依赖
# 
uv venv .venv-wx --python=3.10
.venv-wx\Scripts\activate
uv pip install pywxdump
```

### 1.3 导出语音文件
```bash
python weclone-audio/src/get_sample_audio.py --db-path "导出数据库路径" --MsgSvrID "导出聊天记录的MsgSvrID字段"
```

## 2. 语音合成推理
### Spark-TTS模型

**环境安装**
可不创建新环境，直接安装`sparktts`依赖组到WeClone共主环境

```bash
uv venv .venv-sparktts --python=3.10
source .venv-sparktts/bin/activate
uv pip install --group sparktts -e .

git clone https://github.com/SparkAudio/Spark-TTS.git weclone-audio/src/Spark-TTS
```


**模型下载**

通过python下载:
```python
from huggingface_hub import snapshot_download

# 假设此 Python 代码在 weclone-audio 目录下运行 模型将下载到 weclone-audio/pretrained_models/Spark-TTS-0.5B
snapshot_download("SparkAudio/Spark-TTS-0.5B", local_dir="pretrained_models/Spark-TTS-0.5B")
```

或通过git下载:
```bash
# 假设当前在 weclone-audio 目录
mkdir -p pretrained_models

# Make sure you have git-lfs installed (https://git-lfs.com)
git lfs install
git clone https://huggingface.co/SparkAudio/Spark-TTS-0.5B pretrained_models/Spark-TTS-0.5B
```
使用代码推理
```python
import os
import SparkTTS  
import soundfile as sf
import torch

from SparkTTS import SparkTTS 

# 假设此 Python 代码在 weclone-audio 目录下运行
# 模型路径相对于当前目录
model_path = "pretrained_models/Spark-TTS-0.5B"
sample_audio = "sample.wav"
output_audio = "output.wav"

model = SparkTTS(model_path, "cuda")

with torch.no_grad():
    wav = model.inference(
        text="晚上好啊,小可爱们，该睡觉了哦",
        prompt_speech_path=sample_audio, # 使用相对路径
        prompt_text="对，这就是我万人敬仰的太乙真人，虽然有点婴儿肥，但也掩不住我逼人的帅气。",
    )
    sf.write(output_audio, wav, samplerate=16000) # 使用相对路径
```
### Llasa模型 （已弃用）
### 2.1 环境配置
```bash
# 创建并配置推理环境 
## 可不创建新环境，与LLaMA-Factory环境共用
uv venv .venv-xcodec --python=3.9
source .venv-xcodec/bin/activate
uv pip install --group xcodec -e .
# 退出环境
deactivate

# 系统依赖安装（如果需要）
sudo apt install python3-dev 
sudo apt install build-essential
```

### 2.2 使用代码推理
如果遇到问题，请尝试将参考音频转换为WAV或MP3格式，将其裁剪至15秒以内，并缩短提示文本。
```python
import os
import soundfile as sf
# 假设 text_to_speech.py 位于 src/ 或其他可导入的位置
from text_to_speech import TextToSpeech


sample_audio_text = "对，这就是我万人敬仰的太乙真人，虽然有点婴儿肥，但也掩不住我逼人的帅气。"  # 示例音频文本
# 假设此 Python 代码在 weclone-audio 目录下运行
# 示例音频路径相对于当前目录
sample_audio_path = "sample.wav"
output_audio = "output.wav"


tts = TextToSpeech(sample_audio_path, sample_audio_text)
target_text = "晚上好啊"  # 生成目标文本
result = tts.infer(target_text)
sf.write(output_audio, result[1], result[0])  # 使用相对路径
```
   
