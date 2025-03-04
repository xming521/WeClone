# WeClone-audio 模块

WeClone-audio 是一个使用微信语音消息克隆声音的模块，使用 Llasa 模型实现高质量语音合成。
### 显存需求
- **3B 模型**: 约 16GB 显存
- **1B 模型**: 约 9GB 显存

## 1. 导出微信语音数据

### 1.1 准备工作
- 使用 [PyWxDump](https://github.com/xaoyaoo/PyWxDump) 提取微信聊天记录
- 下载软件并解密数据库
- 点击聊天备份，导出类型选择"解密文件"

### 1.2 环境配置
语音导出仅支持Windows环境

WeClone Audio使用uv作为包管理器，暂时独立于WeClone项目。请确保已安装uv。 
```bash
# 为 PyWxDump 创建 Python 环境和安装依赖
# 
cd ./WeClone-audio
uv venv .venv-wx --python=3.9
source .venv-wx/bin/activate
# 安装 wx 依赖组
uv pip install -e '.[wx]'
```

### 1.3 导出语音文件
```bash
python ./WeClone-audio/get_sample_audio.py --db-path "导出数据库路径" --MsgSvrID "导出聊天记录的MsgSvrID字段"
```

## 2. 语音合成推理

### 2.1 环境配置
```bash
# 创建并配置推理环境
uv venv .venv-xcodec --python=3.9
source .venv-xcodec/bin/activate
uv pip install -e '.[xcodec]'

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
from text_to_speech import TextToSpeech


sample_audio_text = "对，这就是我万人敬仰的太乙真人，虽然有点婴儿肥，但也掩不住我逼人的帅气。"  # 示例音频文本
sample_audio_path = os.path.join(os.path.dirname(__file__), "sample.wav")  # 示例音频路径
tts = TextToSpeech(sample_audio_path, sample_audio_text)
target_text = "晚上好啊"  # 生成目标文本
result = tts.infer(target_text)
sf.write(os.path.join(os.path.dirname(__file__), "output.wav"), result[1], result[0])  # 保存生成音频
```
   
