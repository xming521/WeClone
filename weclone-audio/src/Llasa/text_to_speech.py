import os

import soundfile as sf
import torch
import torchaudio
from transformers import AutoModelForCausalLM, AutoTokenizer
from xcodec2.modeling_xcodec2 import XCodec2Model


class TextToSpeech:
    def __init__(self, sample_audio_path, sample_audio_text):
        self.sample_audio_text = sample_audio_text
        # 初始化模型
        llasa_3b = "HKUSTAudio/Llasa-3B"
        xcodec2 = "HKUSTAudio/xcodec2"

        self.tokenizer = AutoTokenizer.from_pretrained(llasa_3b)
        self.llasa_3b_model = AutoModelForCausalLM.from_pretrained(
            llasa_3b,
            trust_remote_code=True,
            device_map="auto",
        )
        self.llasa_3b_model.eval()

        self.xcodec_model = XCodec2Model.from_pretrained(xcodec2)
        self.xcodec_model.eval().cuda()

        # 处理音频
        waveform, sample_rate = torchaudio.load(sample_audio_path)
        if len(waveform[0]) / sample_rate > 15:
            print("已将音频裁剪至前15秒。")
            waveform = waveform[:, : sample_rate * 15]

        # 检查音频是否为立体声
        if waveform.size(0) > 1:
            waveform_mono = torch.mean(waveform, dim=0, keepdim=True)
        else:
            waveform_mono = waveform

        self.prompt_wav = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(waveform_mono)

        # Encode the prompt wav
        vq_code_prompt = self.xcodec_model.encode_code(input_waveform=self.prompt_wav)
        vq_code_prompt = vq_code_prompt[0, 0, :]
        self.speech_ids_prefix = self.ids_to_speech_tokens(vq_code_prompt)
        self.speech_end_id = self.tokenizer.convert_tokens_to_ids("<|SPEECH_GENERATION_END|>")

    def ids_to_speech_tokens(self, speech_ids):
        speech_tokens_str = []
        for speech_id in speech_ids:
            speech_tokens_str.append(f"<|s_{speech_id}|>")
        return speech_tokens_str

    def extract_speech_ids(self, speech_tokens_str):
        speech_ids = []
        for token_str in speech_tokens_str:
            if token_str.startswith("<|s_") and token_str.endswith("|>"):
                num_str = token_str[4:-2]
                num = int(num_str)
                speech_ids.append(num)
            else:
                print(f"Unexpected token: {token_str}")
        return speech_ids

    @torch.inference_mode()
    def infer(self, target_text):
        if len(target_text) == 0:
            return None
        elif len(target_text) > 300:
            print("文本过长，请保持在300字符以内。")
            target_text = target_text[:300]

        input_text = self.sample_audio_text + " " + target_text

        formatted_text = f"<|TEXT_UNDERSTANDING_START|>{input_text}<|TEXT_UNDERSTANDING_END|>"

        chat = [
            {
                "role": "user",
                "content": "Convert the text to speech:" + formatted_text,
            },
            {
                "role": "assistant",
                "content": "<|SPEECH_GENERATION_START|>" + "".join(self.speech_ids_prefix),
            },
        ]

        input_ids = self.tokenizer.apply_chat_template(
            chat, tokenize=True, return_tensors="pt", continue_final_message=True
        )
        input_ids = input_ids.to("cuda")

        outputs = self.llasa_3b_model.generate(
            input_ids,
            max_length=2048,
            eos_token_id=self.speech_end_id,
            do_sample=True,
            top_p=1,
            temperature=0.8,
        )
        generated_ids = outputs[0][input_ids.shape[1] - len(self.speech_ids_prefix) : -1]

        speech_tokens = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        speech_tokens = self.extract_speech_ids(speech_tokens)
        speech_tokens = torch.tensor(speech_tokens).cuda().unsqueeze(0).unsqueeze(0)

        gen_wav = self.xcodec_model.decode_code(speech_tokens)
        gen_wav = gen_wav[:, :, self.prompt_wav.shape[1] :]

        return (16000, gen_wav[0, 0, :].cpu().numpy())


if __name__ == "__main__":
    # 如果遇到问题，请尝试将参考音频转换为WAV或MP3格式，将其裁剪至15秒以内，并缩短提示文本。
    sample_audio_text = "对，这就是我万人敬仰的太乙真人，虽然有点婴儿肥，但也掩不住我逼人的帅气。"
    sample_audio_path = os.path.join(os.path.dirname(__file__), "sample.wav")

    tts = TextToSpeech(sample_audio_path, sample_audio_text)
    target_text = "晚上好啊，吃了吗您"
    result = tts.infer(target_text)
    sf.write(os.path.join(os.path.dirname(__file__), "output.wav"), result[1], result[0])
    target_text = "我是老北京正黄旗！"
    result = tts.infer(target_text)
    sf.write(os.path.join(os.path.dirname(__file__), "output1.wav"), result[1], result[0])
