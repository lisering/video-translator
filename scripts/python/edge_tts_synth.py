#!/usr/bin/env python3
"""
Edge-TTS wrapper for video-translator.

借鉴 pyvideotrans 的 edge-tts 用法，使用微软云端神经 TTS 合成中文语音。
- 免费，无需 API Key
- 自带男声/女声（无需 pitch shifting，音质自然）
- RTF ~0.1x（云端合成，极快）

用法:
  python edge_tts_synth.py --text "你好世界" --voice zh-CN-XiaoxiaoNeural --output out.wav
  python edge_tts_synth.py --text "你好世界" --voice auto --output out.wav --ref original.wav

  --voice auto: 根据 --ref 参考音频自动检测男/女声
"""

import sys
import os
import asyncio
import argparse
import re

def parse_args():
    parser = argparse.ArgumentParser(description="Edge-TTS synthesis")
    parser.add_argument("--text", required=True, help="Text to synthesize")
    parser.add_argument("--voice", required=True, help="Voice name or 'auto'")
    parser.add_argument("--output", required=True, help="Output WAV file path")
    parser.add_argument("--ref", default=None, help="Reference audio for gender detection (when --voice auto)")
    parser.add_argument("--rate", default="+0%", help="Speaking rate (e.g., '+0%', '-10%')")
    return parser.parse_args()

# 男声/女声映射
MALE_VOICE = "zh-CN-YunxiNeural"        # 温暖男声
FEMALE_VOICE = "zh-CN-XiaoxiaoNeural"   # 自然女声

def detect_gender_from_audio(audio_path):
    """从音频检测说话人性别：基于基频(F0)分析"""
    import struct
    import wave
    import numpy as np

    try:
        with wave.open(audio_path, 'rb') as wf:
            sample_rate = wf.getframerate()
            channels = wf.getnchannels()
            sample_width = wf.getsampwidth()
            n_frames = wf.getnframes()
            raw = wf.readframes(n_frames)
    except Exception as e:
        print(f"[edge_tts] 读取音频失败: {e}", file=sys.stderr)
        return "female"

    if sample_width == 2:
        data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sample_width == 4:
        data = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        print(f"[edge_tts] 不支持的采样宽度: {sample_width}", file=sys.stderr)
        return "female"

    if channels > 1:
        data = data[::channels]

    max_samples = min(len(data), sample_rate * 10)
    data = data[:max_samples]

    frame_size = int(sample_rate * 0.03)
    n_frames = len(data) // frame_size
    voiced_frames = []

    for i in range(n_frames):
        frame = data[i * frame_size : (i + 1) * frame_size]
        energy = np.mean(frame ** 2)
        if energy > 0.001:
            voiced_frames.append(frame)

    if len(voiced_frames) < 5:
        print("[edge_tts] 有效语音帧太少，使用默认女声", file=sys.stderr)
        return "female"

    f0_values = []
    for frame in voiced_frames:
        min_lag = int(sample_rate / 400)
        max_lag = int(sample_rate / 50)

        frame_centered = frame - np.mean(frame)
        energy = np.sum(frame_centered ** 2)
        if energy < 1e-6:
            continue

        autocorr = np.correlate(frame_centered, frame_centered, mode='full')
        autocorr = autocorr[len(autocorr) // 2:]
        autocorr = autocorr / (energy + 1e-10)

        peak_lag = 0
        peak_val = 0
        for lag in range(min_lag, min(max_lag, len(autocorr))):
            if autocorr[lag] > peak_val and autocorr[lag] > 0.3:
                peak_val = autocorr[lag]
                peak_lag = lag

        if peak_lag > 0:
            f0 = sample_rate / peak_lag
            if 50 <= f0 <= 400:
                f0_values.append(f0)

    if len(f0_values) < 3:
        print("[edge_tts] F0 检测失败，使用默认女声", file=sys.stderr)
        return "female"

    f0_values.sort()
    median_f0 = f0_values[len(f0_values) // 2]

    gender = "male" if median_f0 < 165 else "female"
    print(f"[edge_tts] F0={median_f0:.1f}Hz → {gender}", file=sys.stderr)
    return gender


def clean_text_for_tts(text):
    """清理文本，移除 edge-tts 无法处理的字符"""
    # 移除控制字符和零宽字符
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
    # 移除零宽空格、零宽连接符等
    text = re.sub(r'[\u200b\u200c\u200d\ufeff]', '', text)
    # 移除 [[T0]] 等占位符（翻译术语占位符）
    text = re.sub(r'\[\[T\d+\]\]', '', text)
    # 移除 URL
    text = re.sub(r'https?://\S+', '', text)
    # 移除 markdown 代码块 ```...```
    text = re.sub(r'```[\s\S]*?```', '', text)
    # 移除行内代码 `...`
    text = re.sub(r'`[^`]*`', '', text)
    # 移除 markdown 标题标记 #
    text = re.sub(r'^#+\s+', '', text, flags=re.MULTILINE)
    # 移除 markdown 链接 [text](url)
    text = re.sub(r'\[([^\]]*)\]\([^\)]*\)', r'\1', text)
    # 移除 markdown 加粗/斜体标记
    text = re.sub(r'\*{1,3}([^*]+)\*{1,3}', r'\1', text)
    # 移除 HTML 标签
    text = re.sub(r'<[^>]+>', '', text)
    # 移除连续特殊符号（如 :::、---、===）
    text = re.sub(r'[:=\-]{3,}', '', text)
    # 移除纯标点行
    text = re.sub(r'^[\s。，！？．、；：""''（）【】《》…—,.!?;:\'\"()\[\]{}\-]+$', '', text, flags=re.MULTILINE)
    # 多余空白压缩为单个空格
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def split_long_text(text, max_chars=150):
    """将长文本按句号/问号/感叹号拆分为短句

    edge-tts 对超过 ~200 字符的文本可能返回 NoAudioReceived 错误。
    拆分后逐句合成，再用 ffmpeg 拼接。
    """
    if len(text) <= max_chars:
        return [text]

    # 按中文/英文标点拆分
    sentences = re.split(r'([。！？.!?;；])', text)

    # 重新组合：标点附着到前一个句子
    chunks = []
    current = ""
    for part in sentences:
        current += part
        # 如果当前句以结束标点结尾且长度足够，切分
        if current and current[-1] in '。！？.!?;；' and len(current) >= 10:
            chunks.append(current)
            current = ""
    if current.strip():
        chunks.append(current)

    # 如果某句仍然太长，按逗号再拆分
    result = []
    for chunk in chunks:
        if len(chunk) <= max_chars:
            result.append(chunk)
        else:
            sub_parts = re.split(r'([，,])', chunk)
            sub = ""
            for p in sub_parts:
                sub += p
                if len(sub) >= max_chars // 2:
                    result.append(sub)
                    sub = ""
            if sub.strip():
                result.append(sub)

    return [c for c in result if c.strip()]


def fallback_piper(text, voice_name, output_wav):
    """使用本地 Piper TTS 作为离线 fallback（优先），say 作为最终 fallback

    根据 voice_name 判断男/女声：
    - 男声 (Yunxi/Yunjian/Yunyang): Piper 女声 + asetrate 降调 0.85
    - 女声 (Xiaoxiao/Xiaoyi): Piper 女声原调
    """
    import subprocess

    is_male = "Yunxi" in voice_name or "Yunjian" in voice_name or "Yunyang" in voice_name
    # 男声 pitch multiplier: 0.65 (huayan 女声 F0≈251Hz, ×0.65→163Hz 男声范围)
    # 之前 0.85 不够 (F0→193Hz 仍是女声), 0.65 是经过 F0 分析验证的最低自然值

    pitch_mult = 0.65 if is_male else 1.0
    piper_model = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                               "models", "piper", "zh_CN-huayan-medium.onnx")
    piper_config = piper_model + ".json"
    piper_bin = os.environ.get("PIPER_BIN",
                               "/Users/john/soft/pyvideotrans/.venv/bin/piper")

    if os.path.isfile(piper_model) and os.path.isfile(piper_bin):
        try:
            length_scale = "0.95" if is_male else "1.0"

            proc = subprocess.run(
                [piper_bin, "-m", piper_model, "-c", piper_config,
                 "-f", output_wav, "--length-scale", length_scale],
                input=text.encode("utf-8"),
                capture_output=True,
                timeout=30.0
            )
            if proc.returncode == 0 and os.path.exists(output_wav) and os.path.getsize(output_wav) > 100:
                # Piper 输出 22050Hz，重采样到 24000Hz + 男声降调
                tmp = output_wav + ".tmp.wav"
                os.rename(output_wav, tmp)

                # 构建 ffmpeg 滤镜链
                filters = []
                if (pitch_mult - 1.0) > 0.001 or (1.0 - pitch_mult) > 0.001:
                    new_rate = int(22050 * pitch_mult)
                    tempo = 1.0 / pitch_mult
                    filters.append(f"asetrate={new_rate},atempo={tempo:.4f}")
                filters.append("dynaudnorm=f=150:g=15:p=0.9")

                filter_chain = ",".join(filters)
                subprocess.run(
                    ["ffmpeg", "-y", "-i", tmp, "-af", filter_chain,
                     "-ar", "24000", "-ac", "1", "-c:a", "pcm_s16le", output_wav],
                    check=True, capture_output=True
                )
                os.unlink(tmp)
                print(f"[tts] Piper fallback 成功 (male={is_male}, pitch={pitch_mult})", file=sys.stderr)
                return True
            else:
                stderr = proc.stderr.decode() if proc.stderr else ""
                print(f"[tts] Piper fallback 失败: {stderr[:200]}", file=sys.stderr)
        except subprocess.TimeoutExpired:
            print(f"[tts] Piper fallback 超时 30s", file=sys.stderr)
        except Exception as e:
            print(f"[tts] Piper fallback 异常: {e}", file=sys.stderr)
    else:
        if not os.path.isfile(piper_model):
            print(f"[tts] Piper 模型不存在: {piper_model}", file=sys.stderr)
        if not os.path.isfile(piper_bin):
            print(f"[tts] Piper 可执行文件不存在: {piper_bin}", file=sys.stderr)

    # 2. Piper 不可用，回退到 macOS say（也应用男声降调）
    say_voice = "Tingting"
    tmp_aiff = output_wav + ".tmp.aiff"
    try:
        subprocess.run(["say", "-v", say_voice, "-o", tmp_aiff, text],
                       check=True, capture_output=True, timeout=30.0)

        # 构建 ffmpeg 滤镜链（男声降调）
        filters = []
        if (pitch_mult - 1.0) > 0.001 or (1.0 - pitch_mult) > 0.001:
            new_rate = int(24000 * pitch_mult)
            tempo = 1.0 / pitch_mult
            filters.append(f"asetrate={new_rate},atempo={tempo:.4f}")
        filters.append("dynaudnorm=f=150:g=15:p=0.9")

        filter_chain = ",".join(filters)
        subprocess.run(
            ["ffmpeg", "-y", "-i", tmp_aiff, "-af", filter_chain,
             "-ar", "24000", "-ac", "1", "-c:a", "pcm_s16le", output_wav],
            check=True, capture_output=True
        )
        os.unlink(tmp_aiff)
        print(f"[tts] say fallback 成功 (male={is_male}, pitch={pitch_mult})", file=sys.stderr)
        return True
    except Exception as e:
        print(f"[tts] say fallback 也失败: {e}", file=sys.stderr)
        return False


async def synthesize_single(text, voice, output_mp3, rate="+0%"):
    """合成单段文本为 MP3，带重试和超时"""
    import edge_tts

    print(f"[edge_tts] 合成: {text[:60]}..." if len(text) > 60 else f"[edge_tts] 合成: {text}", file=sys.stderr)

    for attempt in range(3):
        try:
            communicate = edge_tts.Communicate(text, voice, rate=rate)
            # 15 秒超时，防止网络挂起
            await asyncio.wait_for(communicate.save(output_mp3), timeout=15.0)
            # 检查文件非空
            if os.path.exists(output_mp3) and os.path.getsize(output_mp3) > 100:
                return True
            print(f"[edge_tts] 合成结果为空 (attempt {attempt+1})", file=sys.stderr)
        except asyncio.TimeoutError:
            print(f"[edge_tts] 合成超时 15s (attempt {attempt+1})", file=sys.stderr)
        except Exception as e:
            print(f"[edge_tts] 合成失败 (attempt {attempt+1}): {e}", file=sys.stderr)
        if attempt < 2:
            await asyncio.sleep(1.0 * (attempt + 1))  # 递增等待

    return False


async def synthesize(text, voice, output, rate="+0%"):
    """合成语音：Piper 优先（保证声音一致），Edge-TTS 作为 fallback

    策略变更原因：
    - Edge-TTS 对部分文本随机返回 NoAudioReceived，导致部分段走 fallback
    - Edge-TTS 和 Piper 音色不同，混用导致声音不统一
    - Piper 本地合成，声音始终一致，速度更快
    """
    import subprocess
    import tempfile

    text = clean_text_for_tts(text)
    if not text:
        print("[edge_tts] 文本为空（清理后）", file=sys.stderr)
        sys.exit(1)

    # 1. 优先使用 Piper（本地，一致，快速）
    if fallback_piper(text, voice, output):
        return

    # 2. Piper 失败，尝试 Edge-TTS
    print(f"[tts] Piper 失败，尝试 Edge-TTS", file=sys.stderr)
    chunks = split_long_text(text, max_chars=150)
    tmp_mp3 = output + ".tmp.mp3"

    if len(chunks) == 1:
        ok = await synthesize_single(chunks[0], voice, tmp_mp3, rate)
        if ok:
            # 转 WAV
            subprocess.run(
                ["ffmpeg", "-y", "-i", tmp_mp3, "-ar", "24000", "-ac", "1",
                 "-c:a", "pcm_s16le", output],
                check=True, capture_output=True
            )
            if os.path.exists(tmp_mp3):
                os.unlink(tmp_mp3)
            print(f"[tts] Edge-TTS fallback 成功", file=sys.stderr)
            return
    else:
        # 长文本逐句合成
        tmp_dir = tempfile.mkdtemp()
        mp3_files = []
        for i, chunk in enumerate(chunks):
            tmp_mp3_part = os.path.join(tmp_dir, f"part_{i:04d}.mp3")
            ok = await synthesize_single(chunk, voice, tmp_mp3_part, rate)
            if ok:
                mp3_files.append(tmp_mp3_part)
            else:
                # Edge-TTS 也失败，生成静音
                subprocess.run(
                    ["ffmpeg", "-y", "-f", "lavfi", "-i", "anullsrc=r=24000:cl=mono",
                     "-t", "0.5", tmp_mp3_part],
                    capture_output=True
                )
                mp3_files.append(tmp_mp3_part)

        if mp3_files:
            concat_list = os.path.join(tmp_dir, "concat.txt")
            with open(concat_list, 'w') as f:
                for mp3 in mp3_files:
                    f.write(f"file '{mp3}'\n")
            subprocess.run(
                ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", concat_list,
                 "-c", "copy", tmp_mp3],
                capture_output=True
            )
            import shutil
            shutil.rmtree(tmp_dir, ignore_errors=True)

            subprocess.run(
                ["ffmpeg", "-y", "-i", tmp_mp3, "-ar", "24000", "-ac", "1",
                 "-c:a", "pcm_s16le", output],
                check=True, capture_output=True
            )
            if os.path.exists(tmp_mp3):
                os.unlink(tmp_mp3)
            print(f"[tts] Edge-TTS fallback 成功 (长文本)", file=sys.stderr)
            return

    # 3. 全部失败，生成静音
    print(f"[tts] 所有引擎失败，生成静音", file=sys.stderr)
    subprocess.run(["ffmpeg", "-y", "-f", "lavfi", "-i", "anullsrc=r=24000:cl=mono",
                    "-t", "1.0", output], capture_output=True)


def main():
    args = parse_args()

    voice = args.voice
    if voice == "auto":
        if args.ref and os.path.isfile(args.ref):
            gender = detect_gender_from_audio(args.ref)
            voice = MALE_VOICE if gender == "male" else FEMALE_VOICE
            print(f"[edge_tts] 自动选择声音: {voice} ({gender})", file=sys.stderr)
        else:
            voice = FEMALE_VOICE
            print(f"[edge_tts] 无参考音频，使用默认女声: {voice}", file=sys.stderr)

    asyncio.run(synthesize(args.text, voice, args.output, args.rate))
    print(f"[edge_tts] 合成完成: {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
