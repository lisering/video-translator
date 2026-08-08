#!/usr/bin/env python3
"""
Qwen3-TTS Python Server for video-translator.

完全照搬 pyvideotrans (videotrans/process/qwen_tts.py) 的调用方式:
- 使用 Qwen3TTSModel.from_pretrained 加载 Base 模型
- 使用 generate_voice_clone 进行声音克隆
- ref_text 为空时使用 x_vector_only_mode=True

P1 增强: 多说话人缓存 + 预热机制 (借鉴 dots.tts CAM++ speaker encoder)
- 多说话人支持: speaker_cache 字典按 speaker_id 缓存 voice_clone_prompt
- 预热机制: prewarm_speaker action 提前提取 voice clone prompt
- speaker_id 字段: TTS 请求可直接引用已缓存的说话人
- list_speakers / clear_speakers: 管理缓存

P7 增强: 可选 ERes2NetV2 声纹提取 (借鉴 GPT-SoVITS v2Pro)
- --speaker-model 参数加载 ERes2NetV2 模型
- 支持 action="extract_speaker_embedding" 请求
- 从参考音频提取 20480 维声纹向量 (16kHz, 80-dim Fbank)
- 声纹向量 L2 归一化后返回给 Rust 端

通信协议 (stdin/stdout JSON, 兼容 PersistentSubprocessCloneEngine):
  预热请求:  {"action":"prewarm_speaker", "speaker_id":"spk1", "voice":"/path/ref.wav", "ref_text":"..."}
  预热响应:  {"status":"ok", "speaker_id":"spk1", "elapsed_secs":N}

  TTS 请求:  {"text":"...", "voice":"/path/ref.wav", "output":"/path/out.wav", "ref_text":"...", "speaker_id":"spk1"}
  TTS 响应:  {"status":"ok", "output":"...", "duration_secs":N, "elapsed_secs":N}

  列表请求:  {"action":"list_speakers"}
  列表响应:  {"status":"ok", "speakers":[{"speaker_id":"spk1","ref_audio":"/path/ref.wav"}, ...]}

  清除请求:  {"action":"clear_speakers"}
  清除响应:  {"status":"ok", "cleared":N}

  声纹请求:  {"action":"extract_speaker_embedding", "ref_audio":"/path/ref.wav", "use_eres2net":true}
  声纹响应:  {"status":"ok", "embedding":[...], "dim":20480, "elapsed_secs":N}

用法:
  python tts_server.py --model /path/to/qwen3-tts --device cpu
  python tts_server.py --model /path/to/qwen3-tts --device cpu --speaker-model /path/to/eres2net
"""

import sys
import os
import json
import time
import traceback

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# ─── stdout 保护策略 ──────────────────────────────────────
# Hugging Face transformers / qwen_tts 等库会向 stdout 打印警告信息
# (如 "Setting pad_token_id...", "Warning: flash-attn is not installed...")
# 这会破坏 stdin/stdout JSON 行协议。
#
# 解决方案: 程序启动时将 stdout 重定向到 stderr，
# 只在 send_json() 时临时恢复真实 stdout 写入 JSON 响应。

_real_stdout = sys.stdout  # 保存真实 stdout 句柄


def log(msg):
    """日志输出到 stderr"""
    print(f"[tts_server] {msg}", file=sys.stderr, flush=True)


def send_json(obj):
    """向真实 stdout 写入一行 JSON 响应（唯一允许写 stdout 的地方）"""
    line = json.dumps(obj)
    _real_stdout.write(line + "\n")
    _real_stdout.flush()


def redirect_stdout_to_stderr():
    """将 sys.stdout 重定向到 stderr，防止库输出污染 JSON 协议"""
    sys.stdout = sys.stderr


def main():
    # 解析命令行参数
    model_path = None
    device = "cpu"
    speaker_model_path = None  # P7: ERes2NetV2 模型路径

    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == "--model" and i + 1 < len(args):
            model_path = args[i + 1]
            i += 2
        elif args[i] == "--device" and i + 1 < len(args):
            device = args[i + 1]
            i += 2
        elif args[i] == "--speaker-model" and i + 1 < len(args):
            speaker_model_path = args[i + 1]
            i += 2
        else:
            i += 1

    if not model_path:
        log("ERROR: --model 参数是必须的")
        sys.exit(1)

    # 立即将 stdout 重定向到 stderr，防止后续 import 和模型操作污染 JSON 协议
    redirect_stdout_to_stderr()

    # 导入依赖（所有输出都会到 stderr）
    import torch
    import soundfile as sf
    from qwen_tts import Qwen3TTSModel

    # 设备和 dtype 选择 (完全照搬 pyvideotrans)
    if device == "cuda":
        device_map = "cuda:0"
        dtype = torch.float16 if not torch.cuda.is_bf16_supported() else torch.bfloat16
    elif device in ("metal", "mps"):
        device_map = "mps"
        dtype = torch.float32
    else:
        device_map = "cpu"
        dtype = torch.float32

    log(f"加载模型: {model_path}, device={device_map}, dtype={dtype}")

    model = Qwen3TTSModel.from_pretrained(
        model_path,
        device_map=device_map,
        dtype=dtype,
    )
    log("模型加载完成，等待请求...")

    # ─── P7: ERes2NetV2 声纹模型 (可选) ──────────────────────
    # 借鉴 GPT-SoVITS v2Pro: 使用 ERes2NetV2 提取 20480 维声纹向量
    # 增强说话人身份保持能力 (MelStyleEncoder 捕捉频谱风格,
    # ERes2NetV2 捕捉说话人身份, 两者互补)
    speaker_model = None
    if speaker_model_path:
        try:
            log(f"加载 ERes2NetV2 声纹模型: {speaker_model_path}")
            # 尝试加载 ERes2NetV2 (如果依赖可用)
            # 方式1: 从 GPT-SoVITS 目录加载
            import importlib.util
            spec = importlib.util.spec_from_file_location(
                "ERes2NetV2",
                os.path.join(speaker_model_path, "ERes2NetV2.py")
            )
            if spec and spec.loader:
                eres2net_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(eres2net_module)
                ERes2NetV2 = getattr(eres2net_module, "ERes2NetV2", None)
                if ERes2NetV2:
                    speaker_model = ERes2NetV2(baseWidth=24, scale=4, expansion=4)
                    # 加载预训练权重
                    import torch
                    ckpt_path = os.path.join(speaker_model_path, "pretrained.pt")
                    if os.path.isfile(ckpt_path):
                        speaker_model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
                    speaker_model.eval()
                    log("ERes2NetV2 声纹模型加载成功")
                else:
                    log("警告: ERes2NetV2.py 中未找到 ERes2NetV2 类")
            else:
                log(f"警告: 无法加载 ERes2NetV2.py from {speaker_model_path}")
        except Exception as e:
            log(f"警告: ERes2NetV2 加载失败 (声纹增强将不可用): {e}")
            speaker_model = None

    # Fbank 特征提取函数 (ERes2NetV2 输入预处理)
    def extract_fbank(wav_path, target_sr=16000):
        """从音频文件提取 80 维 Fbank 特征 (25ms 窗, 10ms 步长)"""
        import torch
        import torchaudio
        # 加载音频
        wav, sr = torchaudio.load(wav_path)
        # 转单声道
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        # 重采样到 16kHz
        if sr != target_sr:
            resampler = torchaudio.transforms.Resample(sr, target_sr)
            wav = resampler(wav)
        # 提取 Fbank
        fbank = torchaudio.compliance.kaldi.fbank(
            wav,
            num_mel_bins=80,
            frame_length=25,
            frame_shift=10,
            sample_frequency=target_sr,
        )
        # CMVN (减均值)
        fbank = fbank - fbank.mean(dim=0, keepdim=True)
        return fbank.unsqueeze(0)  # [1, T, 80]

    # ─── P1: 多说话人缓存 ──────────────────────────────────
    # 借鉴 dots.tts CAM++ speaker encoder 的独立缓存思路:
    # 将 voice_clone_prompt 按说话人 ID 缓存，支持多说话人视频。
    # 每个 speaker_id 对应一个 (ref_audio_path, ref_text, prompt) 三元组。
    # 当 TTS 请求携带 speaker_id 时，直接从缓存取 prompt，跳过 create_voice_clone_prompt。
    # 当 TTS 请求未携带 speaker_id 但 voice 路径匹配缓存时，也复用。
    speaker_cache = {}  # {speaker_id: {"ref_audio": path, "ref_text": text, "prompt": prompt}}

    def get_or_create_prompt(ref_audio, ref_text):
        """获取或创建 voice clone prompt，支持 speaker_id 缓存和路径匹配"""
        # 1. 尝试按路径匹配已缓存的说话人
        for sid, entry in speaker_cache.items():
            if entry["ref_audio"] == ref_audio:
                log(f"路径匹配缓存说话人: {sid}")
                return entry["prompt"]

        # 2. 未命中缓存，创建新 prompt
        prompt = model.create_voice_clone_prompt(
            ref_audio=ref_audio,
            ref_text=ref_text if ref_text else None,
            x_vector_only_mode=(not ref_text),
        )
        log(f"创建新 voice clone prompt (x_vector_only={not ref_text})")
        return prompt

    def prewarm_speaker(speaker_id, ref_audio, ref_text):
        """预热情说话人: 提前提取 voice clone prompt 并缓存"""
        t_start = time.time()
        prompt = model.create_voice_clone_prompt(
            ref_audio=ref_audio,
            ref_text=ref_text if ref_text else None,
            x_vector_only_mode=(not ref_text),
        )
        speaker_cache[speaker_id] = {
            "ref_audio": ref_audio,
            "ref_text": ref_text,
            "prompt": prompt,
        }
        elapsed = time.time() - t_start
        log(f"预热情说话人 '{speaker_id}': ref_audio={ref_audio}, "
            f"x_vector_only={not ref_text}, {elapsed:.3f}s")
        return elapsed

    # 主循环: 从 stdin 读取 JSON 请求, 向 stdout 写 JSON 响应
    # 注意: sys.stdin 不受 stdout 重定向影响
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        try:
            req = json.loads(line)
        except json.JSONDecodeError as e:
            log(f"JSON 解析失败: {e}")
            continue

        action = req.get("action", "")

        # ─── P1: 预热情说话人 ────────────────────────────────
        if action == "prewarm_speaker":
            speaker_id = req.get("speaker_id", "")
            ref_audio = req.get("voice") or req.get("ref_audio") or ""
            ref_text = (req.get("ref_text") or "").strip()

            if not speaker_id:
                send_json({"status": "error", "error": "missing speaker_id"})
                continue
            if not ref_audio or not os.path.isfile(ref_audio):
                send_json({"status": "error", "error": f"invalid ref_audio: {ref_audio}"})
                continue

            try:
                elapsed = prewarm_speaker(speaker_id, ref_audio, ref_text)
                send_json({
                    "status": "ok",
                    "speaker_id": speaker_id,
                    "elapsed_secs": round(elapsed, 3),
                })
            except Exception as e:
                log(f"预热情失败: {e}\n{traceback.format_exc()}")
                send_json({"status": "error", "error": str(e)})
            continue

        # ─── P1: 列出已缓存说话人 ────────────────────────────
        if action == "list_speakers":
            speakers = [
                {"speaker_id": sid, "ref_audio": entry["ref_audio"]}
                for sid, entry in speaker_cache.items()
            ]
            send_json({"status": "ok", "speakers": speakers})
            continue

        # ─── P1: 清除说话人缓存 ──────────────────────────────
        if action == "clear_speakers":
            count = len(speaker_cache)
            speaker_cache.clear()
            log(f"清除 {count} 个说话人缓存")
            send_json({"status": "ok", "cleared": count})
            continue

        # ─── P3: 批量合成 ────────────────────────────────────
        # 借鉴 dots.tts OnlineBatcher: 一次请求合成多段文本，
        # 减少 Rust↔Python 往返开销。同一 speaker_id 共享 prompt。
        if action == "batch_synthesize":
            items = req.get("items", [])
            speaker_id = req.get("speaker_id", "")
            ref_audio = req.get("voice") or req.get("ref_audio") or ""
            ref_text = (req.get("ref_text") or "").strip()

            if not items:
                send_json({"status": "error", "error": "empty items list"})
                continue

            t_start = time.time()
            log(f"批量合成: {len(items)} items, speaker_id={speaker_id or '(none)'}")

            try:
                # 获取或创建 prompt（复用 P1 缓存逻辑）
                if speaker_id and speaker_id in speaker_cache:
                    prompt = speaker_cache[speaker_id]["prompt"]
                    log(f"批量合成: 复用缓存说话人 {speaker_id}")
                elif speaker_id and ref_audio and os.path.isfile(ref_audio):
                    elapsed_pw = prewarm_speaker(speaker_id, ref_audio, ref_text)
                    prompt = speaker_cache[speaker_id]["prompt"]
                    log(f"批量合成: 自动预热说话人 {speaker_id}: {elapsed_pw:.3f}s")
                elif ref_audio and os.path.isfile(ref_audio):
                    prompt = get_or_create_prompt(ref_audio, ref_text)
                else:
                    send_json({"status": "error",
                               "error": "no valid speaker_id or ref_audio for batch_synthesize"})
                    continue

                results = []
                for i, item in enumerate(items):
                    item_text = (item.get("text") or "").strip()
                    item_output = item.get("output") or ""

                    if not item_text or not item_output:
                        results.append({"index": i, "status": "error",
                                        "error": "empty text or output"})
                        continue

                    try:
                        wavs, sr = model.generate_voice_clone(
                            text=item_text,
                            language="Auto",
                            voice_clone_prompt=prompt,
                            do_sample=False,
                        )
                        wav_data = wavs[0]
                        sf.write(item_output, wav_data, sr)

                        duration_secs = len(wav_data) / sr
                        results.append({
                            "index": i,
                            "status": "ok",
                            "output": item_output,
                            "duration_secs": round(duration_secs, 2),
                        })
                    except Exception as e:
                        log(f"批量合成 item {i} 失败: {e}")
                        results.append({"index": i, "status": "error", "error": str(e)})

                elapsed = time.time() - t_start
                ok_count = sum(1 for r in results if r.get("status") == "ok")
                log(f"批量合成完成: {ok_count}/{len(items)} 成功, {elapsed:.1f}s")
                send_json({
                    "status": "ok",
                    "results": results,
                    "total": len(items),
                    "success": ok_count,
                    "elapsed_secs": round(elapsed, 2),
                })
            except Exception as e:
                log(f"批量合成失败: {e}\n{traceback.format_exc()}")
                send_json({"status": "error", "error": str(e)})
            continue

        # ─── P7: 声纹提取请求处理 ─────────────────────────────
        if action == "extract_speaker_embedding":
            ref_audio = req.get("ref_audio", "")
            use_eres2net = req.get("use_eres2net", True)
            if not ref_audio or not os.path.isfile(ref_audio):
                send_json({"status": "error", "error": f"invalid ref_audio: {ref_audio}"})
                continue
            if speaker_model is None:
                send_json({"status": "error", "error": "speaker model not loaded"})
                continue
            try:
                t_start = time.time()
                import torch
                fbank = extract_fbank(ref_audio)
                with torch.no_grad():
                    embedding = speaker_model(fbank)
                # L2 归一化
                embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)
                embedding_list = embedding.squeeze(0).cpu().tolist()
                elapsed = time.time() - t_start
                log(f"声纹提取成功: dim={len(embedding_list)}, {elapsed:.3f}s")
                send_json({
                    "status": "ok",
                    "embedding": embedding_list,
                    "dim": len(embedding_list),
                    "elapsed_secs": round(elapsed, 3),
                })
            except Exception as e:
                log(f"声纹提取失败: {e}\n{traceback.format_exc()}")
                send_json({"status": "error", "error": str(e)})
            continue

        # ─── TTS 合成请求处理 ─────────────────────────────────
        text = (req.get("text") or "").strip()
        ref_audio = req.get("voice") or req.get("ref_audio") or ""
        output = req.get("output") or ""
        ref_text = (req.get("ref_text") or "").strip()
        speaker_id = req.get("speaker_id") or ""  # P1: 可选说话人 ID

        if not text:
            log("文本为空，跳过")
            send_json({"status": "error", "error": "empty text"})
            continue

        if not output:
            log("未指定输出路径")
            send_json({"status": "error", "error": "no output path"})
            continue

        # speaker_id 优先；如果未提供 speaker_id，则必须有 voice 路径
        if not speaker_id and (not ref_audio or not os.path.isfile(ref_audio)):
            log(f"参考音频无效: {ref_audio}")
            send_json({"status": "error", "error": f"invalid ref_audio: {ref_audio}"})
            continue

        t_start = time.time()
        log(f"请求: text={text[:40]}... speaker_id={speaker_id or '(none)'} "
            f"ref_audio={ref_audio or '(cached)'} ref_text={'(none)' if not ref_text else ref_text[:20] + '...'}")

        try:
            # P1: 按说话人 ID 获取缓存的 prompt
            if speaker_id and speaker_id in speaker_cache:
                prompt = speaker_cache[speaker_id]["prompt"]
                log(f"复用缓存说话人: {speaker_id}")
            elif speaker_id:
                # speaker_id 提供但未预热：如果有 voice 路径则创建并缓存
                if not ref_audio or not os.path.isfile(ref_audio):
                    send_json({"status": "error",
                               "error": f"speaker_id '{speaker_id}' not prewarmed and no valid ref_audio"})
                    continue
                elapsed_pw = prewarm_speaker(speaker_id, ref_audio, ref_text)
                prompt = speaker_cache[speaker_id]["prompt"]
                log(f"自动预热说话人 {speaker_id}: {elapsed_pw:.3f}s")
            else:
                # 无 speaker_id：按路径匹配缓存或创建新 prompt
                prompt = get_or_create_prompt(ref_audio, ref_text)

            # 生成语音 (完全照搬 pyvideotrans)
            # 关键修复: 设置 do_sample=False 消除随机采样，确保声音一致性
            # 默认 do_sample=True + temperature=0.9 会导致每次生成不同的声音特征
            # (有时女声，有时男声)，设置为 False 使用贪心解码确保声音一致
            wavs, sr = model.generate_voice_clone(
                text=text,
                language="Auto",
                voice_clone_prompt=prompt,
                do_sample=False,
            )

            # 写入 WAV (照搬 pyvideotrans: sf.write)
            wav_data = wavs[0]
            sf.write(output, wav_data, sr)

            duration_secs = len(wav_data) / sr
            elapsed = time.time() - t_start

            log(f"成功: {duration_secs:.1f}s 音频, {elapsed:.1f}s 生成, RTF={elapsed / duration_secs:.2f}x")

            send_json({
                "status": "ok",
                "output": output,
                "duration_secs": round(duration_secs, 2),
                "elapsed_secs": round(elapsed, 2),
                "sample_rate": sr,
            })

        except Exception as e:
            tb = traceback.format_exc()
            log(f"生成失败: {e}\n{tb}")
            send_json({"status": "error", "error": str(e)})

    log("服务端关闭 (stdin EOF)")


if __name__ == "__main__":
    main()
