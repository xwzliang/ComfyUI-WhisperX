import os
import re
import json
import srt
import torch
import time
import whisperx
import folder_paths
import cuda_malloc
import translators as ts
from tqdm import tqdm
from datetime import timedelta
import spacy
from spacy.cli import download
from transformers import AutoTokenizer, AutoModelForTokenClassification
from zhpr.predict import DocumentDataset, merge_stride, decode_pred
from torch.utils.data import DataLoader

def load_spacy_model(name="zh_core_web_sm"):
    """
    Load a spaCy Chinese model, downloading it if necessary, and ensure sentence boundary detection.
    """
    try:
        nlp = spacy.load(name)
    except OSError:
        print(f"Model '{name}' not found. Downloading...")
        download(name)
        nlp = spacy.load(name)

    # If there's no sentencizer or parser, add a rule-based sentencizer
    if "sentencizer" not in nlp.pipe_names:
        nlp.add_pipe("sentencizer")
    return nlp

input_path = folder_paths.get_input_directory()
out_path = folder_paths.get_output_directory()

def nlp_resegment_for_zh(segments, word_segments, nlp):
    """
    Split each WhisperX segment into NLP-based sentences and align timings.
    """
    PUNC_MODEL = "p208p2002/zh-wiki-punctuation-restore"
    tokenizer = AutoTokenizer.from_pretrained(PUNC_MODEL)
    model = AutoModelForTokenClassification.from_pretrained(PUNC_MODEL)

    def add_punctuation(text, window_size=256, step=200):
        # chunk+stride dataset so long inputs get handled
        ds = DocumentDataset(text, window_size=window_size, step=step)
        loader = DataLoader(ds, batch_size=2, shuffle=False)
        all_out = []
        for batch in loader:
            # batch is list of token-ID lists
            enc = {"input_ids": batch}
            logits = model(**enc).logits
            preds = logits.argmax(-1)
            for tok_ids, pred_ids in zip(batch, preds):
                # convert predictions back to labels
                labels = [model.config.id2label[i.item()] for i in pred_ids[: len(tok_ids)]]
                all_out.append(list(zip(tokenizer.convert_ids_to_tokens(tok_ids), labels)))
        merged = merge_stride(all_out, step)
        punctuated = decode_pred(merged)
        return "".join(punctuated)
    
    new_subs = []
    idx = 1

    for seg in segments:
        text = seg.get("text", "").strip()
        if not text:
            continue

        # 1. NLP sentence split
        raw = seg.get("text", "").strip()
        if not raw:
            continue
        punctuated = add_punctuation(raw)
        doc = nlp(punctuated)
        sents = [s.text.strip() for s in doc.sents if s.text.strip()]

        # 2. collect word-level timestamps inside this segment
        words = [w for w in word_segments
                 if "start" in w and seg["start"] - 1e-3 <= w["start"] <= seg["end"] + 1e-3]

        w_i = 0
        for sent in sents:
            sent_start = None
            sent_end = None
            accum = ""

            # accumulate words until they match the sentence
            while w_i < len(words) and re.sub(r"\s+", "", accum) != re.sub(r"\s+", "", sent):
                w = words[w_i]
                if sent_start is None:
                    sent_start = w["start"]
                accum += w["word"]
                sent_end = w["end"]
                w_i += 1

            # if we got timings, add subtitle
            if sent_start is not None and sent_end is not None:
                new_subs.append(
                    srt.Subtitle(
                        index=idx,
                        start=timedelta(seconds=sent_start),
                        end=timedelta(seconds=sent_end),
                        content=sent
                    )
                )
                idx += 1

    return new_subs

class PreViewSRT:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {"srt": ("SRT",)},
        }

    CATEGORY = "AIFSH_WhisperX"

    RETURN_TYPES = ()
    OUTPUT_NODE = True

    FUNCTION = "show_srt"

    def show_srt(self, srt):
        srt_name = os.path.basename(srt)
        dir_name = os.path.dirname(srt)
        dir_name = os.path.basename(dir_name)
        with open(srt, "r", encoding="utf-8") as f:
            srt_content = f.read()
        return {"ui": {"srt": [srt_content, srt_name, dir_name]}}


class SRTToString:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {"srt": ("SRT",)},
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "read"

    CATEGORY = "AIFSH_FishSpeech"

    def read(self, srt):
        srt_name = os.path.basename(srt)
        dir_name = os.path.dirname(srt)
        dir_name = os.path.basename(dir_name)
        with open(srt, "r") as f:
            srt_content = f.read()
        return (srt_content,)


class WhisperX:
    @classmethod
    def INPUT_TYPES(s):
        model_list = ["large-v3", "distil-large-v3", "large-v2"]
        translator_list = [
            "alibaba",
            "apertium",
            "argos",
            "baidu",
            "bing",
            "caiyun",
            "cloudTranslation",
            "deepl",
            "elia",
            "google",
            "hujiang",
            "iciba",
            "iflytek",
            "iflyrec",
            "itranslate",
            "judic",
            "languageWire",
            "lingvanex",
            "mglip",
            "mirai",
            "modernMt",
            "myMemory",
            "niutrans",
            "papago",
            "qqFanyi",
            "qqTranSmart",
            "reverso",
            "sogou",
            "sysTran",
            "tilde",
            "translateCom",
            "translateMe",
            "utibet",
            "volcEngine",
            "yandex",
            "yeekit",
            "youdao",
        ]
        lang_list = ["zh", "en", "ja", "ko", "ru", "fr", "de", "es", "pt", "it", "ar"]
        return {
            "required": {
                "audio": ("AUDIOPATH",),
                "model_type": (model_list, {"default": "large-v3"}),
                "batch_size": ("INT", {"default": 4}),
                "if_mutiple_speaker": ("BOOLEAN", {"default": False}),
                "use_auth_token": (
                    "STRING",
                    {
                        "default": "put your huggingface user auth token here for Assign speaker labels"
                    },
                ),
                "if_translate": ("BOOLEAN", {"default": False}),
                "translator": (translator_list, {"default": "alibaba"}),
                "to_language": (lang_list, {"default": "en"}),
                "transcription": ("STRING", {"multiline": True, "default": ""}),
                "if_return_word_srt": ("BOOLEAN", {"default": True}),
            },
        }

    CATEGORY = "AIFSH_WhisperX"

    RETURN_TYPES = ("SRT", "SRT")
    RETURN_NAMES = ("ori_SRT", "trans_SRT")
    FUNCTION = "get_srt"

    def get_srt(
        self,
        audio,
        model_type,
        batch_size,
        if_mutiple_speaker,
        use_auth_token,
        if_translate,
        translator,
        to_language,
        transcription,
        if_return_word_srt,
    ):
        compute_type = "float16"

        base_name = os.path.basename(audio)[:-4]
        device = "cuda" if cuda_malloc.cuda_malloc_supported() else "cpu"
        # 1. Transcribe with original whisper (batched)
        print("Loading whisper model")
        model = whisperx.load_model(model_type, device, compute_type=compute_type)
        audio = whisperx.load_audio(audio)
        result = model.transcribe(audio, batch_size=batch_size)
        # print(result["segments"]) # before alignment
        language_code = result["language"]
        # 2. Align whisper output
        print("Loading align model")
        model_a, metadata = whisperx.load_align_model(
            language_code=language_code, device=device
        )
        if transcription:
            # result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)
            transcription_for_forced_alignment = json.loads(transcription)
            result = whisperx.align(
                transcription_for_forced_alignment,
                model_a,
                metadata,
                audio,
                device,
                return_char_alignments=False,
            )
            # print(result["segments"])
        else:
            result = whisperx.align(
                result["segments"],
                model_a,
                metadata,
                audio,
                device,
                return_char_alignments=False,
            )

        # print(result["segments"]) # after alignment

        # delete model if low on GPU resources
        import gc

        gc.collect()
        torch.cuda.empty_cache()
        del model_a, model
        if if_mutiple_speaker:
            # 3. Assign speaker labels
            diarize_model = whisperx.DiarizationPipeline(
                use_auth_token=use_auth_token, device=device
            )

            # add min/max number of speakers if known
            diarize_segments = diarize_model(audio)
            # diarize_model(audio, min_speakers=min_speakers, max_speakers=max_speakers)

            result = whisperx.assign_word_speakers(diarize_segments, result)
            import gc

            gc.collect()
            torch.cuda.empty_cache()
            del diarize_model
        # print(diarize_segments)
        # print(result.segments) # segments are now assigned speaker IDs

        srt_path = os.path.join(out_path, f"{time.time()}_{base_name}.srt")
        trans_srt_path = os.path.join(
            out_path, f"{time.time()}_{base_name}_{to_language}.srt"
        )
        srt_line = []
        word_srt_line = []
        trans_srt_line = []
        for i, res in enumerate(result["segments"]):
            start = timedelta(seconds=res["start"])
            end = timedelta(seconds=res["end"])
            try:
                speaker_name = res["speaker"][-1]
            except:
                speaker_name = ""
            content = res["text"]
            srt_line.append(
                srt.Subtitle(
                    index=i + 1, start=start, end=end, content=speaker_name + content
                )
            )

        if language_code == "zh":
            nlp = load_spacy_model()
            srt_line = nlp_resegment_for_zh(
                result["segments"],
                result["word_segments"],
                nlp
            )
            print("After nlp re-segmentation:", srt_line)

        if if_translate:
            for subtitle in srt_line:
                try:
                    content = ts.translate_text(
                        query_text=subtitle.content,
                        translator=translator,
                        from_language=language_code,
                        to_language=to_language,
                    )
                    content = content.replace(".", "").replace("。", "")
                    print(f"Translated segment {subtitle.index} with content {subtitle.content} to: {content}")
                except Exception as e:
                    print(f"Translation failed for segment {i + 1}: {e}")
                    content = ""
                trans_srt_line.append(
                    srt.Subtitle(
                        index=i + 1,
                        start=start,
                        end=end,
                        content=speaker_name + content,
                    )
                )

        if if_return_word_srt:
            for i, res in enumerate(
                tqdm(
                    result["word_segments"],
                    desc="Transcribing ...",
                    total=len(result["word_segments"]),
                )
            ):
                if "start" not in res:
                    continue
                start = timedelta(seconds=res["start"])
                end = timedelta(seconds=res["end"])
                try:
                    speaker_name = res["speaker"][-1]
                except:
                    speaker_name = ""
                content = res["word"]
                word_srt_line.append(
                    srt.Subtitle(
                        index=i + 1, start=start, end=end, content=speaker_name + content
                    )
                )
            print("Writing word SRT file")
            print(word_srt_line)
            with open(srt_path, "w", encoding="utf-8") as f:
                f.write(srt.compose(word_srt_line))
        else:
            print("Writing SRT file")
            print(srt_line)
            with open(srt_path, "w", encoding="utf-8") as f:
                f.write(srt.compose(srt_line))
        
        with open(trans_srt_path, "w", encoding="utf-8") as f:
            f.write(srt.compose(trans_srt_line))

        if if_translate:
            return (srt_path, trans_srt_path)
        else:
            return (srt_path, srt_path)


class LoadAudioPath:
    @classmethod
    def INPUT_TYPES(s):
        files = [
            f
            for f in os.listdir(input_path)
            if os.path.isfile(os.path.join(input_path, f))
            and f.split(".")[-1] in ["wav", "mp3", "WAV", "flac", "m4a"]
        ]
        return {
            "required": {"audio": (sorted(files),)},
        }

    CATEGORY = "AIFSH_WhisperX"

    RETURN_TYPES = ("AUDIOPATH",)
    FUNCTION = "load_audio"

    def load_audio(self, audio):
        audio_path = folder_paths.get_annotated_filepath(audio)
        return (audio_path,)
