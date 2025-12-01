import os
import re
import json
from glob import glob

# ï¼ˆcobine excited/frustration ï¼‰
EMO_MAP = {
    "ang": "angry",
    "fru": "angry",
    "hap": "happy",
    "exc": "happy",
    "neu": "neutral",
    "sad": "sad"
}

def parse_line(line):
    match = re.match(r"\[(.*?) - (.*?)\]\s+(Ses.*?_\w+)\s+(\w+)", line)
    if match:
        start, end, utt_id, emo = match.groups()
        if emo in EMO_MAP:
            return {
                "utt_id": utt_id,
                "emotion": EMO_MAP[emo],
                "start": float(start),
                "end": float(end)
            }
    return None

def collect_all_annotations(iemocap_root, output_path="iemocap_all.json"):
    all_data = []
    for session_num in range(1, 6):
        session = f"Session{session_num}"
        emo_dir = os.path.join(iemocap_root, session, "dialog", "EmoEvaluation")
        wav_dir = os.path.join(iemocap_root, session, "sentences", "wav")

        emo_files = glob(os.path.join(emo_dir, "*.txt"))
        for emo_file in emo_files:
            with open(emo_file, "r") as f:
                for line in f:
                    if line.startswith("["):
                        parsed = parse_line(line)
                        if parsed:
                            dialog_id = "_".join(parsed["utt_id"].split("_")[:3])
                            wav_path = os.path.join(wav_dir, dialog_id, parsed["utt_id"] + ".wav")
                            if os.path.exists(wav_path):
                                parsed["audio_path"] = wav_path
                                parsed["session"] = session
                                all_data.append(parsed)

    with open(output_path, "w") as f:
        json.dump(all_data, f, indent=2)
    print(f"[âœ”] Saved {len(all_data)} utterances to {output_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, required=True, help="Path to IEMOCAP root directory")
    parser.add_argument("--out", type=str, default="iemocap_all.json", help="Output JSON file name")
    args = parser.parse_args()

    collect_all_annotations(args.root, args.out)
