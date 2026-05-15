"""
Split clean and noisy audio files into k-second chunks.

Filelist format: clean | noise | noisy | text [| snr]
"""

import argparse
import soundfile as sf
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Split audio files into k-second chunks")
    parser.add_argument("--k", type=float, default=4.0, help="Chunk duration in seconds")
    parser.add_argument("--output_dir", type=str, default='split_audio', help="Output directory for split audio")
    parser.add_argument(
        "--filelists",
        nargs="+",
        default=[
            "/workspace/DB/librispeech_se_snr-515/metadata.txt",
            "/workspace/DB/librispeech_se_snr-515_eval/dev-clean/metadata.txt",
            "/workspace/DB/librispeech_se_snr-515_eval/test-clean/metadata.txt",
        ],
        help="Filelist paths",
    )
    parser.add_argument(
        "--min_duration", type=float, default=0.5,
        help="Minimum chunk duration to keep (seconds). Shorter trailing chunks are replaced by a tail chunk.",
    )
    return parser.parse_args()


def parse_line(line):
    parts = [p.strip() for p in line.strip().split("|")]
    if len(parts) < 4:
        return None
    clean = parts[0]
    noise = parts[1]
    noisy = parts[2]
    text = parts[3]
    snr = parts[4] if len(parts) >= 5 else None
    return clean, noise, noisy, text, snr


def write_chunk(c_chunk, n_chunk, c_out, n_out, sr, c_path, n_path, noise_path, text, snr):
    sf.write(str(c_out), c_chunk, sr)
    sf.write(str(n_out), n_chunk, sr)
    if snr is not None:
        return f"{c_out} | {noise_path} | {n_out} | {text} | {snr}"
    return f"{c_out} | {noise_path} | {n_out} | {text}"


def process_filelist(filelist_path, k, output_dir, min_duration, split_name=None):
    filelist_path = Path(filelist_path)
    if split_name is None:
        split_name = filelist_path.stem

    clean_out_dir = output_dir / split_name / "clean"
    noisy_out_dir = output_dir / split_name / "noisy"
    clean_out_dir.mkdir(parents=True, exist_ok=True)
    noisy_out_dir.mkdir(parents=True, exist_ok=True)

    new_filelist_path = output_dir / f"{split_name}.txt"
    new_lines = []
    skipped = 0
    written = 0

    with open(filelist_path, "r") as f:
        lines = [l for l in f if l.strip()]

    for line_idx, line in enumerate(lines):
        parsed = parse_line(line)
        if parsed is None:
            print(f"  [WARN] Skipping malformed line {line_idx+1}: {line.strip()[:80]}")
            skipped += 1
            continue

        clean_path, noise_path, noisy_path, text, snr = parsed

        try:
            clean_audio, clean_sr = sf.read(clean_path, always_2d=False)
            noisy_audio, noisy_sr = sf.read(noisy_path, always_2d=False)
        except Exception as e:
            print(f"  [WARN] Cannot read audio (line {line_idx+1}): {e}")
            skipped += 1
            continue

        if clean_sr != noisy_sr:
            print(f"  [WARN] Sample rate mismatch at line {line_idx+1}: clean={clean_sr}, noisy={noisy_sr}. Skipping.")
            skipped += 1
            continue

        sr = clean_sr
        k_samples = int(k * sr)
        min_samples = int(min_duration * sr)

        if len(clean_audio) != len(noisy_audio):
            print(f"  [WARN] Length mismatch at line {line_idx+1}: clean={len(clean_audio)}, noisy={len(noisy_audio)}. Truncating to shorter.")
        min_len = min(len(clean_audio), len(noisy_audio))
        clean_audio = clean_audio[:min_len]
        noisy_audio = noisy_audio[:min_len]

        orig_stem = Path(clean_path).stem

        for chunk_idx, start in enumerate(range(0, min_len, k_samples)):
            c_chunk = clean_audio[start:start + k_samples]
            n_chunk = noisy_audio[start:start + k_samples]

            if len(c_chunk) < min_samples:
                # Trailing chunk too short: replace with last k_samples (overlapping tail chunk)
                if min_len >= k_samples:
                    chunk_name = f"{orig_stem}_{chunk_idx:04d}.wav"
                    c_out = clean_out_dir / chunk_name
                    n_out = noisy_out_dir / chunk_name
                    new_line = write_chunk(
                        clean_audio[-k_samples:], noisy_audio[-k_samples:],
                        c_out, n_out, sr, clean_path, noisy_path, noise_path, text, snr,
                    )
                    new_lines.append(new_line)
                    written += 1
                break

            chunk_name = f"{orig_stem}_{chunk_idx:04d}.wav"
            c_out = clean_out_dir / chunk_name
            n_out = noisy_out_dir / chunk_name
            new_line = write_chunk(
                c_chunk, n_chunk, c_out, n_out, sr, clean_path, noisy_path, noise_path, text, snr,
            )
            new_lines.append(new_line)
            written += 1

        if (line_idx + 1) % 500 == 0:
            print(f"  Processed {line_idx+1}/{len(lines)} lines...")

    with open(new_filelist_path, "w") as f:
        f.write("\n".join(new_lines) + "\n")

    print(f"  Done: {written} chunks written, {skipped} lines skipped -> {new_filelist_path}")
    return written, skipped


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Chunk duration : {args.k}s")
    print(f"Output directory: {output_dir}")
    print(f"Min chunk duration: {args.min_duration}s")
    print()

    stems = [Path(f).stem for f in args.filelists]
    if len(stems) != len(set(stems)):
        split_names = [Path(f).parent.name for f in args.filelists]
    else:
        split_names = stems

    for filelist, split_name in zip(args.filelists, split_names):
        print(f"Processing: {filelist}  ->  {split_name}.txt")
        process_filelist(filelist, args.k, output_dir, args.min_duration, split_name)
        print()


if __name__ == "__main__":
    main()
