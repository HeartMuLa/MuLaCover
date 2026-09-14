"""Command-line entry point for MuLaCover generation."""

import argparse

import torch

from .pipeline import DEFAULT_MAX_AUDIO_LENGTH_MS, MuLaCoverGenPipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a cover from reference audio or melody/chord MIDI."
    )
    parser.add_argument("--model_path", required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--ref_audio")
    source.add_argument("--melody_midi")
    parser.add_argument("--chord_midi")
    parser.add_argument("--drum_midi")
    parser.add_argument("--bpm", type=float)
    parser.add_argument(
        "--lyrics", required=True, help="Lyrics text or a UTF-8 text file"
    )
    parser.add_argument("--tags", required=True, help="Style text or a UTF-8 text file")
    parser.add_argument("--save_path", default="cover.wav")
    parser.add_argument("--symbolic_save_dir")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dtype", choices=("float32", "float16", "bfloat16"), default="bfloat16"
    )
    parser.add_argument(
        "--lazy_load", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--max_audio_length_ms", type=int, default=DEFAULT_MAX_AUDIO_LENGTH_MS
    )
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--topk", type=int, default=250)
    parser.add_argument("--cfg_scale", type=float, default=1.5)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--disable_progress", action="store_true")
    return parser


def main(argv=None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.melody_midi and not args.chord_midi:
        parser.error("--melody_midi requires --chord_midi")
    if args.ref_audio and (args.chord_midi or args.drum_midi):
        parser.error("--ref_audio cannot be combined with MIDI inputs")
    if args.melody_midi and args.bpm is not None:
        parser.error("--bpm is only used with --ref_audio")
    if args.seed is not None:
        torch.manual_seed(args.seed)

    device = torch.device(args.device)
    dtype = torch.float32 if device.type == "cpu" else getattr(torch, args.dtype)
    pipe = MuLaCoverGenPipeline.from_pretrained(
        args.model_path,
        device=device,
        dtype={
            "mulacover": dtype,
            "codec": torch.float32,
            "qwen": torch.float32,
            "transcriptor": torch.float32,
        },
        lazy_load=args.lazy_load,
    )
    inputs = {"lyrics": args.lyrics, "tags": args.tags}
    if args.ref_audio:
        inputs.update(ref_audio=args.ref_audio, bpm=args.bpm)
    else:
        inputs.update(
            melody_midi=args.melody_midi,
            chord_midi=args.chord_midi,
            drum_midi=args.drum_midi,
        )
    pipe(
        inputs,
        save_path=args.save_path,
        symbolic_save_dir=args.symbolic_save_dir,
        max_audio_length_ms=args.max_audio_length_ms,
        temperature=args.temperature,
        topk=args.topk,
        cfg_scale=args.cfg_scale,
        disable_progress=args.disable_progress,
    )


if __name__ == "__main__":
    main()
