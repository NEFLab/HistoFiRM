"""Portable command-line entry points."""

import argparse
from .common import load
from .data import Patients
from .prepare import raw_cache, load_cache
from .training import fit
from .prediction import predict, merge_predictions


def main():
    parser = argparse.ArgumentParser(prog="histofirm")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--manifest", action="append", required=True)
    p.add_argument("--output", required=True)
    p = sub.add_parser("train")
    p.add_argument("--fit", required=True)
    p.add_argument("--validation")
    p.add_argument("--selected")
    p.add_argument("--cache", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--fold", required=True, type=int)
    p.add_argument("--device", default="cpu")
    p = sub.add_parser("predict")
    p.add_argument("--manifest", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--chunk-size", type=int, default=1024)
    p = sub.add_parser("merge")
    p.add_argument("--input", action="append", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--external", action="store_true")
    args = parser.parse_args()
    if args.command == "prepare":
        raw_cache([Patients(p) for p in args.manifest], args.output)
    elif args.command == "train":
        config = load(args.config)
        data = Patients(args.fit, True)
        selected = load(args.selected) if args.selected else None
        if selected is not None:
            from .common import digest

            if (
                selected["fold"] != args.fold
                or selected["target_ids"] != data.target_ids
                or selected["cancer_names"] != data.cancer_names
                or selected["protocol_sha256"] != digest(config)
            ):
                raise ValueError("Refit and inner-selection identities differ.")
        fit(
            data,
            Patients(args.validation, True) if args.validation else None,
            load_cache(args.cache),
            config,
            args.output,
            args.fold,
            args.device,
            selected,
        )
    elif args.command == "predict":
        predict(args.checkpoint, Patients(args.manifest), args.output, args.device, args.chunk_size)
    else:
        merge_predictions(args.input, args.output, args.external)


if __name__ == "__main__":
    main()
