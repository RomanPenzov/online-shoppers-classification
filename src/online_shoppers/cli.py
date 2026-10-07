"""Команды обучения и инференса."""

import argparse
from pathlib import Path

from .experiment import run_experiment
from .inference import predict_file
from .reproducibility import verify_runs


def build_parser() -> argparse.ArgumentParser:
    """Создай парсер CLI.

    Returns:
        Парсер с подкомандами train и predict.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train", help="Полный фиксированный эксперимент")
    train.add_argument("--data", type=Path, default=Path("data/online_shoppers_intention.csv"))
    train.add_argument("--download", action="store_true", help="Скачать CSV UCI, если его нет")
    train.add_argument("--output", type=Path, default=Path("artifacts/part_3"))
    predict = commands.add_parser("predict", help="Инференс из доверенного model.joblib")
    predict.add_argument("--model", type=Path, required=True)
    predict.add_argument("--data", type=Path, required=True)
    predict.add_argument("--output", type=Path, required=True)
    verify = commands.add_parser("verify", help="Сравнить два полных запуска")
    verify.add_argument("--first", type=Path, required=True)
    verify.add_argument("--second", type=Path, required=True)
    return parser


def main() -> None:
    """Прочитай аргументы и запусти соответствующий сценарий."""
    args = build_parser().parse_args()
    if args.command == "train":
        run_experiment(args.data, args.output, download=args.download)
    elif args.command == "verify":
        verify_runs(args.first, args.second)
        print("Протокол, выбор и численные результаты воспроизведены.")
    else:
        predict_file(args.model, args.data, args.output)


if __name__ == "__main__":
    main()
