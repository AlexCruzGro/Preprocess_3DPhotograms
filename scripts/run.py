import argparse
from src.pipeline import process_subject, load_config

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", default="configs/default.yaml")

    args = parser.parse_args()

    config = load_config(args.config)

    process_subject(args.input, args.output, config)


if __name__ == "__main__":
    main()