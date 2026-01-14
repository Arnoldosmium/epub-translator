import argparse
import sys
from pathlib import Path

from tqdm import tqdm

from .llm import LLM
from .translation import FillFailedEvent, translate
from .xml_translator import SubmitKind

DEFAULT_URL = "https://api.openai.com/v1"
DEFAULT_MODEL = "gpt-4o"
DEFAULT_TOKEN_ENCODING = "o200k_base"
DEFAULT_LANGUAGE = "Chinese"
DEFAULT_MODE = "append_block"
DEFAULT_CONCURRENCY = 4


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="translate_epub",
        description="Translate EPUB files using LLM while preserving original text",
    )
    parser.add_argument("epub_path", type=str, help="Path to the source EPUB file")
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Output EPUB file path (default: auto-generated based on mode)",
    )
    parser.add_argument(
        "-k", "--key-file",
        type=str,
        required=True,
        help="Path to file containing API key",
    )
    parser.add_argument(
        "-m", "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"LLM model name (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "-u", "--url",
        type=str,
        default=DEFAULT_URL,
        help=f"API endpoint URL (default: {DEFAULT_URL})",
    )
    parser.add_argument(
        "-e", "--encoding",
        type=str,
        default=DEFAULT_TOKEN_ENCODING,
        help=f"Token encoding (default: {DEFAULT_TOKEN_ENCODING})",
    )
    parser.add_argument(
        "-l", "--language",
        type=str,
        default=DEFAULT_LANGUAGE,
        help=f"Target language (default: {DEFAULT_LANGUAGE})",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["replace", "append_text", "append_block"],
        default=DEFAULT_MODE,
        help=f"Translation mode (default: {DEFAULT_MODE})",
    )
    parser.add_argument(
        "-c", "--concurrency",
        type=int,
        default=DEFAULT_CONCURRENCY,
        help=f"Number of concurrent translation tasks (default: {DEFAULT_CONCURRENCY})",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Cache directory for resuming translations",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Maximum tokens for LLM response (default: model's default)",
    )

    args = parser.parse_args()

    source_path = Path(args.epub_path)
    if not source_path.exists():
        print(f"Error: Source file '{source_path}' does not exist", file=sys.stderr)
        sys.exit(1)

    key_file = Path(args.key_file)
    if not key_file.exists():
        print(f"Error: Key file '{key_file}' does not exist", file=sys.stderr)
        sys.exit(1)

    api_key = key_file.read_text().strip()
    if not api_key:
        print(f"Error: Key file '{key_file}' is empty", file=sys.stderr)
        sys.exit(1)

    mode_map = {
        "replace": SubmitKind.REPLACE,
        "append_text": SubmitKind.APPEND_TEXT,
        "append_block": SubmitKind.APPEND_BLOCK,
    }
    submit_kind = mode_map[args.mode]

    if args.output:
        target_path = Path(args.output)
    else:
        target_path = source_path.with_suffix(f".{args.mode}.epub")

    llm = LLM(
        key=api_key,
        url=args.url,
        model=args.model,
        token_encoding=args.encoding,
        cache_path=args.cache_dir,
        max_tokens=args.max_tokens,
    )

    print(f"Source: {source_path}")
    print(f"Target: {target_path}")
    print(f"Model: {args.model}")
    print(f"Language: {args.language}")
    print(f"Mode: {args.mode}")
    print(f"Concurrency: {args.concurrency}")
    print()

    with tqdm(total=100, desc="Translating", unit="%", bar_format="{l_bar}{bar}| {n:.1f}/{total:.0f}%") as pbar:
        last_progress = 0.0

        def on_progress(progress: float) -> None:
            nonlocal last_progress
            increment = (progress - last_progress) * 100
            pbar.update(increment)
            last_progress = progress

        def on_fill_failed(event: FillFailedEvent) -> None:
            if event.over_maximum_retries:
                tqdm.write(
                    f"Warning: Maximum retries reached. Error: {event.error_message}"
                )
            else:
                tqdm.write(f"Retry {event.retried_count}: {event.error_message}")

        translate(
            llm=llm,
            concurrency=args.concurrency,
            target_language=args.language,
            submit=submit_kind,
            source_path=source_path,
            target_path=target_path,
            on_progress=on_progress,
            on_fill_failed=on_fill_failed,
        )

    print(f"\nTranslation complete: {target_path}")


if __name__ == "__main__":
    main()
