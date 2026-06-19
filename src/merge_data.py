"""Merge raw match data CSV files into a single normalized CSV."""

from pathlib import Path

from .data_collection import load_match_data


def merge_data(input_path: str, output_path: str = 'data/raw/merged_matches.csv') -> Path:
    output_file = Path(output_path)
    input_path = Path(input_path)

    if input_path.is_dir() and output_file.exists() and output_file.parent == input_path:
        output_file.unlink()

    df = load_match_data(str(input_path))
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, index=False)
    print(f'Merged {len(df)} rows into {output_file}')
    return output_file


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description='Merge raw match data CSV files into one normalized CSV file.')
    parser.add_argument('input_path', help='Path to a CSV file or a folder containing raw CSV files.')
    parser.add_argument('--output', default='data/raw/merged_matches.csv', help='Output merged CSV path.')
    args = parser.parse_args()

    merge_data(args.input_path, args.output)


if __name__ == '__main__':
    main()
