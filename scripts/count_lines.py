# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
import os
import argparse

def count_python_lines(directory, exclude_dirs=None):
    total_lines = 0
    exclude_dirs = exclude_dirs or []

    for root, dirs, files in os.walk(directory):
        dirs[:] = [d for d in dirs if d not in exclude_dirs]

        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                        total_lines += len(lines)
                except Exception as e:
                    print(f"Unable to read file {file_path}: {e}")
    return total_lines

def main():
    parser = argparse.ArgumentParser(description="Count the total lines of all Python scripts in the directory and its subdirectories")
    parser.add_argument("directory", help="Directory path to be counted")
    parser.add_argument("--exclude", nargs="*", default=[], help="Directories to exclude (can specify multiple)")
    args = parser.parse_args()

    if os.path.isdir(args.directory):
        lines = count_python_lines(args.directory, args.exclude)
        print(f"The total number of lines in all Python scripts under the directory {args.directory} and its subdirectories is: {lines}")
    else:
        print(f"Error: '{args.directory}' is not a valid directory.")

if __name__ == "__main__":
    main()
