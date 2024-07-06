import os
from pathlib import Path

from pdfminer.high_level import extract_text

data_path = Path("./data/raw")
extracted_path = Path("./data/extracted")

for file in data_path.glob("**/*"):
    print(f"> Extracting {file.stem} and writing to text file")
    text = extract_text(file)
    output = f"{str(extracted_path)}/{file.stem}.txt"

    with open(output, "w") as f:
        f.write(text)
