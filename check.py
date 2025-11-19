import os

base = "./data/SciTSR/val"

folders = [
    "chunk",
    "img",
    "structure"
]

for folder in folders:
    path = os.path.join(base, folder)
    if os.path.exists(path):
        count = len([f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))])
        print(f"{folder}: {count} files")
    else:
        print(f"{folder}: folder does not exist")
