import os
import json
import pandas as pd
import argparse


parser = argparse.ArgumentParser(description='Generating metadata.csv from data folder')
parser.add_argument('--folder_dir', help='dataset folder containing *.png/*.jpg/*.jpeg and coresponding *.txt')
args = parser.parse_args()

images = []
texts = []

files = os.listdir(args.folder_dir)
for file_name in files:
    if file_name.lower().endswith(('jpg', 'jpeg', 'png')):
        prefix = os.path.splitext(file_name)[0]
        txt_file = prefix + '.txt'
        txt_file = os.path.join(args.folder_dir, txt_file)
        text = ""
        if os.path.exists(txt_file):
            text = open(txt_file, 'r', encoding='utf-8').read().strip()
            if text.startswith('"') and text.endswith('"'):
                text = json.loads(text)
            if text.startswith("'") and text.endswith("'"):
                text = eval(text)
        images.append(file_name)
        texts.append(text)

# sort file names
try:
    result = list(zip(images, texts))
    result.sort(key=lambda x: int(os.path.splitext(x[0])[0]))
    images, texts = zip(*result)
except Exception as e:
    print(e)
    pass

# generate dataframe
data = {'file_name': images, 'text': texts}
df = pd.DataFrame(data)
df.to_csv(os.path.join(args.folder_dir, 'metadata.csv'), index=False)
