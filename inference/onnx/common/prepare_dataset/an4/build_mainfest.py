# --- Building Manifest Files --- #
import json
import os
import librosa
import argparse

def parse_args():
    """
    :return parser args
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir_path', default = './data/', help = 'where the directory containing the an4 files')
    parser.add_argument('--dataset_path', default = './an4/wav/an4test_clstk', help = 'where the directory containing the dataset')
    args = parser.parse_args()
    return args

# Function to build a manifest
def build_manifest(transcripts_path, manifest_path, wav_path):
    with open(transcripts_path, 'r') as fin:
        with open(manifest_path, 'w') as fout:
            for line in fin:
                # Lines look like this:
                # <s> transcript </s> (fileID)
                transcript = line[: line.find('(')-1].lower()
                transcript = transcript.replace('<s>', '').replace('</s>', '')
                transcript = transcript.strip()

                file_id = line[line.find('(')+1 : -2]  # e.g. "cen4-fash-b"
                audio_path = os.path.join(
                    data_dir, wav_path,
                    file_id[file_id.find('-')+1 : file_id.rfind('-')],
                    file_id + '.wav')

                duration = librosa.core.get_duration(filename=audio_path)

                # Write the metadata to the manifest
                metadata = {
                    "audio_filepath": audio_path,
                    "duration": duration,
                    "text": transcript
                }
                json.dump(metadata, fout)
                fout.write('\n')

# Building Manifests
if __name__ == "__main__":
    args = parse_args()
    data_dir = args.dir_path
    test_transcripts = data_dir + '/an4/etc/an4_test.transcription'
    test_manifest = data_dir + '/an4/test_manifest.json'
    wav_path = args.dataset_path
    if not os.path.isfile(test_manifest):
        build_manifest(test_transcripts, test_manifest, wav_path)
        print("Test manifest created.")
    else:
        print("Test manifest already exists.")
