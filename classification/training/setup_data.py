# Split data into train and test set
from sklearn.model_selection import train_test_split
import re
import json
import os
import glob

IN_PATH = "./data/labeled_data"
OUT_PATH = "./data/split_data"
TRAIN_FILE_NAME = "train_only_9.json"
TEST_FILE_NAME = "test_only_9.json"

# List with skier ids to not add to the dataset
EXLUDED_SKIER_IDS = [1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13]
# List of videos to exclude
EXLUDED_VIDEO_IDS = []

test_size = 0.1
seed = 42


def main():
    print("Splitting data...")
    data = []

    # Loop through the json files in folder
    for file in glob.glob(IN_PATH + '/*.json'):
        with open(file, 'r') as f:
            data_json = json.load(f)
            # Loop through the cycles in each json file
            for cycle in data_json.values():
                #cycle["Video"] = os.path.basename(file)
                #cycle["Video"] = "".join(re.findall(r"\d+", os.path.basename(file)))
                if cycle["Skier_id"] in EXLUDED_SKIER_IDS:
                    continue
                
                # filter out id and cut if there
                match = re.search(r'(\d+)(?:_cut)?', os.path.basename(file))
                if match:
                    result = match.group(1)
                    if '_cut' in os.path.basename(file):
                        result += '_cut'
                    cycle["Video"] = result
                else:
                    raise ValueError("Filename does not contain ID")
                data.append(cycle)


    train, test = train_test_split(data, test_size=test_size, random_state=seed)

    print(f"Splitting into {len(train)} train cycles and test {len(test)} cycles.")
    train_dict = {}
    test_dict = {}

    # create dict to make readable
    for i, cycle in enumerate(train):
        train_dict[f"Cycle {i+1}"] = cycle
    for i, cycle in enumerate(test):
        test_dict[f"Cycle {i+1}"] = cycle

    # Saves the readable dicts
    with open(os.path.join(OUT_PATH, TRAIN_FILE_NAME), "w") as outfile: 
        json.dump(train_dict, outfile, indent=4)
    with open(os.path.join(OUT_PATH, TEST_FILE_NAME), "w") as outfile:    
        json.dump(test_dict, outfile, indent=4)

    print("Split complete")

if __name__ == '__main__':
    main()