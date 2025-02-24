# Split data into train and test set
from utils import *
from sklearn.model_selection import train_test_split
import re

path = "../cycle_splits/labeled_data"
test_size = 0.1
seed = 42


def main():
    print("Splitting data...")
    data = []

    # Loop through the json files in folder
    for file in glob.glob(path + '/*.json'):
        with open(file, 'r') as f:
            data_json = json.load(f)
            # Loop through the cycles in each json file
            for cycle in data_json.values():
                #cycle["video"] = os.path.basename(file)
                cycle["video"] = "".join(re.findall(r"\d+", os.path.basename(file)))
                data.append(cycle)


    train, test = train_test_split(data, test_size=test_size, random_state=seed)

    print(f"Splitting into {len(train)} train cycles and test {len(test)} cycles.")
    train_dict = {}
    test_dict = {}

    # create dict to make readable
    for i, cycle in enumerate(train):
        train_dict[f"cycle_{i+1}"] = cycle
    for i, cycle in enumerate(test):
        test_dict[f"cycle_{i+1}"] = cycle

    # Saves the readable dicts
    with open("train.json", "w") as outfile: 
        json.dump(train_dict, outfile, indent=4)
    with open("test.json", "w") as outfile:    
        json.dump(test_dict, outfile, indent=4)

    print("Split complete")

if __name__ == '__main__':
    main()