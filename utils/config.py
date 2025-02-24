import easydict
import yaml

def update_config(config_file):
    with open(config_file) as f:
        config = easydict.EasyDict(yaml.load(f, Loader=yaml.FullLoader))
        return config