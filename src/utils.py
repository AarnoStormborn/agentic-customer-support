import yaml
from box import ConfigBox

def read_config(filepath):
    with open(filepath) as f:
        data = yaml.safe_load(f)
    data = ConfigBox(data)
    return data  