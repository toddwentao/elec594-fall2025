import json
import numpy as np

def load_json(file_path):
    """Load JSON data from a file."""
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def save_json(data, file_path):
    """Save JSON data to a file."""
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)


def find_iter(data):
    if isinstance(data, list):
        if isinstance(data[0], list):
            return find_iter(data[0])
        else:
            return data[0]

if __name__ == "__main__":
    result = load_json("/home/elec594/Desktop/luigi/vggt/results/classroom/predictions.json")
    for key, value in result.items():
        print(f"{key}")
        print(type(value))
        if isinstance(value, list):
            np_array = np.array(value)
            print(f"  Converted to numpy array with shape: {np_array.shape} and dtype: {np_array.dtype}")
            # print(f"  Length of list: {len(value)}")
            # print(f"  Type of first element: {type(value[0])}")
            # if isinstance(value[0], list):
            #     print(f"    Length of first element: {len(value[0])}")
            #     print(f"    Type of first element's first element: {type(value[0][0])}")
                
        # iter_data = find_iter(value)
        # print(f"  Type of first iter: {type(iter_data)}")
        # print(iter_data)
        