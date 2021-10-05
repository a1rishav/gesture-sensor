import json
from setup.app_properties import AppProperties as props

with open(props.gesture_config) as json_file:
    data = json.load(json_file)
    print()