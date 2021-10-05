import os

class AppProperties:
    app_home = "/media/data/projects/ai/gesture-sensor"

    # dir paths
    data_dir = os.path.join(app_home, "data")
    gesture_config = os.path.join(app_home, "config", "gesture_config.json")

    # create mandatory dirs
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
