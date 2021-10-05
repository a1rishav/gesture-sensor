import subprocess
import time

while True:
    result = subprocess.run(['xdotool', 'getwindowfocus', 'getwindowname'], stdout=subprocess.PIPE)
    print(result.stdout)
    time.sleep(2)