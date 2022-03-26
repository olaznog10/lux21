import os
import time
import numpy as np
import pandas as pd

mean_time = 0
for i in range(100):
    ts = time.time()
    home_dir = os.system("lux-ai-2021 main.py gonz.py "
                         "--python=venv/bin/python3 "
                         "--out=replays/replay_gonz.json "
                         "--loglevel=0 "
                         "> out.txt")
    ite_time = time.time() - ts
    mean_time = mean_time + ite_time
    if i % 10 == 0:
        print('Game', i, ite_time, (mean_time/(i+1)))
