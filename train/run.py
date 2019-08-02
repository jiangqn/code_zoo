import pynvml
import os
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--program', type=str, default='main.py')
parser.add_argument('--need_memory', type=int, default=12800)
parser.add_argument('--total_gpu', type=int, default=6)
parser.add_argument('--max_memory', type=int, default=16280)
parser.add_argument('--show_round', type=int, default=1000)

config = parser.parse_args()

pynvml.nvmlInit()

flag = False

round = 0
start = time.time()

while True:
    if flag:
        break
    round += 1
    end = time.time()
    if round % config.show_round == 0:
        print("round %d %.4f" % (round, end - start))
    for gpu_id in range(config.total_gpu):
        if flag:
            break
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
        meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
        used = meminfo.used / (1024 * 1024)
        remain = config.max_memory - used
        if remain > config.need_memory:
            print("gpu_id %d" % gpu_id)
            os.system("CUDA_VISIBLE_DEVICES=%d python %s" % (gpu_id, config.program))
            flag = True
