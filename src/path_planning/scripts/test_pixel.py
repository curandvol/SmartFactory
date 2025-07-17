from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# 1) PGM 불러오기 (경로를 실제 위치로 수정)
img = Image.open('/home/seongunkim/SmartFactory/src/path_planning/maps/IT2_slam/it2_last.pgm')
arr = np.array(img)

print(np.unique(arr))
