import numpy as np
import cv2
from matplotlib import pyplot as plt

cap = cv2.VideoCapture("movie.mp4")

count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(count)

ret, frame = cap.read()
count = count - 1
print(count)
hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

satulationAverage = [i[0] for i in cv2.calcHist([hsv], [1], None, [256], [0, 256])]
valueAverage = [i[0] for i in cv2.calcHist([hsv], [2], None, [256], [0, 256])]

for frameNum in range(count - 1):
    ret, frame = cap.read()
    count = count - 1
    print(count)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    satulation = [i[0] for i in cv2.calcHist([hsv], [1], None, [256], [0, 256])]
    value = [i[0] for i in cv2.calcHist([hsv], [2], None, [256], [0, 256])]

    for i in range(256):
        satulationAverage[i] = np.sum([satulationAverage[i], satulation[i]])
        valueAverage[i] = np.sum([valueAverage[i], value[i]])

cap.release()

x = np.arange(256)
plt.plot(x, satulationAverage, color="r", label="satulation")
plt.plot(x, valueAverage, color="b", label="value")

plt.show()
