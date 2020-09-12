import numpy as np
import os
from tkinter import *
window=Tk()
# add widgets here


os.system("cls")

window.title('Hello Python')
window.geometry("300x200+300+300")


poseCount = np.zeros(7, dtype=int)

poseCount[0] = 5
poseCount[1] = 100
poseCount[2] = 200
poseCount[3] = 300
poseCount[4] = 400
poseCount[5] = 500
poseCount[6] = 600


totalFrame = 0
for framePerPose in poseCount:
	totalFrame = totalFrame + framePerPose
print(totalFrame)

secondPerFrame = 1/30


for idx in range(7):
	poseDuration = poseCount[idx]*secondPerFrame
	posePct = poseCount[idx]/totalFrame
	idxStr = "{:2d}".format(idx)

	text = ("Pose %s: %3.2f  sec Pct: %3.2f %%")%(idxStr,poseDuration,posePct)
	lb0=Label(window, text=text, fg='red', font=("Helvetica", 10))
	yPost = 20 + idx*20
	lb0.place(x=20, y=yPost)



window.mainloop()