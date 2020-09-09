import os
import shutil

os.system("cls")


inputPath = "C:\\Users\\INKOM06\\Pictures\\handwash\\mod1\\RTh\\"
targetPath = "C:\\Users\\INKOM06\\Pictures\\handwash\\mod1\\dataset\\"



def copiedTo(st,fn,pose, inputPath, targetPath):
	for i in range(st,(fn+1),1):
		inPath = inputPath+files[i]
		outPath = targetPath+str(pose)+"\\"+files[i]
		shutil.copyfile(inPath, outPath)
		print(inPath+" copied to "+outPath )





files = os.listdir(inputPath)
N = len(files)

print(N)

st = 0
fn = 176
pose = 0
copiedTo(st, fn, pose, inputPath, targetPath)

st = (fn+1)
fn = 399

pose = pose + 1
copiedTo(st, fn, pose, inputPath, targetPath)


st = (fn+1)
fn = 637

pose = pose + 1
copiedTo(st, fn, pose, inputPath, targetPath)


st = (fn+1)
fn = 818

pose = pose + 1
copiedTo(st, fn, pose, inputPath, targetPath)


st = (fn+1)
fn = 1085

pose = pose + 1
copiedTo(st, fn, pose, inputPath, targetPath)

st = (fn+1)
fn = 1317

pose = pose + 1
copiedTo(st, fn, pose, inputPath, targetPath)


st = (fn+1)
fn = 1434

pose = pose + 1
copiedTo(st, fn, pose, inputPath, targetPath)

'''

# 0 : 0000 s.d 0176   0010
# 1 : 0177 s.d 0399   0187
# 2 : 0400 s.d 0637   0410
# 3 : 0638 s.d 0818   0648
# 4 : 0819 s.d 1085   
# 5 : 1086 s.d 1317
# 6 : 1318 s.d 1434

'''