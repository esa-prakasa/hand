import os

os.system("cls")


inputPath = "C:\\Users\\INKOM06\\Pictures\\handwash\\mod1\\RTh\\"

files = os.listdir(inputPath)
N = len(files)

print(N)


st = 0
fn = 176

for i in range(st,(fn+1),1):
	print(i)

st = (fn+1)
fn = 399

for i in range(st,(fn+1),1):
	print(i)

st = (fn+1)
fn = 637

for i in range(st,(fn+1),1):
	print(i)


st = (fn+1)
fn = 818

for i in range(st,(fn+1),1):
	print(i)


st = (fn+1)
fn = 1085

for i in range(st,(fn+1),1):
	print(i)

st = (fn+1)
fn = 1317

for i in range(st,(fn+1),1):
	print(i)


st = (fn+1)
fn = 1434

for i in range(st,(fn+1),1):
	print(i)

# 0 : 0000 s.d 0176
# 1 : 0177 s.d 0399
# 2 : 0400 s.d 0637
# 3 : 0638 s.d 0818
# 4 : 0819 s.d 1085
# 5 : 1086 s.d 1317
# 6 : 1318 s.d 1434