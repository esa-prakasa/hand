import random

def generateFileName():
	text = "ABCDEFGHIJKLMNOPQERSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
	fileName = ""
	for i in range(30):
		fileName = fileName+random.choice(text)
	return fileName





## test, execute the function!

for k in range(1000):
	print(str(k)+"  "+generateFileName())
