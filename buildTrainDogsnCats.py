import os, random
import glob

for i in range(0,2500):
    choice = random.choice(glob.glob("train/cat*"))
    #print(choice.split("/")[-1])
    os.rename(choice, "cats/"+choice.split("/")[-1])
