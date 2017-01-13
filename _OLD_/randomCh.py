import os, random
for i in range(0,10000):
    choice = random.choice(os.listdir("train/0/"))
    os.rename("train/0/"+choice, "validation/0/"+choice)
