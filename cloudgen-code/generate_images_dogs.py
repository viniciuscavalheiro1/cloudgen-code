import dog
import string
import random
import os

#generate names random for imagenames
def random_names(size=10, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))

for i in range(1, 4):
    dog.getDog(filename = 'images/' + random_names())

#get size all images
sizeFolder = 0
cwd = os.getcwd()

for image in os.scandir(cwd):
    sizeFolder += os.path.getsize(image)


print("Size folder image dogs: {0}.\n".format(sizeFolder))
