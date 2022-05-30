#import mydatum_pb2
#import data.mydatum_pb2
import mydata.mydatum_pb2 as mydatum_pb2
#from data.mydatum_pb2 import mydatum_pb2
import cv2
from PIL import Image
import numpy as np
ssss = "/home/user/data1/DBs/antispoofing/CelebA_spoofing/CelebA_Spoof/Data/Test/9994/live/558944.png"
im = Image.open(ssss)
npimg = np.array(im)
print (im)
print (im.width)
print (im.height)
print (npimg.shape)

mydatum = mydatum_pb2.myDatum()
mydatum.width = npimg.shape[1]#im.width
mydatum.height = npimg.shape[0]#im.height
mydatum.channels = npimg.shape[2]
mydatum.label = 1
mydatum.data = npimg.tobytes()
mydatum.path = "abc/abc/j.jpg"

# im.show()

mydatum2 = mydatum_pb2.myDatum()
mydatum2.ParseFromString(mydatum.SerializeToString())
dst = np.fromstring(mydatum2.data, dtype=np.uint8)
dst= dst.reshape(mydatum2.height, mydatum2.width, mydatum2.channels)
imgpil = Image.fromarray(dst)
print (dst.shape)
print (mydatum2.path)
imgpil.show()

