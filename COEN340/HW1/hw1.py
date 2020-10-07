#!/usr/bin/python

import numpy

from scipy import misc
import matplotlib.pyplot as plt

from libtiff import TIFF

import code

def to8bit(im):
  im8 = numpy.zeros(im.shape[0:2],numpy.uint8)
  
  for y,line in enumerate(im):
    for x,pix in enumerate(line):
      im8[y][x] = sum(pix)/3

  return im8

print "Close the windows to proceed to the following step. Some operations may take some time."

print "Displaying Circuit.tif"
circuit = TIFF.open('Circuit.tif').read_image()
plt.imshow(circuit)

plt.show()

print "Converting Circuit.tif to 8 bit..."
circuit8 = to8bit(circuit)

print "Displaying Circuit.tif converted to 8 bit"

plt.imshow(circuit8, cmap='gray', vmin=0, vmax=255)
plt.show()

print "Quantizing Circuit.tif to 4 bit..."

circuit4 = numpy.zeros(circuit.shape[0:2],numpy.uint8)

for y,line in enumerate(circuit8):
  for x,pix in enumerate(line):
    circuit4[y][x] = circuit8[y][x]/16

print "Displaying Circuit.tif quantized to 4 bit by dividing the values"
plt.imshow(circuit4, cmap='gray', vmin=0, vmax=255)
plt.show()

print "Displaying Circuit.tif quantized to 4 bit by knocking off the 4 lower bits"
plt.imshow(circuit4, cmap='gray', vmin=0, vmax=15)
plt.show()


print "Displaying Coins.tif"

coins = TIFF.open('Coins.tif').read_image()
plt.imshow(coins)
plt.show()

print "Converting Coins.tif to 8 bit..."
coins8 = to8bit(coins)

print "Displaying Coins.tif converted to 8 bit"

plt.imshow(coins8, cmap='gray', vmin=0, vmax=255)
plt.show()

coinshalf = numpy.zeros(tuple(cs/2 for cs in coins8.shape),numpy.uint8)

print "Subsamplig Coins.tif to half width and height..."

for y,line in enumerate(coinshalf):
  for x,pix in enumerate(line):
    coinshalf[y][x] = coins8[y*2][x*2]

coinsquarter = numpy.zeros(tuple(cs/4 for cs in coins8.shape),numpy.uint8)

print "Subsamplig Coins.tif to a quarter of the width and height..."

for y,line in enumerate(coinsquarter):
  for x,pix in enumerate(line):
    coinsquarter[y][x] = coins8[y*4][x*4]

plt.imshow(coinsquarter, cmap='gray', vmin=0, vmax=255)
plt.show()



code.interact(local=locals())

