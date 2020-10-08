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

def ave3x3(im):
  im8 = numpy.zeros(tuple(s-2 for s in im.shape)[0:2],numpy.uint8)
  
  for y,line in enumerate(im8):
    for x,pix in enumerate(line):
      im8[y][x] = (0.0+im[y][x]+im[y][x+1]+im[y][x+2] + im[y+1][x]+im[y+1][x+1]+im[y+1][x+2] + im[y+2][x]+im[y+2][x+1]+im[y+2][x+2])/9

  return im8


def gamma_table(gamma):
  table = numpy.zeros(256,numpy.uint8)

  for i in range(256) :
    table[i] = pow(i/255.0,1.0/gamma) * 255.0

  return table

def histogram(im):

  hist_height = 500

  hist = numpy.zeros((hist_height,256),numpy.uint8)
  hcumul = numpy.zeros((hist_height,256),numpy.uint8)

  hist_table = [0]*256

  total_pix=im.shape[0]*im.shape[1]

  for y,line in enumerate(im):
    for x,pix in enumerate(line):
      hist_table[pix] += 1

  percent = [0.0]*256
  cumul = [0.0]*256

  for i,h in enumerate(hist_table):
    percent[i] = (1.0*h*hist_height)/total_pix
    if[i>0]:
      cumul[i] = cumul[i-1] + percent[i]
    else:
      cumul[0] = percent[0]

  for y,line in enumerate(hist):
    for x,pix in enumerate(line):
      if(percent[x] > hist_height-y):
        hist[y][x] = 255
      else:
        hist[y][x] = 0

      if(cumul[x] > hist_height-y):
        hcumul[y][x] = 255
      else:
        hcumul[y][x] = 0

#  code.interact(local=locals())

  return (hist,hcumul)


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

print "Subsampling Coins.tif to half width and height..."

for y,line in enumerate(coinshalf):
  for x,pix in enumerate(line):
    coinshalf[y][x] = coins8[y*2][x*2]

plt.imshow(coinshalf, cmap='gray', vmin=0, vmax=255)
plt.show()

print "Average 3x3 then subsampling Coins.tif to half width and height..."

coinsave3x3 = ave3x3(coins8)
coinsave3x3half = numpy.zeros(tuple(cs/2 for cs in coinsave3x3.shape),numpy.uint8)

for y,line in enumerate(coinsave3x3half):
  for x,pix in enumerate(line):
    coinsave3x3half[y][x] = coinsave3x3[y*2][x*2]

plt.imshow(coinsave3x3half, cmap='gray', vmin=0, vmax=255)
plt.show()



coinsquarter = numpy.zeros(tuple(cs/4 for cs in coins8.shape),numpy.uint8)

print "Subsampling Coins.tif to a quarter of the width and height..."

for y,line in enumerate(coinsquarter):
  for x,pix in enumerate(line):
    coinsquarter[y][x] = coins8[y*4][x*4]

plt.imshow(coinsquarter, cmap='gray', vmin=0, vmax=255)
plt.show()

print "Average 3x3 and subsampling twice Coins.tif..."

coinsave3x3_2 = ave3x3(coinsave3x3half)
coinsave3x3quart = numpy.zeros(tuple(cs/2-1 for cs in coinsave3x3half.shape),numpy.uint8)

for y,line in enumerate(coinsave3x3quart):
  for x,pix in enumerate(line):
    coinsave3x3quart[y][x] = coinsave3x3_2[y*2][x*2]

plt.imshow(coinsave3x3quart, cmap='gray', vmin=0, vmax=255)
plt.show()



for gamma in [0.5, 0.8, 1.5, 2.0]:

  print gamma
  gt = gamma_table(gamma)

  print "Calculating gamma correction of Circuit.tif of ", gamma, " ..."

  circuit_gamma = numpy.zeros(circuit.shape[0:2],numpy.uint8)

  for y,line in enumerate(circuit_gamma):
    for x,pix in enumerate(line):
      circuit_gamma[y][x] = gt[circuit8[y][x]]

  print "Displaying Circuit.tif with gamma correction"

  plt.imshow(circuit_gamma, cmap='gray', vmin=0, vmax=255)
  plt.show()

  print "Calculating gamma correction of Coins.tif of ", gamma, " ..."

  coins_gamma = numpy.zeros(coins.shape[0:2],numpy.uint8)

  for y,line in enumerate(coins_gamma):
    for x,pix in enumerate(line):
      coins_gamma[y][x] = gt[coins8[y][x]]

  print "Displaying Coins.tif with gamma correction"

  plt.imshow(coins_gamma, cmap='gray', vmin=0, vmax=255)
  plt.show()

print "Calculating the histogram of Coins.tif..."

(coins_hist,coins_cumul) = histogram(coins8)

print "Displaying Coins.tif histogram"

plt.imshow(coins_hist, cmap='gray', vmin=0, vmax=255)
plt.show()

print "Displaying Coins.tif cumulative histogram"

plt.imshow(coins_cumul, cmap='gray', vmin=0, vmax=255)
plt.show()

print "Calculating the histogram of Circuit.tif..."

(circuit_hist,circuit_cumul) = histogram(circuit8)

print "Displaying Circuit.tif histogram"

plt.imshow(circuit_hist, cmap='gray', vmin=0, vmax=255)
plt.show()

print "Displaying Circuit.tif cumulative histogram"

plt.imshow(circuit_cumul, cmap='gray', vmin=0, vmax=255)
plt.show()

code.interact(local=locals())

