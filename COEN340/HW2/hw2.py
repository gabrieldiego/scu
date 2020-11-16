#!/usr/bin/python

import numpy

from scipy import misc
import matplotlib.pyplot as plt

from libtiff import TIFF

import code

def plot(im):
  plt.imshow(im, cmap='gray', vmin=0, vmax=255)
  plt.show()

def to8bit(im):
  im8 = numpy.zeros(im.shape[0:2],numpy.uint8)
  
  for y,line in enumerate(im):
    for x,pix in enumerate(line):
      im8[y][x] = sum(pix)/3

  return im8

def ave_nxn(im,n):
  nm1 = n - 1
  im8 = numpy.zeros(tuple(s-nm1 for s in im.shape)[0:nm1],numpy.uint8)
  
  for y,line in enumerate(im8):
    for x,pix in enumerate(line):
      sum_matrix = 0.0
      for i in range(n):
        for j in range(n):
          sum_matrix += im[y+j][x+i]
      im8[y][x] = sum_matrix / (n*n)

  return im8

def center_nxn(im,n):
  nm1 = n - 1
  np1 = n + 1

  im8 = numpy.zeros(tuple(s-nm1 for s in im.shape)[0:nm1],numpy.uint8)
  
  for y,line in enumerate(im8):
    for x,pix in enumerate(line):
      sum_matrix = 0.0
      for i in range(n):
        for j in range(n):
          sum_matrix += im[y+j][x+i]

        if(i == nm1/2):
          sum_matrix *= 2
        if(j == nm1/2):
          sum_matrix *= 2

      im8[y][x] = sum_matrix / (np1*np1)

  return im8

def sharp_nxn(im,n):
  nm1 = n - 1
  im8 = numpy.zeros(tuple(s-nm1 for s in im.shape)[0:nm1],numpy.uint8)
  
  for y,line in enumerate(im8):
    for x,pix in enumerate(line):
      sum_matrix = 0.0
      for i in range(n):
        for j in range(n):
          mul = 0          
          if(i == nm1/2):
            mul = -1
          if(j == nm1/2):
            mul = -1

          if(i == nm1/2 and j == nm1/2):
            mul = 2*nm1 + 1

          sum_matrix += mul * im[y+j][x+i]

      im8[y][x] = sum_matrix

  return im8


def histogram(im):

  hist_table = numpy.zeros(256,numpy.uint64)
  hist_cumul = numpy.zeros(256,numpy.uint64)

  total_pix=im.shape[0]*im.shape[1]

  for y,line in enumerate(im):
    for x,pix in enumerate(line):
      hist_table[pix] += 1

  for i,h in enumerate(hist_table):
    if[i>0]:
      hist_cumul[i] = hist_cumul[i-1] + hist_table[i]
    else:
      hist_cumul[0] = hist_table[0]

  return (hist_table,hist_cumul,total_pix)


def plot_histogram(hist,total_pix):
  hist_height = 100

  plot_hist = numpy.zeros((hist_height,256),numpy.uint8)

  for y,line in enumerate(plot_hist):
    for x,pix in enumerate(line):
      if(hist[x]*hist_height*1.0/total_pix > hist_height*1.0-y):
        plot_hist[y][x] = 255
      else:
        plot_hist[y][x] = 0

  return plot_hist

def norm_cum_hist(hist_cumul):
  ch = ((hist_cumul - hist_cumul.min()) * 255.0) / (hist_cumul.max() - hist_cumul.min())
  return ch.astype('uint8')

def eq_hist(im, nch):
  eq_im = numpy.zeros((im.shape[0],im.shape[1]),numpy.uint8)

  for y,line in enumerate(im):
    for x,pix in enumerate(line):
      eq_im[y][x] = nch[pix] 

  return eq_im

print "Close the windows to proceed to the following step. Some operations may take some time."

images = {}
images8 = {}

all_images = ['Coins.tif', 'Apple.tif', 'Gearwheel.tif']
all_images = ['Apple.tif']

for image in all_images:

  print "Displaying", image

  images[image] = TIFF.open(image).read_image()
  plt.imshow(images[image])
  plt.show()

  print "Converting", image, "to 8 bit..."
  images8[image] = to8bit(images[image])

  print "Displaying", image, "converted to 8 bit"

  plt.imshow(images8[image], cmap='gray', vmin=0, vmax=255)
  plt.show()

if(False):

  image = 'Coins.tif'

  print "Calculating the histogram of", image, "..."

  (coins_hist,coins_cumul,total_pix) = histogram(images8[image])

  print "Displaying", image, "histogram"


  plt.plot(coins_hist)
  plt.show()

  print "Displaying", image, "cumulative histogram"

  plt.plot(coins_cumul)
  plt.show()

  print "Displaying", image, "normalized cumulative histogram"

  coins_nch = norm_cum_hist(coins_cumul)

  plt.plot(coins_nch)
  plt.show()

  print "Performing histogram equalization on", image

  coins_eq = eq_hist(images8[image],coins_nch)

  plot(coins_eq)

  print "Displaying", image, "cumulative histogram after normalization"

  plt.plot(histogram(coins_eq)[1])
  plt.show()


all_images = ['Apple.tif', 'Gearwheel.tif']
all_images = ['Apple.tif']

#for image in all_images:
for image in []:
  f_sizes = [3,5,7]
#  f_sizes = [3]

  print "Displaying", image, "converted to 8 bit for reference before applying filters"

  plot(images8[image])
    
  for n in f_sizes:
    print "Calculating average", n, "x",n, "of", image, "..."

    image_ave_nxn = ave_nxn(images8[image],n)

    print "Displaying average", n, "x",n, "of", image

    plot(image_ave_nxn)


    print "Calculating average", n, "x",n, "of", image, "..."

    std_devs = [5,10,25]

    for s in std_devs:
      image_gauss = numpy.random.normal(images8[image],s).astype('uint8')

      print "Displaying image with gaussian noise of std dev of ", s

      plot(image_gauss)

      print "Calculating average", n, "x",n, "of", image, "with gaussian noise..."

      image_ave_nxn = ave_nxn(image_gauss,n)

      print "Displaying average", n, "x",n, "of", image, "after gaussian noise applied"

      plot(image_ave_nxn)


for image in all_images:
  f_sizes = [3,5,7]
#  f_sizes = [3]

  print "Displaying", image, "converted to 8 bit for reference before applying filters"

  plot(images8[image])
    
  for n in f_sizes:
    print "Calculating sharp", n, "x",n, "of", image, "..."

    image_sharp_nxn = sharp_nxn(images8[image],n)

    print "Displaying sharp", n, "x",n, "of", image

    plot(image_sharp_nxn)


    print "Calculating sharp", n, "x",n, "of", image, "..."

    std_devs = [5,10,25]

    for s in std_devs:
      image_gauss = numpy.random.normal(images8[image],s).astype('uint8')

      print "Displaying image with gaussian noise of std dev of ", s

      plot(image_gauss)

      print "Calculating sharp", n, "x",n, "of", image, "with gaussian noise..."

      image_sharp_nxn = sharp_nxn(image_gauss,n)

      print "Displaying sharp", n, "x",n, "of", image, "after gaussian noise applied"

      plot(image_sharp_nxn)



#    print "Calculating center average", n, "x",n, "of", image, "..."

#    image_center_nxn = center_nxn(images8[image],n)

#    print "Displaying center average", n, "x",n, "of", image

#    plt.imshow(image_center_nxn, cmap='gray', vmin=0, vmax=255)
#    plt.show()



code.interact(local=locals())

