#!/usr/bin/python

import numpy

from scipy import misc
from scipy import ndimage
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt

from libtiff import TIFF

import code

def plot(im,f=""):
  if(f):
    plt.imsave(f, im, cmap='gray', vmin=0, vmax=255)
  else:
    plt.imshow(im, cmap='gray', vmin=0, vmax=255)
    plt.show()

def to8bit(im):
  im8 = numpy.zeros(im.shape[0:2],numpy.uint8)
  
  for y,line in enumerate(im):
    for x,pix in enumerate(line):
      im8[y][x] = sum(pix)/3

  return im8

def make_png_filename(original_name, suffix):
  return original_name.lower().split(".")[0] + "_" + suffix +".png"

def gaussian_kernel(n,sigma):
  nm1 = n - 1
  np1 = n + 1
  mid = nm1 / 2

  i = numpy.zeros((n,n))

  i[mid,mid] = 1

  gaussian_kernel = gaussian_filter(i, sigma=sigma)

#  gaussian_kernel /= gaussian_kernel[mid,mid]

  return gaussian_kernel

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

def convo(im,f):
  nm1 = len(f) - 1
  np1 = len(f) + 1

  print len(f)
  print nm1
  print f

  n = len(f)

  im_pad = numpy.pad(im,nm1,'constant')

  print im.shape
  print im_pad.shape

  im_out = numpy.zeros(im.shape,numpy.uint8)

  for y,line in enumerate(im_out):
    for x,pix in enumerate(line):
      sum_matrix = 0.0
      for i in range(n):
        for j in range(n):
          sum_matrix += im_pad[y-nm1/2+j][x-nm1/2+i]*f[n-1-j,n-1-i]

      im_out[y][x] = sum_matrix

  return im_out

print "Close the windows to proceed to the following step. Some operations may take some time."

images = {}
images8 = {}

all_images = ['Coins.tif', 'Apple.tif', 'Gearwheel.tif']
#all_images = []

for image in all_images:

  images[image] = TIFF.open(image).read_image()

  print "Converting", image, "to 8 bit..."
  images8[image] = to8bit(images[image])

  print "Displaying", image, "converted to 8 bit"

  plot(images8[image],make_png_filename(image,"8bit"))

f_sizes = [3,5,7,9,11]

sigma_values = [1.0, 2.0, 5.0]


for f in f_sizes:
  for sigma in sigma_values:
    gk = gaussian_kernel(f,sigma)
    gk /= gk[(f-1)/2,(f-1)/2]  # Normalize to one for better display

    res = 200
    # Use kron to upscale the filter for a more visible image
    plot(numpy.kron(gk*255,numpy.ones((res/f,res/f))),"filter_" + str(f) + "x" + str(f) + "_sigma_" + str(sigma) + ".png")


f_sizes = [3,5,7,9,11]
#f_sizes = [3]

sigma_values = [2.0, 5.0, 20.0]
#sigma_values = [2.0]

for image in all_images:
  for sigma in sigma_values:
    print "Adding Gaussian noise with sigma", sigma, "to image", image, "..."

    image_gauss = numpy.random.normal(images8[image],sigma).clip(0,255).astype('uint8')

    filename = make_png_filename(image, "noise_sigma_" + str(sigma))

    print "Image with noise saved as ", filename

    plot(image_gauss,filename)

    for f in f_sizes:
      gk = gaussian_kernel(f,sigma)

      print "Applying filter", str(f) + "x" + str(f), "with sigma", sigma, "to image", image, "..."

      convo_image = convo(image_gauss,gk)

      filename = make_png_filename(image, str(f) + "x" + str(f) + "_sigma_" + str(sigma))

      plot(convo_image,filename)

      print "Result saved as", filename


code.interact(local=locals())

