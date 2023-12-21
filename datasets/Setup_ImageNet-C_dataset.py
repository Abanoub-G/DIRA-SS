import sys
import os
import wget
from zipfile import ZipFile
import pandas as pd
import tarfile

path = "./ImageNet/ImageNet-C/"

if not os.path.isdir(path):
    os.makedirs(path)
    print("created folder : ", path)

# ===========
# Dataset ImageNet-C
# ===========
# == Download Files 
urls = [('blur.tar', 'https://zenodo.org/record/2235448/files/blur.tar?download=1'),
        ('digital.tar', 'https://zenodo.org/record/2235448/files/digital.tar?download=1'),
        ('noise.tar', 'https://zenodo.org/record/2235448/files/noise.tar?download=1'),
        ('weather.tar', 'https://zenodo.org/record/2235448/files/weather.tar?download=1'),
        # ('extra.tar', 'https://zenodo.org/record/2235448/files/extra.tar?download=1')
        ]

# noise_sets = [["defocus_blur", "glass_blur", "motion_blur", "zoom_blur"],
#               []
               
#                ]

# counter =-1
for file, url in urls:
  if not os.path.isfile(path+ file):
    wget.download(url, path)
  else:
    print("This file exists: " + path+ file)



for file, url in urls:
  # if not os.path.isdir(path+file):
  my_tar = tarfile.open(path+file)
  my_tar.extractall(path)
  my_tar.close()
