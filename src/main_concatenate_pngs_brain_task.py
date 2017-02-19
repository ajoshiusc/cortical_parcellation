#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 18 17:54:44 2017

@author: ajoshi
"""

import sys
from PIL import Image


x_offset = 0
for ind in range(1200):
    im1name = '/home/ajoshi/coding_ground/cortical_parcellation/src/\
rest_vs_hand_after_rot_%d_d.png' % ind
    im2name = '/home/ajoshi/coding_ground/cortical_parcellation/src/\
timing_%d.png' % ind    

    im1 = Image.open(im1name)
    im2 = Image.open(im2name)
    im2 = im2.resize((im1.size[0], im2.size[1]), Image.ANTIALIAS)
    new_im = Image.new('RGB',
                       (im1.size[0], im1.size[1]+im2.size[1]))
    new_im.paste(im1, (0, 0))
    new_im.paste(im2, (0, im1.size[1]))
    x_offset += im1.size[0]

    im1name = '/home/ajoshi/coding_ground/cortical_parcellation/src/\
catimg_%d.png' % ind
    new_im.save(im1name)