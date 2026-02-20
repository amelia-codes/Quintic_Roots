#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 13:18:01 2024
"""
"""
TO DO:
- fix orientation of title and legend
- import numpy add moving controls and box over original graph
- fix animation, have axis change with graph
- make legend dots bigger while keeping graph the same
- show zoom in box
- make colors more distinct?
"""
import matplotlib.pyplot as plt
#from matplotlib.animation import FuncAnimation as animation
from itertools import product
import matplotlib.colors as mcolors
import numpy as np
from ferrari import quartic_roots, cubic_roots 
#import pygame
#import pygame.freetype
#generate all combinations of size 5


#subplots
numbers = np.arange(-8,9)
perms = list(p for p in product(numbers, repeat=5))
fig, ax = plt.subplots(2, 2)
#plot 1
ax1 = plt.subplot(1,2,1)
a = (-4,4)
ax1.set_xlim(a)
ax1.set_ylim(a)


#plot 2
ax2 = plt.subplot(1,2,2)
ax1.patch.set_facecolor('xkcd:black')
ax2.patch.set_facecolor('xkcd:black')
#fig.patch.set_facecolor('xkcd:charcoal')
#get roots 
qroots = [] 
croots = []
sroots = []
siroots = []
#sort real and complex
for i in perms:
    if i[0] != 0:
        qroots += [i]
    elif i[1] !=0:
        croots += [i[1:]]
    elif i[2] != 0:
        j = np.array(i[2:])
        s = np.roots(j) 
        sroots.append(s)
    elif i[3] != 0:
        j = np.array(i[3:])
        si = np.roots(j)
        siroots.append(si)
        pass
    else: pass #fails
n = .2
croots = np.array(croots)
q = quartic_roots(qroots)
ax1.scatter(np.real(q),np.imag(q), c = mcolors.XKCD_COLORS['xkcd:magenta'], s=n, label='Quartic Roots')
c = cubic_roots(croots)
ax1.scatter(np.real(c),np.imag(c), c = mcolors.XKCD_COLORS['xkcd:pink'], s=n, label='Cubic Roots')
s = np.array(sroots)
ax1.scatter(np.real(s),np.imag(s), c = mcolors.XKCD_COLORS['xkcd:plum'], s=n, label='Quadratic Roots')
si = np.array(siroots)
ax1.scatter(np.real(si),np.imag(si), c = mcolors.XKCD_COLORS['xkcd:lavender'], s=n, label="Simple Roots") 
fig.legend(loc='lower center',ncol=2) #make bigger dots
fig.suptitle('Roots of Polynomials with Coefficients in Range (-9,9)',fontsize = 20)
fig.set_figheight(6.5) #fix title orientation
fig.set_figwidth(10)

#ax2.set_xlim(b)
#def init():
b = (0,1)
c = (0,1)
ax2.set_ylim(c)
ax2.set_xlim(b)
ax2.scatter(np.real(q),np.imag(q), c = mcolors.XKCD_COLORS['xkcd:magenta'], s=n, label='Quartic Roots')
ax2.scatter(np.real(c),np.imag(c), c = mcolors.XKCD_COLORS['xkcd:pink'], s=n, label='Cubic Roots')
ax2.scatter(np.real(s),np.imag(s), c = mcolors.XKCD_COLORS['xkcd:plum'], s=n, label='Quadratic Roots')
ax2.scatter(np.real(si),np.imag(si), c = mcolors.XKCD_COLORS['xkcd:lavender'], s=n, label="Simple Roots") 
plt.show()
#return ax2
#ax2.set_ylim(c)

"""
ani = animation.FuncAnimation(fig, animate, frames=10)
plt.show()
#transform matrix

def update(i,b,c):
     minx,maxx = i/5,i/5
     b.set_data([minx,minx],[0,1])
     ax2.set_xlim(minx,maxx)
     return fig
 
ani = animation(fig,update,frames=100,interval=100,blit=True)
"""
running = True
while running:
    keys = pygame.key.get_pressed()
    i = 0
    if keys[pygame.K_x]:
        i+=1
        ax2.set_xlim(b+i)
    if keys[pygame.K_s]:
        i+=1
        ax2.set_ylim(a+i)

