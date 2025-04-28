# -*- coding: utf-8 -*-
"""
Created on Fri Apr 25 21:15:34 2025

@author: kaiwe
"""

import magpylib as magpy

# Create a Cuboid magnet with magnetic polarization
# of 1 T pointing in x-direction and sides of
# 1,2 and 3 cm respectively (notice the use of SI units).

cube = magpy.magnet.Cuboid(polarization=(1,0,0), dimension=(0.01,0.02,0.03))

# Create a Sensor for measuring the field

sensor = magpy.Sensor()

# By default, the position of a Magpylib object is
# (0,0,0) and its orientation is the unit rotation,
# given by a scipy rotation object.

print(cube.position)                   # -> [0. 0. 0.]
print(cube.orientation.as_rotvec())    # -> [0. 0. 0.]

# Manipulate object position and orientation through
# the respective attributes (move 10 mm and rotate 45 deg):

from scipy.spatial.transform import Rotation as R
cube.position = (0.01,0,0)
cube.orientation = R.from_rotvec((0,0,45), degrees=True)

print(cube.position)                            # -> [0.01 0.   0.  ]
print(cube.orientation.as_rotvec(degrees=True)) # -> [0. 0. 45.]

# Apply relative motion with the powerful `move`
# and `rotate` methods.
sensor.move((-0.01,0,0))
sensor.rotate_from_angax(angle=-45, axis='z')

print(sensor.position)                            # -> [-0.01  0.    0.  ]
print(sensor.orientation.as_rotvec(degrees=True)) # -> [ 0.  0. -45.]

# Use the `show` function to view your system
# through Matplotlib, Plotly or Pyvista backends.

magpy.show(cube, sensor, backend='matplotlib')

# Compute the B-field for some positions.

points = [(0,0,-.01), (0,0,0), (0,0,.01)] # in SI Units (m)
B = magpy.getB(cube, points)

print(B.round(2)) # -> [[ 0.26  0.07  0.08]
                  #     [ 0.28  0.05  0.  ]
                  #     [ 0.26  0.07 -0.08]] # in SI Units (T)

# Compute the H-field at the sensor.

H = magpy.getH(cube, sensor)

print(H.round()) # -> [51017. 24210.     0.] # in SI Units (A/m)
