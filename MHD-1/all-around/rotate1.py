#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt

def getBorder(mesh, b):
    boundaryEdges = mesh.edges[mesh.Boundaries[list(mesh.Boundaries.keys())[b]]]
    return np.concatenate((boundaryEdges[:, 0], boundaryEdges[[-1], 1]))-1

def getBorders(mesh):
    return [np.concatenate((mesh.edges[mesh.Boundaries[bc]][:, 0],
                            mesh.edges[mesh.Boundaries[bc]][[-1], 1]))-1 for bc in mesh.Boundaries.keys()]

def plotBorders(ax, mesh, c1, lwidth=0.4, bwidth=0.6, current_color='red'):
    for i, border in enumerate(getBorders(mesh)):
        vertices = mesh.vertices[border][:,:2].T
        if i == 0: color = "red"
        elif i == 1: color = c1
        elif i == 2: color = "blue"
        else: color = "black"       
        if abs(vertices[0][0]) > 1e-6 or abs(vertices[0][1]) > 1e-6 or abs(vertices[0][-1]) > 1e-6:
            ax.plot(vertices[0], vertices[1], c=color, linewidth=bwidth, zorder=30)
        elif current_color is not None: ax.plot(vertices[0], vertices[1], c=current_color, linewidth=lwidth)

def plotMagnets(ax, s, a, b, c, bwidth=0.6, fill=True):
    ct, st = np.cos(np.pi*s), np.sin(np.pi*s)
    x0, y0 = a/2, b*ct/2
    x1, y1, x2, y2, x3, y3 = x0+c*ct/2, y0+c*st/2, x0+c*ct, y0+c*st, x0+b*st, y0-b*ct
    if x2 < 0.0: return False
    x4, y4, x5, y5 = x3+c*ct/2, y3+c*st/2, x3+c*ct, y3+c*st
    if s == 1.0 and 2*c == a:
        if fill:
            ax.fill([0.0, -x0, -x3, 0.0, 0.0], [y1, y0, y3, y4, y1], c="limegreen", zorder=10)
            ax.fill([0.0, +x0, +x3, 0.0, 0.0], [y1, y2, y5, y4, y1], c="chocolate", zorder=10)
        ax.plot([0.0, -x0, -x3, 0.0], [y1, y0, y3, y4], c="green", linewidth=bwidth, zorder=20)
        ax.plot([0.0, +x0, +x3, 0.0], [y1, y2, y5, y4], c="brown", linewidth=bwidth, zorder=20)
        return True        
    if fill:
        ax.fill([-x1, -x0, -x3, -x4, -x1], [y1, y0, y3, y4, y1], c="limegreen", zorder=10)
        ax.fill([-x1, -x2, -x5, -x4, -x1], [y1, y2, y5, y4, y1], c="chocolate", zorder=10)
        ax.fill([+x1, +x0, +x3, +x4, +x1], [y1, y0, y3, y4, y1], c="chocolate", zorder=10)
        ax.fill([+x1, +x2, +x5, +x4, +x1], [y1, y2, y5, y4, y1], c="limegreen", zorder=10)
    ax.plot([-x1, -x0, -x3, -x4], [y1, y0, y3, y4], c="green", linewidth=bwidth, zorder=20)
    ax.plot([-x1, -x2, -x5, -x4], [y1, y2, y5, y4], c="brown", linewidth=bwidth, zorder=20)
    ax.plot([+x1, +x0, +x3, +x4], [y1, y0, y3, y4], c="brown", linewidth=bwidth, zorder=20)
    ax.plot([+x1, +x2, +x5, +x4], [y1, y2, y5, y4], c="green", linewidth=bwidth, zorder=20)
    return True


# In[2]:


import copy
import io
from PIL import Image

def plotP1F(p1f, ax, niso=41, vmin=None, vmax=None, lwidth=None, color=None, **kwargs):
    x, y, v = p1f.mesh.vertices[:, 0], p1f.mesh.vertices[:, 1], p1f.sol
    triang = p1f.mesh.triangles[:, :-1]-1
    if vmin is None: vmin = min(v)
    if vmax is None: vmax = max(v)
    levels = kwargs.get('levels', np.linspace(vmin, vmax, niso))[1:-1]
    plot = ax.tricontour(x, y, triang, v, cmap=None, extend="both", levels=levels, linewidths=lwidth, colors=color)

def plotVhIh(ax, vhr, ihr, lwidth=0.4, bwidth=0.6, vquantiles=40, iquantiles=40,
             potential_color='red', current_color='cyan'):
    vhl, ihl = copy.deepcopy(vhr), copy.deepcopy(ihr)
    vhl.mesh.vertices, ihl.mesh.vertices = vhl.mesh.vertices*[[-1.0, 1.0, 1.0]], ihl.mesh.vertices*[[-1.0, 1.0, 1.0]]
    if potential_color is not None:
        plotP1F(vhl, ax, niso=vquantiles+1, lwidth=lwidth, color=potential_color)
        plotP1F(vhr, ax, niso=vquantiles+1, lwidth=lwidth, color=potential_color)
    if current_color is not None:
        plotP1F(ihl, ax, niso=iquantiles//2+1, lwidth=lwidth, color=current_color)
        plotP1F(ihr, ax, niso=iquantiles//2+1, lwidth=lwidth, color=current_color)
    plotBorders(ax, ihl.mesh, c1='green', lwidth=lwidth, bwidth=bwidth, current_color=current_color)
    plotBorders(ax, ihr.mesh, c1='brown', lwidth=lwidth, bwidth=bwidth, current_color=current_color)


# In[3]:


import magpylib as magpy
import pyvista as pv

def getMagnets(s, a, b, c, l, Br):
    xm, ym = a/2+b/2*np.sin(s*np.pi)+c/2*np.cos(s*np.pi), c/2*np.sin(s*np.pi)
    b1, c1 = 0.999999*b, 0.999999*c if s != 1 else c
    magnetl = magpy.magnet.Cuboid(polarization=(1000*Br, 0, 0), dimension=(c1, b1, l), position=[-xm, ym, 0.0])
    magnetl.rotate_from_angax(-s*180, 'z')
    magnetr = magpy.magnet.Cuboid(polarization=(1000*Br, 0, 0), dimension=(c1, b1, l), position=[+xm, ym, 0.0])
    magnetr.rotate_from_angax(+s*180, 'z')
    return magnetl+magnetr

def getMagMod(magnets, XLIM, YLIM, nx, ny):
    tsx, tsy = XLIM[0]+(XLIM[1]-XLIM[0])*(np.arange(nx)+0.5)/nx, YLIM[0]+(YLIM[1]-YLIM[0])*(np.arange(ny)+0.5)/ny
    grid = np.array([[(x,y,0) for x in tsx] for y in tsy]) # slow Python loop
    return grid[:,:,0], grid[:,:,1], np.sqrt(np.sum(np.square(magnets.getB(grid)), axis=2))

def getPVgrid(magnets, XLIM, YLIM, nx, ny):
    dx, dy, dz = (XLIM[1]-XLIM[0])/nx, (YLIM[1]-YLIM[0])/ny, 1
    ox, oy, oz = XLIM[0]+dx/2, YLIM[0]+dy/2, 0
    pvgrid = pv.ImageData(dimensions=(nx, ny, 1), spacing=(dx, dy, dz), origin=(ox, oy, oz))
    pvgrid["B"] = magpy.getB(magnets, pvgrid.points)
    return pvgrid


# In[4]:


def borderCurrent(eh, b, sigma):
    mesh, sol = eh.mesh, eh.sol
    border = getBorder(mesh, b)
    # Edge vectors
    ve = mesh.vertices[border][1:,:2]-mesh.vertices[border][:-1,:2]  # mm
    # Electric field at edges
    ee = (sol[border][1:]+sol[border][:-1])/2  # V/mm
    # Total current
    return sigma*np.sum(np.cross(ve, ee)) # A/m

def pressureGradient(eh, magnets, sigma, z = 0):
    vertices = eh.mesh.vertices*[[1, 1, 0]]+[[0, 0, z]]  # Replace the index assciated to the 2D vertex by a z value
    return np.cross(sigma*eh.sol, magnets.getB(vertices)[:,:2])  # Pa/m, getB returns in mT and eh is in V/mm

def pressureDifference(eh, magnets, sigma, nv=40):
    l = magnets[0].dimension[2]
    B = np.average(np.array([magnets.getB(eh.mesh.vertices*[[1, 1, 0]]+[[0, 0, z]])[:,:2]
                             for z in (np.arange(nv//2)+0.5)*l/nv]), axis=0)/1000  # getB returns in mT
    return l*np.cross(sigma*eh.sol, B)  # Pa, l is in mm and eh is in V/mm

def getInt2dForm(mesh):
    # To get the integral of a quantity defined at mesh vertices, sum over it multiplied by the returned value
    vertices, triangles = mesh.vertices[:,:2], mesh.triangles[:,:3]-1
    int2dForm = np.zeros(vertices.shape[0])
    for triangle in triangles:
        # Add one third of the triangle' area to each of its vertex
        int2dForm[triangle] += np.linalg.det(np.append(vertices[triangle], [[1], [1], [1]], axis = 1))/6
    return int2dForm/1000000   # m²


# In[5]:


from matplotlib.tri import Triangulation, LinearTriInterpolator

def my_eval(mesh, sol, xy, mask=True):
    if mask:
        triObj = Triangulation(mesh.vertices[:,0], mesh.vertices[:,1], triangles=mesh.triangles[:, :-1]-1)
    else:
        triObj = Triangulation(mesh.vertices[:,0], mesh.vertices[:,1])
    triInterp = LinearTriInterpolator(triObj, sol)
    return triInterp(xy[:,0],xy[:,1])


# In[6]:


import math
import cv2
from pyfreefem import FreeFemRunner
from pymedit import P1Function
from closestreamlines import (get_pv_pairs, merge_or_close_pairs)

calib = [None, None, None]

def experiment(svalues, calib,
               a=20.0, # mm, width of duct and electrodes
               b=20.0, # mm, height of duct and width of magnets
               c=10.0, # mm, thickness of magnets
               l=100.0, # mm, length of duct, electrodes and magnets
               Br=1.48, # T, Residual induction (or flux density), 1.48 for N52 grade
               sigma=20.0, # S/m cobductivity of the solution (brine case)
               bp = 1000.0, # W, target for the electrical power, ignorin electrolysis
               va=10.0, # V, voltage between electrodes for the FEM simulation
               nx = 1920, ny = 1080, # resolution of the output image or video
               figWidth = 10.0, # width of the figure, height derived from XLIM and YLIM
               XLIM = [-80.0, 80.0], YLIM = [-45.0, 45.0], # field of view in space coordinates
               magnetic_color='orange', potential_color=None, current_color='cyan',
               isomagnetic_color='white', isopressure_color='white',
               mquantiles=40, # number of quantiles for the magnetic lines (not really quantiles here)
               vquantiles=40, # number of quantiles for the potential lines
               iquantiles=40, # number of quantiles for the current lines
               pquantiles0=10,  # number of quantiles for the pressure integration, related to the duct case
               showMagnets=True, fillMagnets=True, showPressure=False, itext=[0.0, 0.5, 1.0],
               lwidth = 0.4, bwidth = 0.6, png=None, mp4=None, fps=30, extraFrames=[],
               calibratemNpW=False, calibrateBounds=False, calibrateSumThrust=False, linearPressure=False):

    if mp4 is not None:
        out = cv2.VideoWriter(mp4, cv2.VideoWriter_fourcc(*'mp4v'), fps, (1920,1080))
    
    rotate1 = FreeFemRunner("rotate1.edp")

    for s in svalues:
        print("%.03f" % s, end="  ")

        # Initialize magnets
        magnets = getMagnets(s, a, b, c, l, Br)
        # get magnetic field module on image grid, full domain
        x, y, m = getMagMod(magnets, XLIM, YLIM, nx, ny)
        # Half-domain for magnetic line extraction
        XLIM2 = [0, max(XLIM[1], -XLIM[0])]
        nx2, ny2 = math.ceil(2*nx*XLIM2[1]/(XLIM[1]-XLIM[0])), 2*ny
        # get magnetic field module on image grid, right half-domain, double resolution, pyvista format
        pvgrid = getPVgrid(magnets, XLIM2, YLIM, nx2, ny2)

        # Compute magnetic field lines using pyvista integrator
        ct, st = np.cos(np.pi*s), np.sin(np.pi*s)
        pvline = [(a/2+c*ct/2, b*ct/2+c*st/2, 0), (a/2+b*st+c*ct/2, -b*ct/2+c*st/2, 0)]
        pvseeds = pv.Line(*pvline, mquantiles)
        strl = pvgrid.streamlines_from_source(pvseeds, vectors="B", max_time=1000, max_steps=5000,
            initial_step_length=0.01, integration_direction="both", integrator_type=45)        
        # Extract raw line pairs associated to seeds from the pyvista streamlines object
        pairs = get_pv_pairs(strl, pvseeds)
        # Close or merge streamline pairs
        rlines = merge_or_close_pairs(pairs, verbose=-1)
        llines = [line*np.array([[-1, 1]]) for line in rlines]
        lines = llines+rlines

        # Set figure layout
        fig = plt.figure()
        fig.set_size_inches(figWidth, figWidth*(YLIM[1]-YLIM[0])/(XLIM[1]-XLIM[0]))
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_xlim(XLIM)
        ax.set_ylim(YLIM)
        ax.set_axis_off()
        fig.add_axes(ax)
        
        if not showPressure:
            # Plot magnetic field module
            plt.imshow(m, vmax=1000*Br, extent=[XLIM[0], XLIM[1], YLIM[0], YLIM[1]], origin='lower', zorder=-10)
            # Plot magnetic field module isolines
            if isomagnetic_color is not None:
                CS = plt.contour(x, y, m/1000, levels=np.linspace(0.1,0.5,5), colors=isomagnetic_color, linewidths=lwidth)
                plt.contour(x, y, m/1000, levels=np.linspace(0.6,1.0,5), colors=isomagnetic_color, linewidths=lwidth)
                if s in itext:
                    ax.clabel(CS, inline=True, fontsize=5)

        # Magnets layout, abort if magnets overlap
        if showMagnets:
            magnetsOK = plotMagnets(ax, s, a, b, c, bwidth=0.6, fill=fillMagnets)
            if not magnetsOK:
                plt.close(fig)
                continue

        # Potential and current lines, right half-domain, call to FreeFEM++
        exports = rotate1.execute({'A':a, 'B':b, 'R':240, 'S':s})
        vhr = P1Function(exports['Tvh'],exports['vh1[]'])
        ihr = P1Function(exports['Tih'],exports['ih1[]'])
        print(vhr.mesh.vertices.shape[0], end="  ")
        print(ihr.mesh.vertices.shape[0], end="  ")
        ehr = -vhr.gradientP1()  # V/mm

        # Get the current for a voltage va of 10 V
        ip, im = 2*borderCurrent(ehr, 0, sigma)*(l/1000), 2*borderCurrent(ehr, 2, sigma)*(l/1000)  # A
        ipm, ierr = (ip-im)/2, 200*abs(im+ip)/(ip-im)
        
        # Get normalization factor for a total power of bp
        rbp = np.sqrt(bp/(ipm*va))
        
        int2dFormr = getInt2dForm(ehr.mesh)  # m²
        # gradPr = pressureGradient(ehr, magnets, sigma)  # Pa/m
        # thrust = 2*np.sum(int2dFormr*gradPr)*(l/1000)
        diffPr = pressureDifference(ehr, magnets, sigma)  # Pa
        thrust = 2*np.sum(int2dFormr*diffPr) # N

        if showPressure: # Works only for full field, bug with zoom
            xy = np.array([np.tile(np.linspace(XLIM[0], XLIM[1], nx), ny),
                           np.repeat(np.linspace(YLIM[0], YLIM[1], ny), nx)]).T 
            minDiffPr = np.where(diffPr > 0, diffPr, np.inf).min()  # Spurious negative values sometimes in corners
            diffPrP = rbp*np.maximum(diffPr, minDiffPr)
            diffPrl10 = np.log(diffPrP)/np.log(10)
            # print(np.min(diffPr), np.max(diffPr), end="  ")
            # print(np.nanmin(diffPrl10), np.nanmax(diffPrl10), end="  ")
            meshl, meshr = copy.deepcopy(vhr.mesh), vhr.mesh
            meshl.vertices = meshr.vertices*[[-1.0, 1.0, 1.0]]
            vim = np.concatenate([my_eval(meshl, diffPrl10, xy, mask=True).reshape(ny, nx)[:,:nx//2],
                                  my_eval(meshr, diffPrl10, xy, mask=True).reshape(ny, nx)[:,nx//2:]], axis=1)    

            # calib[1] stores the bounds for vim display, set at the first call with pressure and s = 1.0
            if calibrateBounds:
                calib[1] = [np.nanmin(vim), np.nanmax(vim)]
            # Display pressure map with a linear scale
            if linearPressure:
                ax.set_facecolor('silver') # Does not work
                ax.imshow(np.exp(vim), interpolation='none', extent=[XLIM[0], XLIM[1], YLIM[0], YLIM[1]], zorder=-10,
                          cmap='plasma', origin='lower')
            # Display pressure map with a log scale
            elif calib[1] is not None:
                ax.set_facecolor('silver') # Does not work
                ax.imshow(vim, interpolation='none', extent=[XLIM[0], XLIM[1], YLIM[0], YLIM[1]], zorder=-10,
                          vmin=calib[1][0], vmax=calib[1][1], cmap='plasma', origin='lower')
            
            ind = np.argsort(diffPrP)
            # print(diffPrP[ind[0]], diffPrP[ind[-1]])
            cumThrust = np.cumsum((int2dFormr*diffPrP)[ind])
            # print(cumThrust[-1])
            
            # calib[2] stores the reference sumThrust for the duct, set at the first call with pressure and s = 0.0
            if calibrateSumThrust:
                calib[2] = cumThrust[-1]
            if calib[2] is not None:
                pquantiles = math.floor(cumThrust[-1]/calib[2]*pquantiles0)
                targets = np.linspace(cumThrust[-1]-pquantiles*calib[2]/pquantiles0, cumThrust[-1], pquantiles+1)
                plevels = np.log(np.interp(targets, cumThrust, diffPrP[ind]))/np.log(10)
                # print(plevels[1:-1])
                if isopressure_color is not None:
                    plt.contour(x, y, vim, levels=plevels, colors=isopressure_color, linewidths=lwidth,
                                negative_linestyles='solid')
                    plt.contour(x, y, vim, levels=plevels[-11:-10], colors=isopressure_color, linewidths=1.5*lwidth)
            
            plotBorders(ax, meshl, c1='green', current_color=None)
            plotBorders(ax, meshr, c1='brown', current_color=None)
        else:
            plotVhIh(ax, vhr, ihr, lwidth=lwidth, vquantiles=vquantiles, iquantiles=iquantiles,
                     potential_color=potential_color, current_color=current_color)

        thrustbp, ibp, vbp = thrust*rbp, ipm*rbp, va*rbp
        print("%8.03f mN" % (1000*thrustbp), end="  ")
        
        mNpW = 1000*thrustbp/bp # mN/W
        print("%7.03f V  %7.03f A  %6.02f %%  %7.03f mN/W" % (vbp, ibp, ierr, mNpW), end="")

        # calib[0] stores the reference trhust/power value in mN/W
        if calibratemNpW:
            calib[0] = mNpW
        print("    %5.03f gain" % (mNpW/calib[0]) if calib[0] is not None else "")

        # Print pressure bounds if calibrated
        if calibrateBounds:
                print(calib[1], 10**(calib[1][1]-calib[1][0]))

        # Magnetic lines
        if not showPressure and magnetic_color is not None:
            for line in lines:
                xl, yl = line.T
                plt.plot(xl, yl, c=magnetic_color, linewidth=lwidth)
        
        # Get the figure as an image array
        img_buf = io.BytesIO()
        plt.savefig(img_buf, format='png', dpi=nx/figWidth, bbox_inches='tight', pad_inches = 0)
        plt.close(fig)
        im = np.asarray(Image.open(img_buf))[:,:,:3]
        img_buf.close()
        
        # Write mp4 and/or png files
        if mp4 is not None:
            out.write(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
            if s in extraFrames:
                for k in range(math.ceil(fps)):
                    out.write(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        if png is not None:
            im = Image.fromarray(im)
            im.save(png)

    if mp4 is not None:
        out.release()
        cv2.destroyAllWindows()


# In[7]:


# Calibrations
# Get reference trhust/power value in mN/W, should be done with s = 0.0 for duct reference
experiment([0.0], calib, calibratemNpW=True)
# Calibrate SumThrust
experiment([0.0], calib, showPressure=True, calibrateSumThrust=True)
# Get bounds for pressure display with a log scale, should be done with s = 1.0
experiment([1.0], calib, showPressure=True, calibrateBounds=True)


# In[ ]:


# Magnetic and electric lines, images
experiment([0.0], calib, isomagnetic_color=None, png='duct-nomod.png')
experiment([0.0], calib, png='duct.png')
experiment([0.0], calib, png='duct-quadrant.png', XLIM = [-3.0, 29.0], YLIM = [-2.0, 16.0])
experiment([0.5], calib, png='flat.png')
experiment([1.0], calib, isomagnetic_color=None, png='all-around-nomod.png')
experiment([1.0], calib, png='all-around.png')
experiment([1.0], calib, png='all-around-quadrant.png', XLIM = [0.0, 48.0], YLIM = [0.0, 27.0])


# In[ ]:


# Pressure fields, images
experiment([0.0], calib, showPressure=True, linearPressure=True, isopressure_color=None,
           png='duct-pressure-linear-no-lines.png')
experiment([0.0], calib, showPressure=True, linearPressure=True, png='duct-pressure-linear.png')
experiment([1.0], calib, showPressure=True, png='all-around-pressure.png')
experiment([1.0], calib, showPressure=True, isopressure_color=None, png='all-around-pressure-no-lines.png')
experiment([0.0], calib, showPressure=True, png='duct-pressure.png')
experiment([0.5], calib, showPressure=True, png='flat-pressure.png')


# In[ ]:


# Not supported so far, does not produce what is expected
# experiment([1.0], calib, showPressure=True, png='all-around-quadrant-pressure.png',
#            XLIM = [0.0, 48.0], YLIM = [0.0, 27.0])


# In[ ]:


# Magnetic and electric lines, videos
experiment(np.linspace(0.0, 0.5, 251), calib, mp4='from_duct_to_flat.mp4')
experiment(np.linspace(0.5, 1.0, 251), calib, mp4='from_flat_to_all_around.mp4')
experiment(np.linspace(0.0, 1.0, 501), calib, extraFrames=[0.0, 0.5, 1.0], mp4='from_duct_to_all_around.mp4')


# In[ ]:


# Pressure fields, videos
experiment(np.linspace(0.0, 1.0, 401), calib, showPressure=True, extraFrames=[0.0, 0.5, 1.0],
           mp4='from_duct_to_all_around_pressure.mp4')


# In[ ]:


for aff in np.sqrt(np.linspace(1.0, 2.5, 16)):
    print("%5.3f" % (aff*aff))
    experiment([0.5], calib, a=20/aff, b=20/aff, c=10*aff)


# In[ ]:


# Optimal Value
aff = math.sqrt(1.6)
experiment([0.5], calib, a=20/aff, b=20/aff, c=10*aff, png='flat-16.png')


# In[ ]:


# William Fraser's magnets' ratio
aff = math.sqrt(2.4)
experiment([0.5], calib, a=20/aff, b=20/aff, c=10*aff, png='flat-24.png')


# In[ ]:


# William Fraser's magnets' and electrodes' ratio
aff = math.sqrt(2.4)
experiment([0.5], calib, a=22/aff, b=20/aff, c=10*aff, png='flat-24-22.png')


# In[ ]:


# William Fraser's magnets' and electrodes' ratio with magnet length correction
aff = math.sqrt(2.4)
aff2 = math.sqrt(1/2.2)
a=22/aff
b=20/aff
c=10*aff
l=100
print(a, b, c, l, b*c*l)
a=22/aff/aff2
b=20/aff/aff2
c=10*aff/aff2
l=100*aff2*aff2
print(a, b, c, l, b*c*l)
# experiment([0.5], calib, a=22/aff/aff2, b=20/aff/aff2, c=10*aff/aff2, l=100*aff2*aff2, png='flat-24-22-50.png')

