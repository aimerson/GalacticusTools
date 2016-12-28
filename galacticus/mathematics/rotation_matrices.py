#! /usr/bin/env python

import numpy as np


def Rx(r,angle):
    """
    Rx(): Rotates (x,y,z) about x-axis
          by a specifed angle.

            (   1     0      0     )
            (                      )
       Rx = (   0   cos(a) -sin(a) )
            (                      )
            (   0  -sin(a)  cos(a) )
          
    USAGE: xnew,ynew,znew = Rx(r,angle)

           r = (x,y,z) : position(s) of point(s)
           angle : rotation angle (in radians)
           xnew,ynew,znew: new position(s)

           x,y,z can be scalars or arrays
          
    """
    x = r[0]; y = r[1]; z = r[2]
    xnew = np.copy(x)
    ynew = np.array(y)*np.cos(angle) - np.array(y)*np.sin(angle)
    znew = np.array(y)*np.sin(angle) + np.array(z)*np.cos(angle)
    if(np.isscalar(x)):
       xnew = np.asscalar(xnew)
       ynew = np.asscalar(ynew)
       znew = np.asscalar(znew)
    return xnew,ynew,znew


def Ry(r,angle):
    """
    Ry(): Rotates (x,y,z) about y-axis
          by a specifed angle.

            ( cos(a)   0  sin(a) )
            (                    )
       Ry = (   0      1    0    )
            (                    )
            ( -sin(a)  0  cos(a) )
          
    USAGE: xnew,ynew,znew = Ry(r,angle)

           r = (x,y,z) : position(s) of point(s)
           angle : rotation angle (in radians)
           xnew,ynew,znew: new position(s)

           x,y,z can be scalars or arrays
          
    """
    x = r[0]; y = r[1]; z = r[2]
    xnew = np.array(x)*np.cos(angle) + np.array(z)*np.sin(angle)
    ynew = np.copy(y)
    znew = -1.0*np.array(x)*np.sin(angle) + np.array(z)*np.cos(angle) 
    if(np.isscalar(x)):
       xnew = np.asscalar(xnew)
       ynew = np.asscalar(ynew)
       znew = np.asscalar(znew)
    return xnew,ynew,znew


def Rz(r,angle):
    """
    Rz(): Rotates (x,y,z) about z-axis
          by a specifed angle.

            ( cos(a) -sin(a)  0  )
            (                    )
       Rz = ( sin(a)  cos(a)  0  )
            (                    )
            (  0       0      1  )
          
    USAGE: xnew,ynew,znew = Rz(r,angle)

           r = (x,y,z) : position(s) of point(s)
           angle : rotation angle (in radians)
           xnew,ynew,znew: new position(s)

           x,y,z can be scalars or arrays
          
    """
    x = r[0]; y = r[1]; z = r[2]
    xnew = np.array(x)*np.cos(angle) - np.array(y)*np.sin(angle)
    ynew = np.array(x)*np.sin(angle) + np.array(y)*np.cos(angle)
    znew = np.copy(z)
    if(np.isscalar(x)):
       xnew = np.asscalar(xnew)
       ynew = np.asscalar(ynew)
       znew = np.asscalar(znew)
    return xnew,ynew,znew


def R2D(x,y,angle):
    """
    R2D(): Rotates (x,y) about in 2D plane 
           according to right-hand ruleby
           a specifed angle.

       R2D = ( cos(a) -sin(a) )
             ( sin(a)  cos(a) )
  
    USAGE: xnew,ynew = R2D(x,y,angle)

           x,y : position(s) of point(s)
           angle : rotation angle (in radians)
           xnew,ynew: new position(s)

           x,y can be scalars or arrays 
    """
    xnew = np.array(x)*np.cos(angle) + np.array(y)*np.sin(angle)
    ynew = np.array(x)*np.sin(angle) + np.array(y)*np.cos(angle)
    if(np.isscalar(x)):
       xnew = np.asscalar(xnew)
       ynew = np.asscalar(ynew)
    return xnew,ynew
