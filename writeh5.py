#!/usr/bin/env python


import h5py
import numpy as np



def main():
    fh5 = h5py.File('test.h5')

    nx = 100
    ny = 20
    nz = 40
    dx = 20./float(nx)
    dy = 10/float(ny)
    dz = 40/float(nz)

    print(dx,dy,dz)

    x  = np.arange(nx+1)*dx
    y  = np.arange(ny+1)*dy
    x0 = x.max()*0.5

    Bx = np.random.randn(nz+1,ny+1,nx+1)*0.01
    By = np.random.randn(nz+1,ny+1,nx+1)*0.01
    Bz = np.random.randn(nz+1,ny+1,nx+1)*0.02

    for i in np.arange(ny+1):
        Bx[:,i,:] += np.tanh((y[i] - y.max()*0.5))

    for i in np.arange(nx+1):
        By[:,:,i] += -0.1 + (0.2)*0.5*(1+np.tanh(x[i]-x0))

    B = np.zeros((nz+1,ny+1,nx+1,3))
    B[...,0] = Bx
    B[...,1] = By
    B[...,2] = Bz

    fh5.create_dataset('Bx', shape=Bx.shape, dtype=Bx.dtype, data=Bx)
    fh5.create_dataset('By', shape=By.shape, dtype=By.dtype, data=By)
    fh5.create_dataset('Bz', shape=Bz.shape, dtype=Bz.dtype, data=Bz)

    fh5.create_dataset('Bvec', shape=B.shape, dtype=B.dtype, data=B)

    fh5.close()

if __name__ == '__main__':
    main()
