#!/usr/bin/env python


import h5py
import numpy as np



def main():
    fh5 = h5py.File('test.h5')

    nx = 100
    ny = 11
    nz = 8
    dx = 0.1
    x  = np.arange(nx+1)*dx
    x0 = x.max()*0.5

    Bx = np.random.randn(nx+1,ny+1,nz+1)*0.01
    By = np.random.randn(nx+1,ny+1,nz+1)*0.01
    Bz = np.random.randn(nx+1,ny+1,nz+1)*0.02

    for i in np.arange(nx+1):
        By[i,...] += np.tanh((x[i] - x0))

    B = np.zeros((nx+1,ny+1,nz+1,3))
    B[...,0] = Bx
    B[...,1] = By
    B[...,2] = Bz

    fh5.create_dataset('Bx', shape=Bx.shape, dtype=Bx.dtype, data=Bx)
    fh5.create_dataset('By', shape=By.shape, dtype=Bx.dtype, data=By)
    fh5.create_dataset('Bz', shape=Bz.shape, dtype=Bx.dtype, data=Bz)

    fh5.create_dataset('Bvec', shape=B.shape, dtype=B.dtype, data=B)

    fh5.close()

if __name__ == '__main__':
    main()
