import numpy as np
from scipy.fft import *
# import pandas as pd
import Pk_library as PKL
import MAS_library as MASL
import density_field_library as DFL
import contextlib

from nbodykit.source.catalog import ArrayCatalog

# import os, glob
# import camb
# import scipy

def high_res_pos(pos, mass, grid=128, BoxSize=1000., n=8):
    pos_scaled = pos*grid/BoxSize
    pos_scaled = pos_scaled.reshape(-1,1,3)
    
    pos_x = np.linspace(0.5/n,1-0.5/n,n)
    X, Y, Z = np.meshgrid(pos_x, pos_x, pos_x)
    pairs = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()]).reshape(1,-1,3)

    updated_pos_scaled = pairs+pos_scaled-0.5
    updated_pos = updated_pos_scaled*BoxSize/grid
    updated_pos = updated_pos.astype(np.float32).reshape(-1,3)
    updated_mass = np.array(n**3*[mass/n**3]).T.flatten()
    
    return updated_pos, updated_mass

def get_delta(pos, grid, W=0, mass=1, BoxSize=1000.0, MAS='CIC', verbose=True):
    
    """
    W: weights corresponding to each particle position. (i.e. W=0 for equal weighting to each particle, W=1, mass=mass for mass weighting)
    
    """
    
    # define 3D density field
    delta = np.zeros((grid,grid,grid), dtype=np.float32)
    
    # construct 3D density field
    if W:
        MASL.MA(pos, delta, BoxSize, MAS, W=mass, verbose=verbose)
    else:
        MASL.MA(pos, delta, BoxSize, MAS, verbose=verbose)
          
    
    # at this point, delta contains the effective number of particles in each voxel
    # now compute overdensity and density constrast
    delta /= np.mean(delta, dtype=np.float64)
    delta -= 1.0
    
    return delta

def get_delta_nbodykit(pos, grid, W=0, mass=1, BoxSize=1000.0, compensated=False, MAS='cic', interlaced=False):
    
    data = {'Position': pos}
    catalog = ArrayCatalog(data)

    # Paint onto mesh
    mesh = catalog.to_mesh(Nmesh=grid, BoxSize=BoxSize, compensated=compensated, window=MAS, interlaced=interlaced)

    # Get the density contrast field
    delta = mesh.compute()-1
    
    return delta

def compute_Pk(delta, BoxSize=1000.0, MAS='CIC', axis=0, verbose=True):    

    # compute power spectrum
    Pk = PKL.Pk(delta, BoxSize, axis=axis, MAS=MAS, threads=1, verbose=verbose)
    
    # 3D P(k)
    k       = Pk.k3D
    Pk0     = Pk.Pk[:,0] #monopole
    Pk2     = Pk.Pk[:,1] #quadrupole
    Pk4     = Pk.Pk[:,2] #hexadecapole

    return k, Pk0, Pk2, Pk4


def compute_XPk(delta1, delta2, BoxSize=1000.0, MAS=['CIC', 'CIC'], axis=0):    

    # compute cross-power spectrum    
    Pk = PKL.XPk([delta1,delta2], BoxSize, axis=axis, MAS=MAS, threads=1)
    
    # 3D P(k)
    k       = Pk.k3D
    Pk0_X  = Pk.XPk[:,0,0] #monopole of 1-2 cross P(k)
    Pk2_X  = Pk.XPk[:,1,0] #quadrupole of 1-2 cross P(k)
    Pk4_X  = Pk.XPk[:,2,0] #hexadecapole of 1-2 cross P(k)

    return k, Pk0_X, Pk2_X, Pk4_X

def get_Bk(delta, BoxSize, k1, k2, theta, MAS, threads):
    with contextlib.redirect_stdout(None):
        BBk = PKL.Bk(delta, BoxSize, k1, k2, theta, MAS, threads)
    Bk  = BBk.B     #bispectrum
    Qk  = BBk.Q     #reduced bispectrum
    k   = BBk.k     #k-bins for power spectrum
    Pk  = BBk.Pk    #power spectrum
    
    return Bk, Qk

def compute_Bk(delta, grid, BoxSize, MAS, threads):
    k12=np.linspace(2*np.pi/1000, np.pi*grid/2000,10)
    theta=np.linspace(0, np.pi, 10)
    Bk=np.array([])
    Qk=np.array([])
    for ind, k1 in enumerate(k12):
        for k2 in k12[ind:]:
            Bk_, Qk_ = get_Bk(delta, BoxSize, k1, k2, theta, MAS, threads);
            Bk = np.concatenate([Bk,Bk_])
            Qk = np.concatenate([Qk,Qk_])
    return Bk, Qk

def calculate_Qk(k, Pk0, Pk0_target, BoxSize=1000.0):
    
    I=np.zeros([len(k), len(k)])
    dz=1e-4
    z=np.arange(-1,1+dz,dz)
    for i,ki in enumerate(k):
        for j,qj in enumerate(k):

            k_in=np.sqrt(ki**2+qj**2-2*ki*qj*z)
            Pk_nl=np.interp(k_in, k, Pk0)
            if ki==qj:
                Pk_nl[-1]=0

            I[i,j]=8*np.pi**2*ki**2*qj**2*np.trapz(Pk_nl*z**2,z)
            
    I2=np.identity(len(k))
    for i, ki in enumerate(k):
        I2[i,i]*=ki**4

    V=BoxSize**3
    # A=(((2*np.pi)**3/V**2)*I+I2)
    A=(I/V+I2)
    B=Pk0_target-Pk0
    
    Qk=np.matmul(np.linalg.inv(A),B)
    
    return Qk


def get_phi(k, Qk, grid=128, BoxSize=1000.0, Rayleigh_sampling=1, seed=1, threads=1, verbose=True):
    
    # seed              = 1      #value of the initial random seed
    # Rayleigh_sampling = 1      #whether sampling the Rayleigh distribution for modes amplitudes
    # threads           = 1      #number of openmp threads
    # verbose           = True   #whether to print some information

    k, Qk = k.astype(np.float32), Qk.astype(np.float32)

    # generate a 3D Gaussian density field
    phi = DFL.gaussian_field_3D(grid, k, Qk, Rayleigh_sampling, seed,
                                  BoxSize, threads, verbose)

    return phi

def calculate_acc(phi, grid=128, BoxSize=1000.):
    phi_k = fftshift(fftn(phi))
    freq = fftshift(fftfreq(int(grid), BoxSize/grid)) # h/Mpc

    a_kx = np.zeros([grid,grid,grid], dtype=complex)
    a_ky = np.zeros([grid,grid,grid], dtype=complex)
    a_kz = np.zeros([grid,grid,grid], dtype=complex)
    for i in range(grid):
        for j in range(grid):
            for k in range(grid):
                a_kx[i,j,k]=2j*np.pi*freq[i]*phi_k[i,j,k]
                a_ky[i,j,k]=2j*np.pi*freq[j]*phi_k[i,j,k]
                a_kz[i,j,k]=2j*np.pi*freq[k]*phi_k[i,j,k]
                
    a_x=fftshift(ifftn(a_kx)).real
    a_y=fftshift(ifftn(a_ky)).real
    a_z=fftshift(ifftn(a_kz)).real
    
    return a_x, a_y, a_z

def calculate_acc_from_phi_k(phi_k, grid=128, BoxSize=1000.):
    freq = fftshift(fftfreq(int(grid), BoxSize/grid)) # h/Mpc

    a_kx = np.zeros([grid,grid,grid], dtype=complex)
    a_ky = np.zeros([grid,grid,grid], dtype=complex)
    a_kz = np.zeros([grid,grid,grid], dtype=complex)
    for i in range(grid):
        for j in range(grid):
            for k in range(grid):
                a_kx[i,j,k]=2j*np.pi*freq[i]*phi_k[i,j,k]
                a_ky[i,j,k]=2j*np.pi*freq[j]*phi_k[i,j,k]
                a_kz[i,j,k]=2j*np.pi*freq[k]*phi_k[i,j,k]
                
    a_x=fftshift(ifftn(a_kx)).real
    a_y=fftshift(ifftn(a_ky)).real
    a_z=fftshift(ifftn(a_kz)).real
    
    return a_x, a_y, a_z



# def correct_pos(pos, a_x, a_y, a_z, grid=128, BoxSize=1000.):
#     pos_scaled=pos*grid/BoxSize - 0.5
#     a_x, a_y, a_z = a_x*grid/BoxSize, a_y*grid/BoxSize, a_z*grid/BoxSize
#     for i, posit in enumerate(pos_scaled):
#         if np.ceil(posit[0])!=np.floor(posit[0]):
#             p1=int(np.ceil(posit[0])%grid)+(posit[0]-np.floor(posit[0]))*a_x[int(np.ceil(posit[0])%grid),int(np.floor(posit[1]%grid)),int(np.floor(posit[2]%grid))]
#             p0=int(np.floor(posit[0])%grid)+(np.ceil(posit[0])-posit[0])*a_x[int(np.floor(posit[0])%grid),int(np.floor(posit[1]%grid)),int(np.floor(posit[2]%grid))]
#             posit[0]=(posit[0]-np.floor(posit[0]))*p1+(np.ceil(posit[0])-posit[0])*p0
#         else:
#             posit[0]+=a_x[int(posit[0]%grid),int(np.floor(posit[1]%grid)),int(np.floor(posit[2]%grid))]

#         if np.ceil(posit[1])!=np.floor(posit[1]):
#             p1=int(np.ceil(posit[1])%grid)+(posit[1]-np.floor(posit[1]))*a_y[int(np.floor(posit[0]%grid)),int(np.ceil(posit[1])%grid),int(np.floor(posit[2]%grid))]
#             p0=int(np.floor(posit[1])%grid)+(np.ceil(posit[1])-posit[1])*a_y[int(np.floor(posit[0]%grid)),int(np.floor(posit[1])%grid),int(np.floor(posit[2]%grid))]
#             posit[1]=(posit[1]-np.floor(posit[1]))*p1+(np.ceil(posit[1])-posit[1])*p0
#         else:
#             posit[1]+=a_y[int(np.floor(posit[0]%grid)),int(posit[1]%grid),int(np.floor(posit[2]%grid))]

#         if np.ceil(posit[2])!=np.floor(posit[2]):
#             p1=int(np.ceil(posit[2])%grid)+(posit[2]-np.floor(posit[2]))*a_z[int(np.floor(posit[0]%grid)),int(np.floor(posit[1]%grid)),int(np.ceil(posit[2])%grid)]
#             p0=int(np.floor(posit[2])%grid)+(np.ceil(posit[2])-posit[2])*a_z[int(np.floor(posit[0]%grid)),int(np.floor(posit[1]%grid)),int(np.floor(posit[2])%grid)]
#             posit[2]=(posit[2]-np.floor(posit[2]))*p1+(np.ceil(posit[2])-posit[2])*p0
#         else:
#             posit[2]+=a_z[int(np.floor(posit[0]%grid)),int(np.floor(posit[1]%grid)),int(posit[2]%grid)]

#         pos_scaled[i]=[posit[0]%grid, posit[1]%grid, posit[2]%grid]

#     pos=((pos_scaled+0.5)%grid)*BoxSize/grid
    
#     return pos

def acc_at_particle_pos(pos, a_x, a_y, a_z, grid=128, BoxSize=1000.):
    grid_pos = pos*grid/BoxSize - 0.5
    acc = np.moveaxis(np.array([a_x, a_y, a_z]),0,-1) #= a_x*grid/BoxSize, a_y*grid/BoxSize, a_z*grid/BoxSize
    acc_at_particles = np.zeros_like(pos)
    
    i0 = np.floor(grid_pos).astype(int)%grid  # Base index per particle
    d = grid_pos - i0  # Fractional offset in cell
    
    for shift in np.ndindex(2, 2, 2):  # Loop over 8 corners
        w = (
            (1 - d[:, 0]) if shift[0] == 0 else d[:, 0]
        ) * (
            (1 - d[:, 1]) if shift[1] == 0 else d[:, 1]
        ) * (
            (1 - d[:, 2]) if shift[2] == 0 else d[:, 2]
        )

        # Get indices with shift, apply periodic boundary conditions
        ix = (i0[:, 0] + shift[0]) % grid
        iy = (i0[:, 1] + shift[1]) % grid
        iz = (i0[:, 2] + shift[2]) % grid
        
        acc_at_particles += w[:, None]*acc[ix, iy, iz]

    return acc_at_particles

def correct_pos(pos, a_x, a_y, a_z, grid=128, BoxSize=1000.):
    acc_at_particles = acc_at_particle_pos(pos, a_x, a_y, a_z, grid, BoxSize)
    pos += acc_at_particles
    pos = pos%BoxSize
    
    return pos
    



def MAS_correction(x, MAS_index):
    return (1.0 if (x==0.0) else pow(x/np.sin(x),MAS_index))

def caculate_Pk(delta, grid=128, BoxSize=1000., MAS_index=2):
    dft = fftshift(fftn(delta))
    freq = fftshift(fftfreq(grid, BoxSize/grid)) # h/Mpc
    
    rad = np.zeros(3*[grid])
    MAS_factor = np.zeros(3*[grid])
    for x in range(grid):
        for y in range(grid):
            for z in range(grid):
                rad[x,y,z]=np.sqrt(freq[x]**2+freq[y]**2+freq[z]**2)
                MAS_factor[x,y,z]=MAS_correction(np.pi*BoxSize*freq[x]/grid, MAS_index)*MAS_correction(np.pi*BoxSize*freq[y]/grid, MAS_index)*MAS_correction(np.pi*BoxSize*freq[z]/grid, MAS_index)
    
    dft=np.multiply(MAS_factor,dft)
    mat = np.multiply(dft,np.conjugate(dft))
    
    binlist=np.linspace(rad.min(), rad.max(), int(rad.max()*BoxSize))
    index=np.digitize(rad, binlist)
    kmodes=np.bincount(index.flatten(), rad.flatten(),minlength=int(rad.max()*BoxSize))[1:]/np.bincount(index.flatten())[1:]
    powerspec=np.bincount(index.flatten(), np.real(mat).flatten(),minlength=int(rad.max()*BoxSize))[1:]/np.bincount(index.flatten())[1:]
    kmodes=2*np.pi*kmodes
    powerspec=powerspec*BoxSize**3/grid**6

    return kmodes, powerspec