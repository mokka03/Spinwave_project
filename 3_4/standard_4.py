"""Optimize a focusing model"""
import torch
import os
import spintorch
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from spintorch.utils import tic, toc, stat_cuda
from scipy.io import savemat


mpl.use('Agg') # uncomment for plotting without GUI
mpl.rcParams['figure.figsize'] = [8.0, 6.0]
mpl.rcParams['figure.dpi'] = 600

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

t0 = t1 = tic()


"""Parameters"""
dt = 5e-13     # timestep (s)
dx = 500e-9/128      # discretization (m)
dy = 125e-9/32      # discretization (m)
dz = 3e-9      # discretization (m)
B1 = 50e-3      # training field multiplier (T)
B0 = 226.5e-3     # bias field (T)
Bt = 0       # excitation field (T)
Ms = 800e3      # saturation magnetization (A/m)
alpha = 0.02    # damping coefficient ()
gamma_LL = 1.7595e11    # gyromagnetic ratio (rad/Ts)
A_exch = 13e-12       # exchange coefficient (J/m)

f1 = 1.5e9         # source frequency (Hz)
timesteps = 2000 # number of timesteps for wave propagation

nx = 128        # size x
ny = 32        # size y

'''Directories'''
basedir = 'standard_4/'
plotdir = 'plots/' + basedir
if not os.path.isdir(plotdir):
    os.makedirs(plotdir)
savedir = 'models/' + basedir
if not os.path.isdir(savedir):
    os.makedirs(savedir)    

'''Geometry, sources, probes, model definitions'''
geom = spintorch.WaveGeometryFreeForm((nx, ny), dx, dy, dz, B0, B1, Ms)
# geom = spintorch.WaveGeometryMs((nx, ny), dx, dy, dz, Ms, B0)
src = spintorch.WaveLineSource(10, 0, 10, ny-1, dim=1)
probes = []
Np = 21  # number of probes
for p in range(Np):
    probes.append(spintorch.WaveIntensityProbeDisk(nx-25, int(ny*(p+1)/(Np+1)), 5))
model = spintorch.WaveCell(geom, dt, Ms, gamma_LL, alpha, A_exch, src, probes)

dev = torch.device('cuda')  # 'cuda' or 'cpu'
print('Running on', dev)
model.to(dev)   # sending model to GPU/CPU


'''Define the source signal'''
t = torch.arange(0, timesteps*dt, dt, device=dev).unsqueeze(0).unsqueeze(2) # time vector
# X = Bt*torch.sin(2*np.pi*f1*t)  # sinusoid signal at f1 frequency, Bt amplitude
X = Bt*torch.sin(0*2*np.pi*f1*t)
# print(X)

INPUTS = X  # here we could cat multiple inputs
OUTPUTS = torch.tensor([int(Np/2)]).to(dev) # desired output

'''Plot wave propagation'''
print("Wave propagation for plotting")

epoch = 0

with torch.no_grad():
    u_field, M, mx_sum, my_sum, mz_sum = model(INPUTS, output_fields=True)
    stat_cuda('after plot propagating')

    # print(mx_sum.shape)

    magnetization = M[0,]
    savemat("Matlab plotter/magnetization.mat", {"magnetization": magnetization.to(torch.device("cpu")).numpy().transpose()})

    savemat("Matlab plotter/mx_sum.mat", {"mx_sum": mx_sum.to(torch.device("cpu")).numpy()})
    savemat("Matlab plotter/my_sum.mat", {"my_sum": my_sum.to(torch.device("cpu")).numpy()})
    savemat("Matlab plotter/mz_sum.mat", {"mz_sum": mz_sum.to(torch.device("cpu")).numpy()})
    

    # t1 = toc(t0, t1)
    
    # # timesteps = u_field.size(1)
    # Nx, Ny = 2, 2
    # times = np.ceil(np.linspace(timesteps/Nx/Ny, timesteps, num=Nx*Ny))-1

    # spintorch.plot.field_snapshot(model, u_field, times, Ny=Ny)
    # plt.gcf().savefig(plotdir+'field_4snapshots_epoch%d.png' % (epoch))
    
    # spintorch.plot.total_field(model, u_field,cbar=False)
    # plt.gcf().savefig(plotdir+'total_field_epoch%d.png' % (epoch))
    
    # spintorch.plot.field_snapshot(model, u_field, [timesteps-1],label=False)
    # plt.gcf().savefig(plotdir+'field_snapshot_epoch%d.png' % (epoch))
    
    # spintorch.plot.geometry(model, cbar=True, saveplot=True, epoch=epoch, plotdir=plotdir)