"""Optimize a focusing model"""
import torch
import os
import spintorch
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from spintorch.utils import tic, toc, stat_cuda
from scipy.io import savemat


mpl.use('Agg') # uncomment for plotting without GUI
mpl.rcParams['figure.figsize'] = [8.0, 6.0]
mpl.rcParams['figure.dpi'] = 600

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

t0 = t1 = tic()


"""Parameters"""
dt = 20e-12     # timestep (s)
dx = 100e-9      # discretization (m)
dy = 100e-9      # discretization (m)
dz = 70e-9      # discretization (m)
B1 = 50e-3      # training field multiplier (T)
B0 = 283e-3     # bias field (T)
Bt = 1e-3       # excitation field (T)
Ms = 1.3567e5      # saturation magnetization (A/m)
alpha = 1e-4    # damping coefficient ()
gamma_LL = 1.7595e11    # gyromagnetic ratio (rad/Ts)
A_exch = 3.65e-12       # exchange coefficient (J/m)

f1 = 3.3423e9         # source frequency (Hz)
timesteps = 5000 # number of timesteps for wave propagation

nx = 500        # size x
ny = 500        # size y

'''Directories'''
basedir = 'binary_pictures_binary_lens/'
plotdir = 'plots/' + basedir
if not os.path.isdir(plotdir):
    os.makedirs(plotdir)
savedir = 'models/' + basedir
if not os.path.isdir(savedir):
    os.makedirs(savedir)    

'''Geometry, sources, probes, model definitions'''
# geom = spintorch.WaveGeometryFreeForm((nx, ny), dx, dy, dz, B0, B1, Ms)
geom = spintorch.WaveGeometryMs((nx, ny), dx, dy, dz, Ms, B0)
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
X = Bt*torch.sin(2*np.pi*f1*t)  # sinusoid signal at f1 frequency, Bt amplitude

INPUTS = X  # here we could cat multiple inputs
OUTPUTS = torch.tensor([int(Np/2)]).to(dev) # desired output

'''Define optimizer and criterion'''
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
criterion = torch.nn.CrossEntropyLoss(reduction='sum')


'''Load checkpoint'''
epoch_init = 10 # select previous checkpoint (-1 = don't use checkpoint)
epoch = epoch_init
if epoch_init>=0:
    checkpoint = torch.load('models/focus_binary_lens/' + 'model_e%d.pt' % (epoch_init))
    epoch = checkpoint['epoch']
    loss_iter = checkpoint['loss_iter']
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
else:
    loss_iter = []


Msat = model.geom.Msat.detach()

'''Save .mat file'''
# savemat("matlab/Msat.mat", {"Msat": Msat.to(torch.device("cpu")).numpy().transpose()})

Msat = model.geom.Msat.detach().cpu().numpy().transpose()
print(np.min(Msat))
print(np.max(Msat))

'''Binary pictures'''

import random

Msat_difference = Msat-np.min(Msat)
Msat_difference_norm = (Msat_difference)/np.max(Msat_difference)

Msat_binary = np.ones_like(Msat)
damping_with = 20
for i in range(damping_with-1,Msat.shape[0]-damping_with):
    for j in range(damping_with-1,Msat.shape[1]-damping_with):
        if Msat_difference_norm[i,j] > random.uniform(0,1):
            Msat_binary[i,j] = 0

plt.imsave(plotdir+'%d.png' % (0), Msat_binary, cmap=cm.gray)