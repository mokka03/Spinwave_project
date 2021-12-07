"""Optimize a focusing model"""
import torch
import os
import spintorch
import numpy as np
import matplotlib as mpl
from spintorch.utils import tic, toc, stat_cuda
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.io import savemat



mpl.use('Agg') # uncomment for plotting without GUI
mpl.rcParams['figure.figsize'] = [8.0, 6.0]
mpl.rcParams['figure.dpi'] = 600

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

t0 = t1 = tic()


"""Parameters"""
dt = 10e-12     # timestep (s)
dx = 25e-9      # discretization (m)
dy = 25e-9      # discretization (m)
dz = 25e-9      # discretization (m)
B1 = 50e-3      # training field multiplier (T)
# B1 = 0      # training field multiplier (T)
B0 = 176e-3+50e-3     # bias field (T)
Bt = 1e-3       # excitation field (T)
Ms = 140e3      # saturation magnetization (A/m)
alpha = 5e-3    # damping coefficient ()
gamma_LL = 1.7595e11    # gyromagnetic ratio (rad/Ts)
A_exch = 3.65e-12       # exchange coefficient (J/m)

f1 = 3e9         # source frequency (Hz)
timesteps = 10 # number of timesteps for wave propagation

nx = 200        # size x
ny = 200        # size y

'''Directories'''
basedir = 'binary_pictures/'
plotdir = 'plots/' + basedir
if not os.path.isdir(plotdir):
    os.makedirs(plotdir)
savedir = 'models/' + basedir
if not os.path.isdir(savedir):
    os.makedirs(savedir)    

'''Geometry, sources, probes, model definitions'''
geom = spintorch.WaveGeometryMs((nx, ny), dx, dy, dz, Ms, B0)
src = spintorch.WaveLineSource(10, 0, 10, ny-1, dim=1)
probes = []
Np = 21  # number of probes
for p in range(Np):
    probes.append(spintorch.WaveIntensityProbeDisk(nx-25, int(ny*(p+1)/(Np+1)), 3))
model = spintorch.WaveCell(geom, dt, Ms, gamma_LL, alpha, A_exch, src, probes)

dev = torch.device('cuda')  # 'cuda' or 'cpu'
print('Running on', dev)
model.to(dev)   # sending model to GPU/CPU


'''Define the source signal'''
t = torch.arange(0, timesteps*dt, dt, device=dev).unsqueeze(0).unsqueeze(2) # time vector
X = Bt*torch.sin(2*np.pi*f1*t)  # sinusoid signal at f1 frequency, Bt amplitude

INPUTS = X  # here we could cat multiple inputs
OUTPUTS = torch.tensor([4]).to(dev) # desired output

'''Define optimizer and criterion'''
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss(reduction='sum')

'''Load checkpoint'''
epoch_init = 4 # select previous checkpoint (-1 = don't use checkpoint)
epoch = epoch_init
if epoch_init>=0:
    checkpoint = torch.load('models/focus/' + 'model_e%d.pt' % (epoch_init))
    epoch = checkpoint['epoch']
    loss_iter = checkpoint['loss_iter']
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
else:
    loss_iter = []

'''Round'''
round2 = 1000   # Round to 3 decimals
round2 = 100   # Round to 2 decimals
rho = model.geom.Msat.detach().cpu().numpy().transpose()/Ms
rho = np.round(rho*round2)/round2
print("Min: ", np.min(rho))
print("Max: ", np.max(rho))

unique = np.unique(rho, return_counts=False)
print("Unique_num: ", unique.size)

'''Binary pictures'''
k = 0
for i in range(int(0.95*round2),int((1.05+1/round2)*round2), 1):
     if i != round2:
        x = np.where(rho==i/round2, 0, 1)
        plt.imsave(plotdir+'%d.png' % (k), x, cmap=cm.gray)
        k += 1



'''MuMax3 code for 3 decimals'''
f = open("for_mumax.txt", "w")
# for i in range(0,100):
#     f.write('defregion(%d,imageShape("C:/Users/mauch/Desktop/Önlab/Spintorch/Aktiv/plots/binary_pictures/%d.png"))' % (255-i,i))
#     f.write("             //%f" % (0.95+i/1000))
#     f.write("\n")
#     f.write("Msat.setregion(%d, Ms*%f)" % (255-i,0.95+i/1000))
#     f.write("\n"+"\n")

'''Save .mat file'''
# trained_Msat = model.geom.Msat
# savemat("trained_Msat.mat", {"Msat": trained_Msat.detach().cpu().numpy().transpose()})