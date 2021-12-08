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
dt = 18e-12     # timestep (s)
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
timesteps = 5600 # number of timesteps for wave propagation

nx = 500        # size x
ny = 500        # size y

'''Directories'''
basedir = 'f_change_3132MHz_to_3567MHz/'
plotdir = 'plots/' + basedir
if not os.path.isdir(plotdir):
    os.makedirs(plotdir)
savedir = 'models/' + basedir
if not os.path.isdir(savedir):
    os.makedirs(savedir)    

out_list = []
f_values = []

for f_MHz in range(3132,3567,15):
    f1 = f_MHz*1e6
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
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss(reduction='sum')


    '''Load checkpoint'''
    epoch_init = -1 # select previous checkpoint (-1 = don't use checkpoint)
    epoch = epoch_init
    if epoch_init>=0:
        checkpoint = torch.load('models/focus/' + 'model_e%d.pt' % (epoch_init))
        epoch = checkpoint['epoch']
        loss_iter = checkpoint['loss_iter']
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    else:
        loss_iter = []


    '''Plot wave propagation'''
    print("Wave propagation for plotting")

    epoch = 0
    with torch.no_grad():
        u_field = model(INPUTS, output_fields=True)
        # stat_cuda('after plot propagating')
        
        t1 = toc(t0, t1)

        # magnetization = u_field[0,]
        # magnetization = magnetization / model.geom.Msat
        # savemat("Matlab plotter/magnetization_without_relax.mat", {"magnetization_without_relax": magnetization.to(torch.device("cpu")).numpy().transpose()})
        
        # timesteps = u_field.size(1)
        Nx, Ny = 2, 2
        times = np.ceil(np.linspace(timesteps/Nx/Ny, timesteps, num=Nx*Ny))-1

        if f_MHz == 3342:            
            spintorch.plot.geometry(model, cbar=True, saveplot=True, epoch=epoch, plotdir=plotdir)
        
        spintorch.plot.field_snapshot(model, u_field, [timesteps-1],label=False)
        plt.gcf().savefig(plotdir+'field_snapshot_%de6Hz.png' % (f_MHz))

        u = spintorch.utils.normalize_power(model(INPUTS).sum(dim=1))    
        spintorch.plot.plot_output(u[0,], OUTPUTS[0]+1, f_MHz, "e6Hz", plotdir)
        outs = u[0,].detach().cpu().numpy()
        out = outs[int(Np/2)]
        out_list.append(out)
        f_values.append(f_MHz)
    stat_cuda('befor del')
    del u_field
    torch.cuda.empty_cache()
    stat_cuda('after del')
    print('---------------------------------------')


out_array = np.array(out_list)
f_array = np.array(f_values)
f_out = np.zeros((2,f_array.shape[0]))
f_out[0,] = f_array
f_out[1,] = out_array

savemat("f_out.mat", {"f_out": f_out})