clear all, close all;

%% Magnonic Crystals: From Simple Models toward Applications Jaros?awW. K?os and Maciej Krawczyk (pg. 288)

% set magnetization directin here:
oop = 1;    % (1: out-of-plane, 0: in-plane)


%% constants
mu0 = 4*pi*1e-7;
if oop
    theta = 0;  % H0 angle, 0 if oop
else
    theta = pi/2;
end

d = 69e-9;      % film thickness
if oop
    Bext = 300e-3;   % applied field in T
else
    Bext = 30e-3;   % applied field in T
end
%% material parameters
Ms_CoFe = 1.4e6;
A_CoFe = 30E-12;
Ms_YIG = 1.4e5;
A_YIG = 3.65E-12;
Ms_Py = 8.6e5;
A_Py = 13E-12;
Bext = 292.79e-3;

Ms0 = Ms_YIG;
A = A_YIG;
gamma = 2*pi*28e9; % Hz/T gamma*mu0=2.21e5 used in OOMMF

%% Measured data
lambda = [7000 5000 4570 4215 4000 3875 3780]*1e-9;
% lambda = [];
ion_dose = [0 4 6 8 10 12 14]*1e12;

figure
plot(lambda*1e6,ion_dose,'o-')
xlabel('Wavelength [\mum]');
ylabel('Ion Dose [ions/cm^2]');
title({['Dependence of Ga+ ion dose'] ['upon wavelength']});

kxx_ = 2*pi./lambda;
Ms_ = [(Ms0*0.5):(Ms0*1.5)];  % If the Ms significantly differs from 
                              % 140 kA/m it's possible that chenge is
                              % needed in limits.
%% Dependence of saturation magnetization upon Ga+ ion dose
for kxx = kxx_
    for Ms = Ms_
        if oop
            H0 = Bext/mu0-Ms; %% effective field
        else
            H0 = Bext/mu0; %% effective field
        end
        
        omegaM = gamma*mu0*Ms;
        omegaHx = gamma*mu0*(H0+2*A/(mu0*Ms).*kxx.^2);
        Px = 1-(1-exp(-abs(kxx)*d))./(abs(kxx)*d);
        Phi2 = pi/2*0;
        omegax = sqrt(omegaHx.*(omegaHx+omegaM*(Px+sin(theta)^2*(1-Px.*(1+cos(Phi2).^2)+omegaM./omegaHx.*(Px.*(1-Px).*sin(Phi2).^2)))));
        if Ms == Ms_(1)
            f = omegax/2/pi;
        else
            f(end+1) = omegax/2/pi;
        end
    end
    [p,~,mu] = polyfit(f,Ms_,2);
    y1 = polyval(p,3.25e9,[],mu);
    if kxx == kxx_(1)
        Ms_levels = y1;
    else
        Ms_levels(end+1) = y1;
    end

    clear f;
end

%% polyfit
[p2,~,mu2] = polyfit(Ms_levels,ion_dose,2);
x2 = min(Ms_levels):max(Ms_levels);
y2 = polyval(p2,x2,[],mu2);

figure
plot(Ms_levels,ion_dose,'.')
hold on;
plot(x2,y2)
xlabel('Saturation Magnetization [A/m]');
ylabel('Ion Dose [ions/cm^2]');

title({['Dependence of Ga+ ion dose upon'] ['saturation magnetization']});
legend({'data', 'fitted curve'},'location', 'NorthWest');

%% Dosemap
load('Msat_50umx50um_binary.mat')

dosemap = polyval(p2,Msat,[],mu2);

dose_min = min(min(dosemap));
dose_max = max(max(dosemap));
binary_dosemap = zeros(size(dosemap));
binary_dosemap(dosemap>((dose_min+dose_max)/2)) = dose_max;

figure
pcolor(binary_dosemap); axis equal; shading interp;
colormap autumn;
c = colorbar;
c.Label.String = "Ga+ Ion Dose [ions/cm^{2}]";

title("Dosemap for binary focusing lens");
xlabel("*50 nm");
ylabel("*50 nm");
xlim([0 1000]);
ylim([0 1000]);

dose_min_binary = min(min(binary_dosemap));
dose_max_binary = max(max(binary_dosemap));

%% Save .xbm file
% binary_image = zeros(size(dosemap));
% binary_image(dosemap>((dose_min+dose_max)/2)) = 1;
% 
% figure
% pcolor(binary_image); axis equal; shading interp;
% c = colorbar;
% title("Binary image for .xbm");
% xlabel("*50 nm");
% ylabel("*50 nm");
% xlim([0 1000]);
% ylim([0 1000]);
% 
% write_xbm(binary_image,"binary_dosemap_50umx50um.xbm")
