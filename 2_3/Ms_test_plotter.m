clear all; close all;
%% Msat
load('trained_msat.mat');

rho = Msat/140000;
rounded_rho = round(rho*100)/100;
rounded_Msat = rounded_rho * 140000;

%% Magnetization
load('magnetization.mat');
m = magnetization(:,:,1001);


%% Plot
figure(1)
pcolor(m); axis equal; shading interp;
c = colorbar;
caxis([-8*10^(-3) 8*10^(-3)])
c.Label.String = "my";
    
title("SpinTorch, Msat rounded to 2 decimals");
ylabel("*25 nm");
xlabel("*25 nm");
xlim([0 200]);
ylim([0 200]);

figure(2)
% Slice
y = 1:1:200;
plot(y,m(100,1:200));

%% Slice SpinTorch vs. MuMax3

i = 101;
y = 1:1:200;
data = oommf2matlab(fullfile('C:\Users\mauch\Desktop\Spinwave_project\Spintorch\Raktar\Msat_teszt\Ms_test.out',sprintf('m%6.6i.ovf',i)));
absval = data.datax.^2+data.datay.^2+data.dataz.^2;
    
data.datax(absval ==0.0) = NaN;
data.datay(absval ==0.0) = NaN;
data.dataz(absval ==0.0) = NaN;

figure; hold all;
plot(y,data.datax(1:200,100));
plot(y,m(100,1:200));
legend('MuMax3','SpinTorch','Location','northwest')
title("SpinTorch vs. MuMax3");
xlabel("*25 nm");
ylabel("*my");
xlim([0 200]);