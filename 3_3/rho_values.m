%% Load
clear all; close all;
load('trained_Msat.mat')

rho = Msat(11:190,11:190)/140000;

figure
pcolor(rho); axis equal; shading interp;
colormap(summer);
c = colorbar;
% c.Label.String = "rho";

title("\rho after 4 epoch");
xlabel("*25 nm");
ylabel("*25 nm");
xlim([0 180]);
ylim([0 180]);

%% Rho
[C1,ia1,ic1] = unique(rho);
rho_counts = accumarray(ic1,1);

figure
plot(C1, rho_counts, 'o','MarkerFaceColor','b');

title("\rho values after 4 epoch")
xlabel("value");
ylabel("counts");
% size(rho_counts)

%% Rho_round to 3 decimals
rho_rounded = round(rho*1000)/1000;
[C2,ia2,ic2] = unique(rho_rounded);
rho_counts = accumarray(ic2,1);

figure
plot(C2, rho_counts, 'o','MarkerFaceColor','b');

title("\rho values rounded to 3 decimals");
xlabel("value");
ylabel("counts");

%% Rho_round to 2 decimals
rho_rounded = round(rho*100)/100;
[C2,ia2,ic2] = unique(rho_rounded);
rho_counts = accumarray(ic2,1);

figure
plot(C2, rho_counts, 'o','MarkerFaceColor','b');

title("\rho values rounded to 2 decimals");
xlabel("value");
ylabel("counts");