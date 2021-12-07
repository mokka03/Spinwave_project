%% Msat
clear all; close all;
% load('Msat50alpha.mat')
load('Msat_50umx50um_binary.mat')

figure
pcolor(Msat); axis equal; shading interp;
colormap summer;
c = colorbar;
c.Label.String = "Saturation Magnetization [A/m]";

title("Msat in mumax3");
xlabel("*50 nm");
ylabel("*50 nm");
xlim([0 1000]);
ylim([0 1000]);

min_ = min(min(Msat));
max_ = max(max(Msat));

% save_path = "figures/";
% save_png = strcat(save_path,"Msat_focus_binary_lens_YIGdubs2_50umx50um_alpha", ".png");
% save_fig = strcat(save_path,"Msat_focus_binary_lens_YIGdubs2_50umx50um_alpha", ".fig");
% saveas(gcf,save_png)
% saveas(gcf,save_fig)

%% m
close all
load('m100alpha.mat')

m = m/143725;

figure
pcolor(m); axis equal; shading interp;
c = colorbar;
c.Label.String = "my";

title("my");
xlabel("*200 nm");
ylabel("*200 nm");
xlim([0 500]);
ylim([0 500]);

%% Slice
i = 0;
y = 1:1:500;


figure
plot(y,m(250,1:500));
title("Slice of magnetization after 25 ns y = 50");
xlabel(x_label);
ylabel("*my");
xlim(x_limit);

% if save_
%     save_png = strcat(save_path,"slice", save_name, ".png");
%     save_fig = strcat(save_path,"slice", save_name, ".fig");
%     saveas(gcf,save_png)
%     saveas(gcf,save_fig)
% end