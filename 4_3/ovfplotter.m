%% Clear
clear all; close all;

%% Parameters
x_limit = [0 1000];
y_limit = [0 1000];
file_path = 'C:\Users\mauch\Desktop\Spinwave_project\Spintorch\Spintorch_FIB\Focusing_lens\binary_lens_YIGdubs1\binary_focusing_lens\focus_binary_lens_probability_1000x1000.out';
x_label = "*50 nm";
y_label = "*50 nm";

save_ = 1;  % if ==1 save figures
save_path = "figures/";
save_name = "_binary_lens_probability_1000x1000";

%% Msat
close all;

i = 0;
data = oommf2matlab(fullfile(file_path,sprintf('Msat%6.6i.ovf',i)));
absval = data.datax.^2+data.datay.^2+data.dataz.^2;
    
data.datax(absval ==0.0) = NaN;
data.datay(absval ==0.0) = NaN;
data.dataz(absval ==0.0) = NaN;

Msat = data.datax';
    
pcolor(Msat); axis equal; shading interp;
colormap summer;
c = colorbar;
% caxis([-8*10^(-3) 8*10^(-3)])
c.Label.String = "Msat [A/m]";

title("Msat in mumax3");
xlabel(x_label);
ylabel(y_label);
xlim(x_limit);
ylim(y_limit);

if save_
    save_png = strcat(save_path,"Msat", save_name, ".png");
    save_fig = strcat(save_path,"Msat", save_name, ".fig");
    saveas(gcf,save_png)
    saveas(gcf,save_fig)
end
save('Msat_binary.mat','Msat')

%% m

% close all;

i = 51;

data = oommf2matlab(fullfile(file_path,sprintf('m%6.6i.ovf',i)));
absval = data.datax.^2+data.datay.^2+data.dataz.^2;
    
data.datax(absval ==0.0) = NaN;
data.datay(absval ==0.0) = NaN;
data.dataz(absval ==0.0) = NaN;
    
m = data.datay';

% 2D
figure
pcolor(m); axis equal; shading interp;
c = colorbar;
% caxis([-0.05 0.05])
c.Label.String = "my";
title("Magnetization in mumax3");
xlabel(x_label);
ylabel(y_label);
xlim(x_limit);
ylim(y_limit);

if save_
    save_png = strcat(save_path,"m", save_name, ".png");
    save_fig = strcat(save_path,"m", save_name, ".fig");
    saveas(gcf,save_png)
    saveas(gcf,save_fig)
end

%% Subplots

% close all;
% plotrange = 12:12:50;
plotrange = [13 25 38 50];
figure; hold all;
sgtitle('Magnetization in mumax3')

k = 1;
for i = plotrange
    data = oommf2matlab(fullfile(file_path,sprintf('m%6.6i.ovf',i)));
    absval = data.datax.^2+data.datay.^2+data.dataz.^2;
    
    data.datax(absval ==0.0) = NaN;
    data.datay(absval ==0.0) = NaN;
    data.dataz(absval ==0.0) = NaN;
    
    m = data.datay';
    
    % 2D
    subplot(2,2,k)
    pcolor(m); axis equal; shading interp;
    c = colorbar;
%     caxis([-0.05 0.05])
    c.Label.String = "my";
    title(i*2+" ns");
%     title(i*100*20E-12 +" ns");
    xlabel(x_label);
    ylabel(y_label);
    xlim(x_limit);
    ylim(y_limit);


    k = k+1;
end

if save_
    save_png = strcat(save_path,"mSubplots", save_name, ".png");
    save_fig = strcat(save_path,"mSubplots", save_name, ".fig");
    saveas(gcf,save_png)
    saveas(gcf,save_fig)
end

%% Slice
% i = 0;
% y = 1:1:1000;
% data = oommf2matlab(fullfile(file_path,sprintf('Alpha%6.6i.ovf',i)));
% absval = data.datax.^2+data.datay.^2+data.dataz.^2;
%     
% data.datax(absval ==0.0) = NaN;
% data.datay(absval ==0.0) = NaN;
% data.dataz(absval ==0.0) = NaN;
% 
% figure
% plot(y,data.datax(1:1000,500));
% title("Slice of magnetization after 25 ns y = 50");
% xlabel(x_label);
% ylabel("*my");
% xlim(x_limit);
% 
% if save_
%     save_png = strcat(save_path,"slice", save_name, ".png");
%     save_fig = strcat(save_path,"slice", save_name, ".fig");
%     saveas(gcf,save_png)
%     saveas(gcf,save_fig)
% end