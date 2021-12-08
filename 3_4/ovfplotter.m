%% Clear
clear all; close all;
%% Parameters
x_limit = [0 1000];
y_limit = [0 1000];
file_path = 'C:\Users\mauch\Desktop\Spinwave_project\Spintorch\Spintorch_FIB\Focusing_lens\binary_lens_YIGdubs2\mumax\focus_binary_lens_YIGdubs2_100umx100um_alpha.out';
x_label = "*100 nm";
y_label = "*100 nm";
caxes_ = 0;    % if ==1 limits
axses_ = [-0.0025 0.0025];

save_ = 0;  % if ==1 save figures
save_path = "figures/";
save_name = "100umx100um.out";

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
c = colorbar;
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
% save('Msat_100umx100um_binary.mat','Msat')

%% m

% close all;

i = 50;

data = oommf2matlab(fullfile(file_path,sprintf('m%6.6i.ovf',i)));
absval = data.datax.^2+data.datay.^2+data.dataz.^2;
    
data.datax(absval ==0.0) = NaN;
data.datay(absval ==0.0) = NaN;
data.dataz(absval ==0.0) = NaN;
    
m = data.datay';

figure
pcolor(m); axis equal; shading interp;
c = colorbar;
if caxes_
    caxis(axses_)
end
caxis([-0.001 0.001])
c.Label.String = "my";
title("Magnetization in mumax3");
xlabel(x_label);
ylabel(y_label);
% xlim(x_limit);
xlim([750 1000]);
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
    if caxes_
        caxis(axses_)
    end
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
i = 51;
y = 1:1:1000;
data = oommf2matlab(fullfile(file_path,sprintf('m%6.6i.ovf',i)));
absval = data.datax.^2+data.datay.^2+data.dataz.^2;
    
data.datax(absval ==0.0) = NaN;
data.datay(absval ==0.0) = NaN;
data.dataz(absval ==0.0) = NaN;

figure
plot(y,data.datax(1:1000,500));
title("Slice of magnetization");
xlabel(x_label);
ylabel("*my");
xlim(x_limit);

if save_
    save_png = strcat(save_path,"slice", save_name, ".png");
    save_fig = strcat(save_path,"slice", save_name, ".fig");
    saveas(gcf,save_png)
    saveas(gcf,save_fig)
end