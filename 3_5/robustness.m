%% Clear
clear all; close all;

%% Save figure
save_ = 0;  % if ==1 save figures
save_path = "figures/";

%% Bext
load('Bext_out.mat')
load('BextChange_WL_levels_268mT_to_289mT.mat')

figure
yyaxis left
plotx = Bext_out(1,:)*1000;
ploty = Bext_out(2,:);
plot(plotx, ploty, '.-')
ylabel('Normalized intensity on expected output');
yyaxis right
plot(plotx(1:22), Bext_WL_levels*1e6, '.-')
ylabel('Wavelength [\mum]');
xlabel('External field [mT]');
title({['Dependence of wavelength and accuracy'] ['upon changing of external field']});
xlim([plotx(1) plotx(end)]);

if save_
    save_name = "268mT_to_298mT";
    save_png = strcat(save_path,"Bext_out", save_name, ".png");
    save_fig = strcat(save_path,"Bext_out", save_name, ".fig");
    saveas(gcf,save_png)
    saveas(gcf,save_fig)
end

%% f
load('f_out.mat')
load('fChange_WL_levels_3177MHz_to_3552MHz.mat')

figure
yyaxis left
plotx = f_out(1,:)*1e-3;
ploty = f_out(2,:);
plot(plotx,ploty,'.-')
ylabel('Normalized intensity on expected output');
yyaxis right
plot(plotx(4:29), f_WL_levels*1e6, '.-')
xlabel('Frequency [GHz]');
ylabel('Wavelength [\mum]');
title({['Dependence of accuracy upon cahnging'] ['of frequency']});
xlim([plotx(1) plotx(end)]);

if save_
    save_name = "3132MHz_to_3552MHz";
    save_png = strcat(save_path,"f_out", save_name, ".png");
    save_fig = strcat(save_path,"f_out", save_name, ".fig");
    saveas(gcf,save_png)
    saveas(gcf,save_fig)
end

%% Ms
load('Ms_out.mat')
load('MsChange_WL_levels_130670_to_145670.mat')

figure
yyaxis left
plotx = Ms_out(1,:)*1e-3;
ploty = Ms_out(2,:);
plot(plotx, ploty, '.-')
ylabel('Normalized intensity on expected output');
yyaxis right
plotx2 = Ms_out(1,6:21)*1e-3;
ploty2 = Ms_WL_levels*1e6;
plot(plotx2,ploty2,'.-')
ylabel('Wavelength [\mum]');
xlabel('Saturation magnetization [kA/m]');
title({['Dependence of wavelength and accuracy'] ['upon cahnging of Ms']});
xlim([plotx(1) plotx(end)]);

if save_
    save_name = "125670Aperm_to_145670Aperm";
    save_png = strcat(save_path,"Ms_out", save_name, ".png");
    save_fig = strcat(save_path,"Ms_out", save_name, ".fig");
    saveas(gcf,save_png)
    saveas(gcf,save_fig)
end

%% Wavelength
load('Ms_out2.mat')
load('MsChange_WL_levels_130670_to_145670_step500.mat')
figure
plot(Bext_WL_levels*1e6,Bext_out(2,1:22));
hold on
plot(Ms_WL_levels*1e6,Ms_out(2,:));
plot(f_WL_levels*1e6,f_out(2,4:29));
hold off
legend('Changing Bext', 'Changing Ms', 'Changing f');
xlim([0 18]);
% title('Dependence of accuracy upon wavelength');
title({['Dependence of accuracy'] ['upon wavelength']});
xlabel('Wavelength [\mum]');
ylabel('Normalized intensity on expected output');

if save_
    save_png = strcat(save_path,"Wavelength_out", ".png");
    save_fig = strcat(save_path,"Wavelength_out", ".fig");
    saveas(gcf,save_png)
    saveas(gcf,save_fig)
end
