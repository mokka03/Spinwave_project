clear all; close all;
%% mx average Spintorch
% Setup the Import Options and import the data
opts = delimitedTextImportOptions("NumVariables", 5);

% Specify range and delimiter
opts.DataLines = [2, Inf];
opts.Delimiter = "\t";

% Specify column names and types
opts.VariableNames = ["ts", "mx", "my", "mz", "dts"];
opts.VariableTypes = ["double", "double", "double", "double", "double"];

% Specify file level properties
opts.ExtraColumnsRule = "ignore";
opts.EmptyLineRule = "read";

% Import the data
table = readtable("C:\Users\mauch\Desktop\Spinwave_project\Spintorch\Raktar\Standard_problem\standard4.out\table.txt", opts);

% Convert to output type
table = table2array(table);

% Clear temporary variables
clear opts
% close all
figure

load('mx_sum.mat');
load('my_sum.mat');
load('mz_sum.mat');
t = 0:5e-4:1;
plot(t,mx_sum)
hold on
plot(t,my_sum)
plot(t,mz_sum)
plot(table(:,1)*1e9,table(:,2),'.');
plot(table(:,1)*1e9,table(:,3),'.');
plot(table(:,1)*1e9,table(:,4),'.');
legend({"mx SpinTorch", "my SpinTorch", "mz SpinTorch", "mx mumax3", "my mumax3", "mz mumax3"},'NumColumns',2);
title('Standard problem #4');
xlim([0,1]);
ylim([-1,1]);
xlabel('m');
xlabel('t [ns]');
ylabel('m');
hold off
%% Magnetization
% load('magnetization.mat');
% m_x = magnetization(:,:,1);
% m_y = magnetization(:,:,2);
% m_z = magnetization(:,:,3);
load('M0_without_relax.mat');
M0_x = M0(:,:,1);
M0_y = M0(:,:,2);
M0_z = M0(:,:,3);

%% Plot magnetization
figure(10)
pcolor(M0_z); axis equal; shading interp;
c = colorbar;
caxis([-8*10^(-3) 8*10^(-3)])
c.Label.String = "my";
    
% title("SpinTorch, Msat rounded to 2 decimals");
% ylabel("*25 nm");
% xlabel("*25 nm");
xlim([0 128]);
ylim([0 32]);
