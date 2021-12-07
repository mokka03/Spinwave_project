function [data]=oommf2matlab(fileToRead1)
%This is a function to import vector file archives from oommf into Matlab
%arrays
%
%Oommf vector files must be writen with the output Specifications "text %g"
%instead of the default "binary 4" option. And the type of grid must be
%rectangular.
%
%Vector files will be imported into the object "data" which will have this
%fields:
%           field: current applied magnetic field
%           xmin: minimum x value
%           xnodes: number of nodes used along x
%           xmax: maximum x value
%           ymin: minimum y value
%           ynodes: number of nodes used along y
%           ymax: maximum y value
%           zmin: minimum z value
%           znodes: number of nodes used along z
%           zmax: maximum z value
%           datax: component x of vector on data file
%           datay: component y of vector on data file
%           dataz: component z of vector on data file
%           positionx: x positions of vectors
%           positiony: y positions of vectors
%           positionz: z positions of vectors
%
%   Example:
%       We have created with Oommf the archive test.omf (included on the
%       zip). To open it into "data"
%
%       data=oommf2matlab('test.omf')
%
%       Now we can make a 2D vector field
%
%       quiver(data.positionx,data.positiony,data.datax,data.datay,0.5)
%
%       or calculate the divergence and plot it
%
%       div=divergence(data.positionx,data.positiony,data.datax,data.datay);
%       pcolor(data.positionx,data.positiony,div)
%       shading flat
%       colormap bone
%
%       For more examples, you can see my blog (look for Oommf, to be updated shortly):
%       http://thebrickinthesky.wordpress.com/
%
%  References:
%  [1] Oommf Micromagnetic simulator at NIST,
%      http://math.nist.gov/oommf/
%
%This function was written by :
%                             Hctor Corte
%                             B.Sc. in physics 2010
%                             M.Sc. in Complex physics systems 2012
%                             Ph.D Student between NPL (National Physical Laboratory) and Royal Holloway University of London
%                             London,
%                             United kingdom.
%                             Email: leo_corte@yahoo.es
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%We open the data file and start reading lines. The first lines are the
%header with information about the simulation. On this version not all the
%information is stracted, but is quite easy to do
fileA=fopen(fileToRead1,'r');
linea=fgetl(fileA);
data.xmin=[];


while isempty(strfind(linea,'# Begin: Data Text'))==1
    %The headers ends when the line # Begin: Data Text appears    
    
    %%%%%%%%%%%%%%%%%%%%%%%% Each one of these is going to look for some
    %%%%%%%%%%%%%%%%%%%%%%%% information on the header. This one for the
    %%%%%%%%%%%%%%%%%%%%%%%% applied field
    if isempty(strfind(linea,'# Desc: Applied field (T):'))~=1        
        remain = linea;
        while true
            [str, remain] = strtok(remain);
            if strcmp(str,'(T):')==1
                [str, remain] = strtok(remain);
                data.field(1)=  str2num(str);
                [str, remain] = strtok(remain);
                data.field(2)=  str2num(str);
                [str, remain] = strtok(remain);
                data.field(3)=  str2num(str);
            end
            if isempty(str),  break;  end            
        end        
    end
    %%%%%%%%%%%%%%%%%%
    
    %%%%%%%%%%%%%%%%%%%%%%%%This one is for the simulation grid,
    %%%%%%%%%%%%%%%%%%%%%%%%for the number of nodes on x
    if isempty(strfind(linea,'# xnodes'))~=1        
        remain = linea;        
        [~, remain] = strtok(remain);
        [~, remain] = strtok(remain);
        [str, ~] = strtok(remain);
        data.xnodes=  str2num(str);       
    end    
    %%%%%%%%%%%%%%%%%%
    
    %%%%%%%%%%%%%%%%%%%%%%%%Number of nodes on y
    if isempty(strfind(linea,'# ynodes'))~=1        
        remain = linea;        
        [~, remain] = strtok(remain);
        [~, remain] = strtok(remain);
        [str, ~] = strtok(remain);
        data.ynodes=  str2num(str);        
    end    
    %%%%%%%%%%%%%%%%%%    
    
    %%%%%%%%%%%%%%%%%%%%%%%%Number of nodes on z
    if isempty(strfind(linea,'# znodes'))~=1        
        remain = linea;        
        [~, remain] = strtok(remain);
        [~, remain] = strtok(remain);
        [str, ~] = strtok(remain);
        data.znodes=  str2num(str);        
    end    
    %%%%%%%%%%%%%%%%%%    
    
    %%%%%%%%%%%%%%%%%%%%%%%%Now the min and maximum values of x y and z
    if isempty(strfind(linea,'# xmin:'))~=1        
        remain = linea;        
        [~, remain] = strtok(remain);
        [~, remain] = strtok(remain);
        [str, ~] = strtok(remain);
        data.xmin=  str2num(str);        
    end    
    %%%%%%%%%%%%%%%%%%
    
    %%%%%%%%%%%%%%%%%%%%%%%%
    if isempty(strfind(linea,'# ymin:'))~=1        
        remain = linea;        
        [~, remain] = strtok(remain);
        [~, remain] = strtok(remain);
        [str, ~] = strtok(remain);
        data.ymin=  str2num(str);
    end    
    %%%%%%%%%%%%%%%%%%
    
    %%%%%%%%%%%%%%%%%%%%%%%%
    if isempty(strfind(linea,'# zmin:'))~=1        
        remain = linea;        
        [~, remain] = strtok(remain);
        [~, remain] = strtok(remain);
        [str, ~] = strtok(remain);
        data.zmin=  str2num(str);       
    end    
    %%%%%%%%%%%%%%%%%%
    
    %%%%%%%%%%%%%%%%%%%%%%%%
    if isempty(strfind(linea,'# xmax:'))~=1        
        remain = linea;        
        [~, remain] = strtok(remain);
        [~, remain] = strtok(remain);
        [str, ~] = strtok(remain);        
        data.xmax=  str2num(str);         
    end    
    %%%%%%%%%%%%%%%%%%   
    
    %%%%%%%%%%%%%%%%%%%%%%%%
    if isempty(strfind(linea,'# ymax:'))~=1        
        remain = linea;        
        [~, remain] = strtok(remain);
        [~, remain] = strtok(remain);
        [str, ~] = strtok(remain);
        data.ymax=  str2num(str);        
    end    
    %%%%%%%%%%%%%%%%%%
    
    %%%%%%%%%%%%%%%%%%%%%%%%
    if isempty(strfind(linea,'# zmax:'))~=1        
        remain = linea;        
        [~, remain] = strtok(remain);
        [~, remain] = strtok(remain);
        [str, ~] = strtok(remain);
        data.zmax=  str2num(str);        
    end    
    %%%%%%%%%%%%%%%%%%    
    linea=fgetl(fileA);    
end
%Now beguins the reading of the vector field values.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Because we already have the data about the grid, size and number of nodes,
%we can create the arrays containing the coordinates of the points on the
%grid. It is necesary here that the simulation grid is rectangular.
x=linspace(data.xmin,data.xmax,data.xnodes);
y=linspace(data.ymin,data.ymax,data.ynodes);
z=linspace(data.zmin,data.zmax,data.znodes);
[X,Y,Z]=meshgrid(x,y,z);
data.datax=0.*permute(X,[2,1,3]);%Gives the proper size to datax
data.datay=0.*permute(Y,[2,1,3]);%Gives the proper size to datay
data.dataz=0.*permute(Z,[2,1,3]);%Gives the proper size to dataz
data.positionx=permute(X,[2,1,3]);
data.positiony=permute(Y,[2,1,3]);
data.positionz=permute(Z,[2,1,3]);
%Now beguins to read the vector field and store their components.
%Since our scan of the file is at the beguining of the data, we can scan for the
%data in the format of '%f \t%f \t%f' The scan will end at the end of the
%file because the format changes again.
s=textscan(fileA,'%f \t%f \t%f');
S=cell2mat(s);
data.datax(:)=S(:,1);
data.datay(:)=S(:,2);
data.dataz(:)=S(:,3);

fclose(fileA);
end