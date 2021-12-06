clc;clear;

fileID=fopen('signal3.bin','r');
read_data=fread(fileID,100e6,'float');
fclose(fileID);

x=read_data(1:2:end)+1i*read_data(2:2:end);

%% For Signal and Signal2
pow=1;
PLL_Alpha=10;
PLL_Betta=20;
x=x/std(x);

FrameLen=1e5;

Param.PLL_PED_ACC_Vec = 0;
Param.PLL_Phi_Vec     = zeros(1,16);
 
for i=1:numel(x)/FrameLen
    [PLL_Out((1:FrameLen)+(i-1)*FrameLen,1),Param] = PLL_V2_FrameBased_Par(x((1:FrameLen)+(i-1)*FrameLen), pow , PLL_Alpha , PLL_Betta,FrameLen,Param) ;
end
scatterplot(PLL_Out(end-1000:end))




