clc;clear;
SPS=4;
fileAddress='signal_Fc0SPS4ModuQPSKSNR15.bin';

%%
fileID=fopen(fileAddress,'r');
A=fread(fileID,'float');
fclose(fileID);
signal=A(1:2:end)+1i*A(2:2:end);

%% Filter Design

Fpass = 0.28;             % Passband Frequency
Fstop = Fpass+0.02;            % Stopband Frequency
Dpass = 0.0057501127785;  % Passband Ripple
Dstop = 0.0001;          % Stopband Attenuation
dens  = 20;              % Density Factor

% Calculate the order from the parameters using FIRPMORD.
[N, Fo, Ao, W] = firpmord([Fpass, Fstop], [1 0], [Dpass, Dstop]);

% Calculate the coefficients using the FIRPM function.
filterCoef  = firpm(N, Fo, Ao, W, {dens});

filteredSignal=filter(filterCoef,1,signal);
%%
RootRaisedCosineDelay = 10;
RollOff_Factor =  0.1;
NumberOfSamplePerSymbol = SPS;
RootRaisedCosineTaps = rcosdesign(RollOff_Factor,2*RootRaisedCosineDelay,NumberOfSamplePerSymbol,'sqrt');
MatchFilteredSignal = filter(RootRaisedCosineTaps,1,filteredSignal) ;
Normal_Signal =MatchFilteredSignal/std(MatchFilteredSignal);
%%
sym = comm.SymbolSynchronizer('TimingErrorDetector', 'Gardner (non-data-aided)');
sym.SamplesPerSymbol = SPS;
output_M = sym(Normal_Signal);

TR_Gardner_Cubic_Out_Ser = TR_Gardner_Cubic(Normal_Signal , SPS/2 , 10 , 20);
TR_Gardner_Cubic_Out_Par = TR_Gardner_Cubic_Par(Normal_Signal , SPS/2 , 10 , 20);
%%
refSig = pskmod(0:3, 4,pi/4);
width = 335;
height = 400;
constdiag = comm.ConstellationDiagram('ReferenceConstellation', refSig, 'Position', [0, 200, width, height]);
FrameLEn = 1e3;
% RxSig = vec2mat(TR_Gardner_Cubic_Out_Ser(2:2:end),FrameLEn);
RxSig = vec2mat(output_M,FrameLEn);
[idx, idy] = size(RxSig);

for i=1:idx-1
    constdiag(RxSig(i,:).')
    pause(0.05)
end


