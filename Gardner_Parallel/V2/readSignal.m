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
d_bb_r=Normal_Signal;
clearvars -except d_bb_r SPS
