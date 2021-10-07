clear;
clc;

%% input
addpath('./Alg')
addpath('./Signal')

% run synthesisSignal.m
run readSignal.m

%% Framing - Timing recovery
frameLength = 99999;
numFrames  = floor(length(d_bb_r) / frameLength);
frames = zeros(frameLength + 4, numFrames);
frames(3:end, 1) = d_bb_r(1:(frameLength + 2));
for jj = 2: numFrames
    frames(:, jj) = d_bb_r((frameLength * (jj - 1) -1):(frameLength * jj + 2));
end

%% processing each frame independently - Timing recovery
output_perFrame = zeros(floor(frameLength * 11 / (10 * SPS)) , numFrames);
CC     = zeros(numFrames , 1);
COE    = zeros(numFrames , 1);
comulative_CC = zeros(numFrames + 1 , 1);
% SPS=4.1;
for kk = 1:numFrames
    [Y, C, OE] =  TR_Gardner_ForwardBackward(frames(:, kk) ,  SPS/2 , 10, 20);
    output_perFrame(1:C, kk) = Y;
    CC(kk) = C;
    COE(kk) = OE;
    comulative_CC(kk + 1) = sum(CC);
end

%% Output reconstruction - Timing recovery
output_S=FrameMerger(output_perFrame,COE,CC);
% output_S

n_Sym_True = numel(output_S);
%% framing - PLL
numFrames = 3;
frameLength = floor(n_Sym_True / numFrames);
frames = zeros(frameLength, numFrames);
for i = 1:numFrames
    index = ((i - 1) * frameLength + 1):(i * frameLength);
    frames(:, i) = output_S(index);
end
%% processing each frame independently - PLL
pow=1;
PLL_Alpha=10;
PLL_Betta=20;
PLL_Out = zeros(frameLength, numFrames);
phase_mismatch = zeros(numFrames, 2); % [Phi applied to first sample, Phi applied to last sample]
for i = 1:numFrames
    [PLL_Out(: ,i), phase_mismatch(i, 1), phase_mismatch(i, 2)] = PLL_V1(frames(:, i) , pow , PLL_Alpha , PLL_Betta);
end

output_S=PLL_FrameMerger(PLL_Out,phase_mismatch,frameLength);

scatterplot(output_S)




