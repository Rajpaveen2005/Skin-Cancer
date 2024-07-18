clc
clear all
close all
warning off all;

%load the data

load train1
load train2
load train3
load train4
load train5
load train6
load train7
load train8
load train9

T=[train1 train2  train3 train4 train5 train6 train7 train8 train9];
% 
x=[1 2 3 4 5 6 7 8 9];

% create a feed forward neural network
net1 = newff(minmax(T),[5000 70 1],{'logsig','logsig','purelin'},'trainrp');
net1.trainParam.show = 1000;
net1.trainParam.lr = 0.01;
net1.trainParam.epochs = 7000;
net1.trainParam.goal = 1e-6;

%train the neural network using the input,target and the created network
[net1] = train(net1,T,x);
%save the network
save net1 net1

%simulate the network for a particular input
y = round(sim(net1,T));
