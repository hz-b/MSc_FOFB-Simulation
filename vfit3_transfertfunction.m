% Need of the vfit3 or the matrix fitting MatrixFittingToolbox
% https://www.sintef.no/projectweb/vectfit/downloads/

close all
clear all

FILE = 'data/tf.mat';

addpath('../MatrixFittingToolbox');

load(FILE)
opts.relax=1;
 %Use vector fitting with relaxed non-triviality constraint
opts.stable=1;
 %Enforce stable poles
opts.asymp=2;
 %Include both D, E in fitting
opts.skip_pole=0;
 %Do NOT skip pole identification
opts.skip_res=0;
 %Do NOT skip identification of residues (C,D,E)
opts.cmplx_ss=1;
 %Create complex state space model
opts.spy1=0;
 %No plotting for first stage of vector fitting
opts.spy2=1;
 %Create magnitude plot for fitting of f(s)
opts.logx=1;
 %Use logarithmic abscissa axis
opts.logy=1;
 %Use logarithmic ordinate axis
opts.errplot=1;
 %Include deviation in magnitude plot
opts.phaseplot=1;
 %Also produce plot of phase angle (in addition to magnitiude)
opts.legend=1;
 %Do include legends in plots

N = 3;
start = 30;
h0 = h0(start:end);
poles_init = -2*pi*logspace(0,4,N);
weights = ones(size(h0));
s = 2*pi*1i*f(start:end);
[SER,poles,rmserr,fit,opts]=vectfit3(h0,s,poles_init, weights, opts);

A = SER.A;
B = SER.B;
C = SER.C;
D = SER.D;

%[H,w] = freqresp(ss(normal(A),B,C,D), logspace(-3,5));
%figure()
%loglog(w,abs(reshape(H,1,numel(H))))
[num,den] = tfdata(ss2tf(full(A),B,C,D),'v');
H = tf(1,den);
[A,B,C,D]=ss(H);

figure()
loglog(w,abs(reshape(H,1,numel(H))))

%save('data/ss_parameters', 'A','B','C','D')
