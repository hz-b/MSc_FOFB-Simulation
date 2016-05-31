load(filename)
opts.relax=1;
 %Use vector fitting with relaxed non-triviality constraint
opts.stable=1;
 %Enforce stable poles
opts.asymp=3;
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
 
N = 5
[SER,poles,rmserr,fit,opts]=vectfit3(h0,2*pi*1i*f,-2*pi*logspace(0,4,N),ones(size(h0)), opts)
save(SER)
