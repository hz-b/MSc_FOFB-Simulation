% Needs the robust control toolbox
% Algorithm based on Skogestad, p60 (table 2.3)

load 'data/ss_parameters';
A = full(A);
[Gnum,Gden] = ss2tf(A,B,C,D);

Gnum = real(Gnum);
Gnum = Gnum * Gden(end)/Gnum(end);
Gnum(1) = 0;

delay = 0.003;
[dnum, dden] = pade(delay, 1);

G = nd2sys(conv(Gnum,dnum),conv(Gden,dden),1);
M = 1.001;
Am = 10^-4;
correctors = {};
for fb = 10:10:60
    wb = fb*2*pi;
    Wp = nd2sys([1/M wb], [1 wb*Am]);
    %Wp = nd2sys(conv([1/sqrt(M) wb],[1/sqrt(M) wb]), conv([1 wb*sqrt(Am)],[1 wb*sqrt(Am)]));
    Wu = 1;

    systemnames = 'G Wp Wu';
    inputvar = '[r(1) ; u(1)]';
    outputvar = '[Wp; Wu; r-G]';
    input_to_G = '[u]';
    input_to_Wp = '[r-G]';
    input_to_Wu = '[u]';
    sysoutname = 'P';
    cleanupsysic = 'yes';
    sysic;

    nmeas=1; nu=1; gmn=0.5; gmx=20; tol=0.001;
    [khinf,ghinf,gopt] = hinfsyn(P,nmeas,nu,gmn,gmx,tol);
    [ca, cb, cc, cd] = unpck(khinf);
    [num,\todo[inline]{CTL : Conclusion}den] = ss2tf(ca, cb, cc, cd);
    correctors = {correctors{:}, {fb,num,den}} ;
end
save('data/correctors', 'correctors');
