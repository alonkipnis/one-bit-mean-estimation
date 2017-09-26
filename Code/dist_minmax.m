%evalutae minmax asymptotic risk of ML from threshold detection

N_thresh = 713;
N_theta = 513;

optval = 0;

SSig = [0.25];

KK = 0;
for i= 1: length(SSig)

sig = SSig(i);
eta = @(x) (normpdf(x).^2)./ (normcdf(x).*normcdf(-x)+1e-30);
%un_pth = @(x) max(1-abs(0.5*x),0)+1e-7;

b = 1;
th_max = b;
theta_vals = linspace(-th_max,th_max,N_theta);
thr_max = max(th_max+1,4*sig);
thr_max = th_max;

thresh_vals = linspace(-thr_max,thr_max,N_thresh);
dth = theta_vals(2) - theta_vals(1);
dt = thresh_vals(2) - thresh_vals(1);

[THR,TH] = meshgrid(thresh_vals, theta_vals);
delta = (THR - TH) / sig;
%delta(delta>20) = 15;
%delta(delta<-20) = -15;
G =  eta(delta) / sig^2;

cvx_begin
    variable x(N_thresh)
    variable l(1)
    minimize l
    subject to
    -G*x <= l
    sum(x) <= 1
    x >= 0
cvx_end

% cvx_begin
%     variable x(N_thresh)
%     minimize max(-G*x)
%     subject to
%     sum(x) <= 1
%     x >= 0
% cvx_end

opt_R = 1/min(G * x) 
app_R = 1 / (integral(@(x) eta(x/sig),-b,b)) * sig^2


KK(i) = opt_R
end


figure(2)
hold on
plot(thresh_vals,x,'b')
plot(theta_vals,G * x,'r')
plot(theta_vals,G * ones(N_thresh,1)/N_thresh,'g')

