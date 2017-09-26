%evalutae expected asymptotic risk of ML from threshold detection

N_thresh = 513;
N_theta = 207;

optval = 0;
ss = [0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.75, 1, 1.25, 1.5, 2, 2.5, 3];

ss = [0.3162]

for i = 1: length(ss)
    
sig = ss(i);
eta = @(x) (normpdf(x).^2)./ (normcdf(x).*normcdf(-x)+1e-30);
%un_pth = @(x) max(1-abs(0.5*x),0)+1e-7;

b = 1;
%un_pth = @(x) normpdf(x);
%un_pth = @(x) (x<1).*(x>0) + (x<-1).*(x>-2) + 1e-12;
un_pth = @(x) (x<1).*(x>-1) +  1e-12;
%un_pth = @(x) exp(-(x-b).^4) + exp(-(x-b).^4);

pth = @(x) un_pth(x) / integral(un_pth,-b,b);
th_max = b;

theta_vals = linspace(-th_max,th_max,N_theta);
thr_max = max(th_max+1,4*sig);

thresh_vals = linspace(-thr_max,thr_max,N_thresh);
dth = theta_vals(2) - theta_vals(1);
dt = thresh_vals(2) - thresh_vals(1);

[THR,TH] = meshgrid(thresh_vals, theta_vals);
delta = (THR - TH) / sig;
delta(delta>20) = 15;
delta(delta<-20) = -15;
G = eta(delta) ./ (pth(TH));

cvx_begin
 variable x(N_thresh)
 minimize sum (inv_pos(G * x) * dth)
 subject to 
 sum(x) <= 1
 x >= 0
cvx_end


figure(1)
clf
hold on
plot(theta_vals,pth(theta_vals),'g')
plot(theta_vals, 1./ (G*x),'--b')
plot(thresh_vals,x,'r')
legend('prior','opt_val','opt_x');
%plot(theta_vals,G*x)
 
optval(i) = sum(inv_pos(G * x)*dth)
end

%figure(2)
%stem(thresh_vals,x)
%saveas(gca,['opt_density_sig2.eps'],'epsc');
