
N_thresh = 719;
N_theta = 121;

sig = 1;

theta_vals = linspace(-4*sig,4*sig,N_theta);
thresh_vals = linspace(-4*sig,4*sig,N_thresh);
dth = theta_vals(2) - theta_vals(1);
dt = thresh_vals(2) - thresh_vals(1);

g = @(x)   normpdf(x/sig).^2 ./ (normcdf(x/sig).*normcdf(-x/sig)+1e-7)/sig^2;
%un_pth = @(x) max(1-abs(0.5*x),0)+1e-7;
un_pth = @(x) normpdf(x);
%un_pth = @(x) (x<sig).*(x>-sig);
b = 1;
un_pth = @(x) exp(-(x-b).^4) + exp(-(x-b).^4);

pth = @(x) un_pth(x) / integral(un_pth,-5,5);

[THR,TH] = meshgrid(thresh_vals, theta_vals);

%G = g(THR - TH) / pth(TH);
G = g(THR - TH) ./ (pth(TH));
G(isinf(G))=0;

b = 1;
cvx_begin
 variable x(N_thresh)
 obj = 0.5*quad_over_lin(1,g(thresh_vals-b).* x) +...
     0.5*quad_over_lin(1,g(thresh_vals+b).* x)

 minimize obj
 subject to 
 sum(x) <= 1
 x >= 0
cvx_end

plot(thresh_vals,x)

figure(1)
clf
hold on
plot(theta_vals,pth(theta_vals),'b')
plot(theta_vals, 1. / (G*x),'--b')
plot(thresh_vals,x,'r')
%plot(theta_vals,G*x)
sprintf('int = %f ',integral(@(x) pth(x)./g(x),-5,5))
sprintf('opt_val = %f ', sum(dth./(G*x)) )


