g1 = @(x) -(5*abs(x).*(x<0)+(x-2).^1.5.*(x>2))
shift = 8;
g = @(x) g1(x-shift);
xmax = 7+shift;
xmin = -7+shift;
f_un = @(x) (exp(g(x)) ).*(x>xmin).*(x<xmax) + exp(-abs(x));
al = integral(f_un,xmin,xmax);

f = @(x) f_un(x)/al;

xx = xmin:0.05:xmax;
figure(2)
plot(xx,g(xx))

mean = integral(@(x) x.*f(x),xmin,xmax);
m_p = @(t) integral(@(x) x.*f(x),t,xmax)/integral(f,t,xmax);
m_m = @(t) integral(@(x) x.*f(x),xmin,t)/integral(f,xmin,t);

obj1 = 0;
obj2 = 0;
dobj1 = 0
den = 0;
den2 = 0;
h = 0;
fx = 0;
for i = 1:length(xx)
    tau = xx(i);
    obj1(i) = m_p(tau)-tau;
    obj2(i) = tau-m_m(tau);
    dobj1(i) = 1 - integral(@(x) (tau-x).*f(x).*f(tau),xmin,tau)/integral(f,xmin,tau)^2;
    h(i) = integral(@(x) (x-tau).*f(x),xmin,tau);
    den(i) = integral(@(x) f(x),xmin,tau);
    den2(i) = sum(den)*(xx(2)-xx(1));
    fx(i) = 1 - f(tau)*den2(i)/(den(i)*den(i));
end

figure(1)
clf
hold on
%plot(xx,obj1,'b')
plot(xx,obj2,'r')
plot(xx,dobj1,'-ob')
%plot(xx,h,'-.k')
plot(xx,f(xx),'--k')
plot(xx,fx,'-xg')
%plot(xx,den,'--g')
%plot(xx,(obj1-obj2)/2,'g')


