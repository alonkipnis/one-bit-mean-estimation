
Amin = -10;
Amax = 10;

thh = Amin:0.1:Amax;

k = 1;
points = Amin+(Amax-Amin)*sort(rand(2*k,1));

a = points(1:2:end)
b = points(2:2:end)

val = 0;
ind = 0;
post = 0;
for i = 1:length(thh)
    th = thh(i);
    val(i) = (sum(normpdf(a-th)-normpdf(b-th)))^2./(sum(normcdf(b-th)-normcdf(a-th))*(1-sum(normcdf(b-th)-normcdf(a-th)) ) );
    ind(i) = sum((th>a).*(th<b));
    post(i) = sum(normcdf(b-th)-normcdf(a-th));
end

figure(1)
clf
hold on
plot(thh,val,'b')
plot(thh,ind,'r')
%plot(thh,post,'g')
plot(thh,2/pi+0*thh,'--k')