x = -0.9:0.001:0.9;
y = btrain2/ wtrain2(2) - x * wtrain2(1) / wtrain2(2);

scatter(data(:,1), data(:,2))
hold on
plot(x',y')