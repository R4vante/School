function plot_model(data, model, save, name)

fig = figure();
hold on;
grid on;

scatter(data.x, data.y, 'filled')
plot(data.x, model.predictFcn(data), LineWidth=3)

xlabel('x')
ylabel('y')
legend(["Data", "Regression"])

if save == 1
    saveas(fig, "Figures/"+name+".jpg")
end