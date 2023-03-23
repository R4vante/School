function predplot(data_table, model, save, name)

y = data_table.y;

fig = figure();
grid on;
hold on;
scatter(y, model.predictFcn(data_table), 'filled')
plot(y, y, 'k', LineWidth=3)
xlabel("True Response")
ylabel("Predicted Response")

if save == 1
    saveas(fig, "Figures/"+name+".jpg")
end
