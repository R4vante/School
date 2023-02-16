clear; clc;

%% Import data

data = readtable("Data.xlsx");

%% Import regressions

[lin_noval, rmse_lin_noval] = linear_noval(data);
[lin_hold, rmse_lin_hold] = linear_hold(data);
[lin_cross, rmse_lin_cross] = linear_cross(data);
[exp_noval, rmse_exp_noval] = expGPR_noval(data);
[exp_hold, rmse_exp_hold] = expGPR_hold(data);
[exp_cross, rmse_exp_cross] = expGPR_cross(data);
[sqexp_noval, rmse_sqexp_noval] = sqExp_noval(data);
[sqexp_hold, rmse_sqexp_hold] = sqExp_hold(data);
[sqexp_cross, rmse_sqexp_cross] = sqExp_cross(data);


%% Plot linear
fig1 = figure(1);
grid()
hold on;
scatter(data.x, data.y, "filled");
plot(data.x, lin_noval.predictFcn(data), LineWidth=3);
legend(["Data", "Regressie"])
xlabel("x")
ylabel("y")
% saveas(fig1, "Figures/lin_noval.jpg")

fig2 = figure(2);
grid()
hold on;
scatter(data.x, data.y, "filled");
plot(data.x, lin_hold.predictFcn(data), LineWidth=3);
legend(["Data", "Regressie"])
xlabel("x")
ylabel("y")
% saveas(fig2, "Figures/lin_hold.jpg")

fig3 = figure(3);
grid()
hold on;
scatter(data.x, data.y, "filled");
plot(data.x, lin_cross.predictFcn(data), LineWidth=3);
legend(["Data", "Regressie"])
xlabel("x")
ylabel("y")
% saveas(fig3, "Figures/lin_cross.jpg")

%% Plot expGPR

fig4 = figure(4);
grid()
hold on;
scatter(data.x, data.y, "filled");
plot(data.x, exp_noval.predictFcn(data), LineWidth=3);
legend(["Data", "Regressie"])
xlabel("x")
ylabel("y")
% saveas(fig4, "Figures/exp_noval.jpg")

fig5 = figure(5);
grid()
hold on;
scatter(data.x, data.y, "filled");
plot(data.x, exp_hold.predictFcn(data), LineWidth=3);
legend(["Data", "Regressie"])
xlabel("x")
ylabel("y")
% saveas(fig5, "Figures/exp_hold.jpg")

fig6 = figure(6);
grid()
hold on;
scatter(data.x, data.y, "filled");
plot(data.x, exp_cross.predictFcn(data), LineWidth=3);
legend(["Data", "Regressie"])
xlabel("x")
ylabel("y")
% saveas(fig6, "Figures/exp_cross.jpg")

%% Plot sqrexp

fig7 = figure(7);
grid()
hold on;
scatter(data.x, data.y, "filled");
plot(data.x, sqexp_noval.predictFcn(data), LineWidth=3);
legend(["Data", "Regressie"])
xlabel("x")
ylabel("y")
% saveas(fig7, "Figures/sqexp_noval.jpg")

fig8 = figure(8);
grid()
hold on;
scatter(data.x, data.y, "filled");
plot(data.x, sqexp_hold.predictFcn(data), LineWidth=3);
legend(["Data", "Regressie"])
xlabel("x")
ylabel("y")
% saveas(fig8, "Figures/sqexp_hold.jpg")

fig9 = figure(9);
grid()
hold on;
scatter(data.x, data.y, "filled");
plot(data.x, sqexp_cross.predictFcn(data), LineWidth=3);
legend(["Data", "Regressie"])
xlabel("x")
ylabel("y")
% saveas(fig9, "Figures/sqexp_cross.jpg")


%% test exp op testdata
[x_t, y_t] = generate_dataset(250, 0.4);
data_t = table(x_t', y_t', VariableNames={'x', 'y'});

fig10 = figure(10);
grid();
hold on;
scatter(data_t.x, data_t.y, "filled");
plot(data_t.x, exp_noval.predictFcn(data_t), LineWidth=3)
legend(["Data", "Regressie"])
xlabel("x")
ylabel("y")
% saveas(fig10, "Figures/exp_noval_test.jpg")

fig11 = figure(11);
grid();
hold on;
scatter(data_t.x, data_t.y, "filled");
plot(data_t.x, exp_hold.predictFcn(data_t), LineWidth=3)
legend(["Data", "Regressie"])
xlabel("x")
ylabel("y")
% saveas(fig11, "Figures/exp_hold_test.jpg")

fig12 = figure(12);
grid();
hold on;
scatter(data_t.x, data_t.y, "filled");
plot(data_t.x, exp_cross.predictFcn(data_t), LineWidth=3)
legend(["Data", "Regressie"])
xlabel("x")
ylabel("y")
% saveas(fig12, "Figures/exp_cross_test.jpg")

%% Plot sqexp op test data

fig13 = figure(13);
grid();
hold on;
scatter(data_t.x, data_t.y, "filled");
plot(data_t.x, sqexp_noval.predictFcn(data_t), LineWidth=3)
legend(["Data", "Regressie"])
xlabel("x")
ylabel("y")
% saveas(fig13, "Figures/sqexp_noval_test.jpg")

fig14 = figure(14);
grid();
hold on;
scatter(data_t.x, data_t.y, "filled");
plot(data_t.x, sqexp_hold.predictFcn(data_t), LineWidth=3)
legend(["Data", "Regressie"])
xlabel("x")
ylabel("y")
% saveas(fig14, "Figures/sqexp_hold_test.jpg")

fig15 = figure(15);
grid();
hold on;
scatter(data_t.x, data_t.y, "filled");
plot(data_t.x, sqexp_cross.predictFcn(data_t), LineWidth=3)
legend(["Data", "Regressie"])
xlabel("x")
ylabel("y")
% saveas(fig15, "Figures/sqexp_cros_test.jpg")