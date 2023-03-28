%% Clear screen and console
clc; clear;

%% import data
Data = readtable("Data.xlsx");


%% Make linear model without validation and plot
lin_noval = linear_noval(Data);

display(lin_noval)

y1l_fit = lin_noval.predictFcn(Data);

figure(1);
hold on;
scatter(Data, "x", "y", "filled")
plot(Data{:,1}, y1l_fit, LineWidth=5)
grid;
xlabel("Record number")
ylabel("Response value")
saveas(figure(1),[pwd '/Figures/linear_noval.png')

RMSE_lnoval = lin_noval.LinearModel.RMSE();