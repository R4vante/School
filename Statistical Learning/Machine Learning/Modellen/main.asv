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
plot_model(data, lin_noval, 1, "lin_noval")
plot_model(data, lin_hold, 1, "lin_hold")
plot_model(data, lin_cross, 1, "lin_cross")

%% Plot expGPR

plot_model(data, exp_noval, 1, "exp_noval")
plot_model(data, exp_hold, 1, "exp_hold")
plot_model(data, exp_cross, 1, "exp_cross")
%% Plot sqrexp

plot_model(data, sqexp_noval, 1, "sqexp_noval")
plot_model(data, sqexp_hold, 1, "sqexp_hold")
plot_model(data, sqexp_cross, 1, "sqexp_cross")


%% Genereer testdata
data_t = readtable("test_data.xlsx");


%% Bereken test rmse's

err_lin_noval = rmse_test(data_t, lin_noval);
err_lin_hold = rmse_test(data_t, lin_hold);
err_lin_cross = rmse_test(data_t, lin_cross);

err_exp_noval = rmse_test(data_t, exp_noval);
err_exp_hold = rmse_test(data_t, exp_hold);
err_exp_cross = rmse_test(data_t, exp_cross);

err_sqexp_noval = rmse_test(data_t, sqexp_noval);
err_sqexp_hold = rmse_test(data_t, sqexp_hold);
err_sqexp_cross = rmse_test(data_t, sqexp_cross);

%% Plot lineair op test data

plot_model(data_t, lin_noval,1,'test_lin_noval')
plot_model(data_t, lin_hold, 1, 'test_lin_hold')
plot_model(data_t, lin_cross, 1, 'test_lin_cross')

%% Plot exp op test data

plot_model(data_t, exp_noval,1,'test_exp_noval')
plot_model(data_t, exp_hold, 1, 'test_exp_hold')
plot_model(data_t, exp_cross, 1, 'test_exp_cross')

%% Plot sqexp op test data

plot_model(data_t, sqexp_noval,1,'test_sqexp_noval')
plot_model(data_t, sqexp_hold, 1, 'test_sqexp_hold')
plot_model(data_t, sqexp_cross, 1, 'test_sqexp_cross')

%% Plot lineair actual vs predicted op test data

predplot(data_t, lin_noval, 1, "avp_lin_noval")
predplot(data_t, lin_hold, 1, "avp_lin_hold")
predplot(data_t, lin_cross, 1, "avp_lin_cross")

%% Plot exp actual vs predicted op test data

predplot(data_t, exp_noval, 1, "avp_exp_noval")
predplot(data_t, exp_hold, 1, "avp_exp_hold")
predplot(data_t, exp_cross, 1, "avp_exp_cross")

%% Plot sqexp actual vs predicted op test data

predplot(data_t, sqexp_noval, 1, "avp_sqexp_noval")
predplot(data_t, sqexp_hold, 1, "avp_sqexp_hold")
predplot(data_t, sqexp_cross, 1, "avp_sqexp_cross")

%% sqexp extrapoleren -> functie generate dataset wordt aangepast voor groter bereik.

data_e = generate_extrapolation(12*pi, 250, 0.4);

plot_model(data_e, sqexp_cross, 1, "sqexp_extra_12")

data_e2 = generate_extrapolation(5*pi, 250, 0.4);

plot_model(data_e2, sqexp_cross, 1, "sqexp_extra_5")


%% sqexp ruis verhoging -> functie generate data set meermaals oproepen


ruis = linspace(0, 1, 5);
rmse_new = [];

for i=1:length(ruis)
    % make new dataset
    data_n = generate_dataset(250, ruis(i));
    % retrain model
    [sqexp_model, rmse] = sqExp_cross(data_n);
    rmse_new = [rmse_new, rmse];
    % plot model
    subplot(3,2,i)
    ruis_t = ruis(i);
    hold on;
    scatter(data_n.x, data_n.y, 'filled')
    plot(data_n.x, sqexp_model.predictFcn(data_n), LineWidth=3)
    title("Noise: " +  ruis_t +  " RMSE: " + rmse)
end


