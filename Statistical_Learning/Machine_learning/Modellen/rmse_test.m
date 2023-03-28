function [err_hold] = rmse_test(data, model)

y = data.y;
y_hat = model.predictFcn(data);

err_hold = sqrt(mean((y - y_hat).^2));