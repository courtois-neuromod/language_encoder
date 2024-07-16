min_val_mse = 1e4


for alpha in tqdm(
    [0.1, 1, 10, 100, 1000, 1e4, 5e4, 1e5, 1e6, 1e7, 1e8], desc="fitting ridge models"
):
    ridge = Ridge(alpha=alpha)
    ridge.fit(tng_activations, tng_bold)
    tng_preds = ridge.predict(tng_activations)
    tng_mse = mean_squared_error(tng_bold, tng_preds)
    tng_r2 = r2_score(tng_bold, tng_preds, multioutput="raw_values")
    val_preds = ridge.predict(val_activations)
    val_mse = mean_squared_error(val_bold, val_preds)
    val_r2 = r2_score(val_bold, val_preds, multioutput="raw_values")
    if val_mse < min_val_mse:
        best_ridge = ridge
        best_val_r2 = val_r2
        best_tng_r2 = tng_r2
        min_val_mse = val_mse
