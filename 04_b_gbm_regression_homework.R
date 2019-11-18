# GBM - model, predicting residence price.

# xgboost -----------------------------------------------------------------

# Create 10 folds of data
ad_split <- vfold_cv(finn, 10) %>% 
  mutate(recipe      = map(splits, prepper, recipe = finn_recipe, retain = FALSE),
         train_raw   = map(splits, training),
         test_raw    = map(splits, testing),
         ad_id       = map(test_raw, ~select(.x, ad_id)),
         train       = map2(recipe, train_raw, bake),
         test        = map2(recipe, test_raw, bake)) %>% 
  select(-test_raw)


# For hyperparameter tuning - send in different variatons of the same model

xg_recipe <- recipe(ad_tot_price ~. , data = finn_train_raw) %>% 
  step_mutate(ad_home_type  = fct_lump(ad_home_type, 4),
              ad_owner_type = fct_lump(ad_owner_type, 3),
              bedrooms_missing = is.na(ad_bedrooms)) %>%
  step_rm(ad_id, 
          ad_price,
          ad_tot_price_per_sqm,
          ad_debt, 
          kommune_no, 
          kommune_name, 
          fylke_name, 
          zip_no, 
          zip_name) %>%
  prep()

finn_train <- bake(xg_recipe, finn_train_raw)
finn_test  <- bake(xg_recipe, finn_test_raw)

# Note: Parameters should be optimized using cross-validation!
xg_mod <- boost_tree(mode = "regression",
                     trees = 300,
                     min_n = 2,
                     tree_depth = 10,
                     learn_rate = 0.15,
                     loss_reduction = 0.9) %>% 
  set_engine("xgboost", tree_method = "exact") %>% 
  fit(ad_tot_price ~ ., data = finn_train)


prediction <- predict(xg_mod, finn_test) %>% 
  bind_cols(finn_test_raw) %>% 
  rename(estimate     = .pred, 
         truth        = ad_tot_price) %>% 
  mutate(abs_dev      = abs(truth - estimate),
         abs_dev_perc = abs_dev/truth)

prediction %>%
  multi_metric(truth = truth, estimate = estimate)

# Get variable importance:
xgboost::xgb.importance(model = xg_mod$fit) %>% 
  xgboost::xgb.ggplot.importance()

# Check out a particular tree:
xgboost::xgb.plot.tree(model = xg_mod$fit, trees = 50)

# Check distribution of predicted vs truth
prediction %>% 
  select(estimate, truth) %>% 
  rownames_to_column(var = "id") %>% 
  pivot_longer(-id, names_to = "type", values_to = "value") %>% 
  ggplot(aes(x = value, fill = type)) +
  geom_density(alpha = 0.3)

prediction %>% 
  select(estimate, truth, fylke_name) %>% 
  rownames_to_column(var = "id") %>% 
  pivot_longer(-c(id, fylke_name), names_to = "type", values_to = "value") %>% 
  ggplot(aes(x = value, fill = type)) +
  geom_density(alpha = 0.3) +
  facet_wrap(~fylke_name)