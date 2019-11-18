# GLM model - prediction if residence is "expensive" or "cheap"

glm_recipe <- recipe(ad_tot_price ~. , data = finn_train_raw) %>% 
  step_mutate(ad_home_type  = fct_lump(ad_home_type, 4),
              ad_owner_type = fct_lump(ad_owner_type, 2),
              bedrooms_missing = is.na(ad_bedrooms),
              is_expensive = as.factor(ad_tot_price > 4000000)) %>%
  step_medianimpute(all_numeric()) %>%
  step_modeimpute(all_nominal()) %>% 
  step_rm(ad_id) %>% 
  prep()

finn_train <- bake(glm_recipe, finn_train_raw)
finn_test  <- bake(glm_recipe, finn_test_raw)


# Create 10 folds of data
ad_split <- vfold_cv(finn, 10) %>% 
  mutate(recipe      = map(splits, prepper, recipe = finn_recipe, retain = FALSE),
         train_raw   = map(splits, training),
         test_raw    = map(splits, testing),
         ad_id       = map(test_raw, ~select(.x, ad_id)),
         train       = map2(recipe, train_raw, bake),
         test        = map2(recipe, test_raw, bake)) %>% 
  select(-test_raw)

glm_mod <- logistic_reg() %>%
  set_engine("glm") %>%
  fit(
    is_expensive ~ 
      ad_owner_type
    + ad_home_type
    + ad_bedrooms
    + ad_sqm
    + ad_expense
    + avg_income
    + ad_built
    + bedrooms_missing,
    data = finn_train
  )

# View summary
summary(glm_mod$fit)

prediction <- predict(glm_mod, finn_test, type = "prob") %>% 
  bind_cols(finn_test) %>% 
  rename(estimate     = .pred_TRUE, 
         truth        = is_expensive)

# Evaluate model (NOTE: we need different metrics since this is classification!)
prediction %>%
  yardstick::roc_auc(truth, estimate)

prediction %>%
  yardstick::roc_curve(truth = truth, estimate = estimate, na_rm = T) %>% 
  autoplot()

# Legge inn resultater i tabell-format

glm.pred <- rep("Cheap", 5881)
glm.pred[prediction$estimate>.5] <- "Expensive"
actual <- as.data.frame(prediction$truth) %>% 
  mutate(., actual = if_else(.=="TRUE", "Expensive", "Cheap"))
table(glm.pred, actual$actual)

