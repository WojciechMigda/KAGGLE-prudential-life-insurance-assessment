### setup
```python
from sklearn.metrics import make_scorer
qwkappa = make_scorer(kappa, weights='quadratic')

from sklearn.grid_search import GridSearchCV
grid = GridSearchCV(estimator=clf,
                    param_grid=param_grid,
                    cv=10, scoring=qwkappa, n_jobs=2,
                    verbose=1)
```
### RandomForestClassifier
```python
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(random_state=1, n_estimators=10, n_jobs=1)
```
```
Fitting 10 folds for each of 6 candidates, totalling 60 fits
[Parallel(n_jobs=2)]: Done  46 tasks      | elapsed:  9.2min
[Parallel(n_jobs=2)]: Done  60 out of  60 | elapsed: 31.6min finished
grid scores:
  mean: 0.49440, std: 0.01065, params: {'n_estimators': 10}
  mean: 0.51618, std: 0.01632, params: {'n_estimators': 20}
  mean: 0.52326, std: 0.01403, params: {'n_estimators': 50}
  mean: 0.52972, std: 0.01189, params: {'n_estimators': 100}
  mean: 0.52973, std: 0.01495, params: {'n_estimators': 200}
  mean: 0.53241, std: 0.01136, params: {'n_estimators': 500}
best score: 0.53241
best params: {'n_estimators': 500}
```
```
[Parallel(n_jobs=2)]: Done  60 out of  60 | elapsed:  4.5min finished
Fitting 10 folds for each of 6 candidates, totalling 60 fits
grid scores:
  mean: 0.49440, std: 0.01065, params: {'n_estimators': 10, 'criterion': 'gini'}
  mean: 0.51618, std: 0.01632, params: {'n_estimators': 20, 'criterion': 'gini'}
  mean: 0.52326, std: 0.01403, params: {'n_estimators': 50, 'criterion': 'gini'}
  mean: 0.48602, std: 0.01217, params: {'n_estimators': 10, 'criterion': 'entropy'}
  mean: 0.50989, std: 0.01024, params: {'n_estimators': 20, 'criterion': 'entropy'}
  mean: 0.51786, std: 0.01142, params: {'n_estimators': 50, 'criterion': 'entropy'}
best score: 0.52326
best params: {'n_estimators': 50, 'criterion': 'gini'}
```
```
Fitting 10 folds for each of 15 candidates, totalling 150 fits
[Parallel(n_jobs=2)]: Done  46 tasks      | elapsed:   45.3s
[Parallel(n_jobs=2)]: Done 150 out of 150 | elapsed:  4.3min finished
grid scores:
  mean: 0.13220, std: 0.01208, params: {'n_estimators': 10, 'max_depth': 3}
  mean: 0.11826, std: 0.01595, params: {'n_estimators': 20, 'max_depth': 3}
  mean: 0.08574, std: 0.01086, params: {'n_estimators': 50, 'max_depth': 3}
  mean: 0.20403, std: 0.01685, params: {'n_estimators': 10, 'max_depth': 4}
  mean: 0.18797, std: 0.01058, params: {'n_estimators': 20, 'max_depth': 4}
  mean: 0.15386, std: 0.01350, params: {'n_estimators': 50, 'max_depth': 4}
  mean: 0.24975, std: 0.01985, params: {'n_estimators': 10, 'max_depth': 5}
  mean: 0.23813, std: 0.01750, params: {'n_estimators': 20, 'max_depth': 5}
  mean: 0.21102, std: 0.01227, params: {'n_estimators': 50, 'max_depth': 5}
  mean: 0.30216, std: 0.01361, params: {'n_estimators': 10, 'max_depth': 7}
  mean: 0.29943, std: 0.01498, params: {'n_estimators': 20, 'max_depth': 7}
  mean: 0.29258, std: 0.01434, params: {'n_estimators': 50, 'max_depth': 7}
  mean: 0.37534, std: 0.02146, params: {'n_estimators': 10, 'max_depth': 10}
  mean: 0.36558, std: 0.01974, params: {'n_estimators': 20, 'max_depth': 10}
  mean: 0.36396, std: 0.01528, params: {'n_estimators': 50, 'max_depth': 10}
best score: 0.37534
best params: {'n_estimators': 10, 'max_depth': 10}
```
```
Fitting 10 folds for each of 12 candidates, totalling 120 fits
[Parallel(n_jobs=2)]: Done  46 tasks      | elapsed:  1.5min
[Parallel(n_jobs=2)]: Done 120 out of 120 | elapsed:  9.1min finished
grid scores:
  mean: 0.37534, std: 0.02146, params: {'n_estimators': 10, 'max_depth': 10}
  mean: 0.36558, std: 0.01974, params: {'n_estimators': 20, 'max_depth': 10}
  mean: 0.36396, std: 0.01528, params: {'n_estimators': 50, 'max_depth': 10}
  mean: 0.47298, std: 0.01401, params: {'n_estimators': 10, 'max_depth': 20}
  mean: 0.48179, std: 0.01762, params: {'n_estimators': 20, 'max_depth': 20}
  mean: 0.49280, std: 0.01708, params: {'n_estimators': 50, 'max_depth': 20}
  mean: 0.49022, std: 0.01395, params: {'n_estimators': 10, 'max_depth': 50}
  mean: 0.51342, std: 0.01355, params: {'n_estimators': 20, 'max_depth': 50}
  mean: 0.52723, std: 0.01247, params: {'n_estimators': 50, 'max_depth': 50}
  mean: 0.49440, std: 0.01065, params: {'n_estimators': 10, 'max_depth': 100}
  mean: 0.51618, std: 0.01632, params: {'n_estimators': 20, 'max_depth': 100}
  mean: 0.52326, std: 0.01403, params: {'n_estimators': 50, 'max_depth': 100}
best score: 0.52723
best params: {'n_estimators': 50, 'max_depth': 50}
```
```
Fitting 10 folds for each of 12 candidates, totalling 120 fits
[Parallel(n_jobs=2)]: Done  46 tasks      | elapsed:  4.8min
[Parallel(n_jobs=2)]: Done 120 out of 120 | elapsed: 21.4min finished
grid scores:
  mean: 0.48179, std: 0.01762, params: {'n_estimators': 20, 'max_depth': 20}
  mean: 0.49280, std: 0.01708, params: {'n_estimators': 50, 'max_depth': 20}
  mean: 0.49374, std: 0.01184, params: {'n_estimators': 100, 'max_depth': 20}
  mean: 0.51342, std: 0.01355, params: {'n_estimators': 20, 'max_depth': 50}
  mean: 0.52723, std: 0.01247, params: {'n_estimators': 50, 'max_depth': 50}
  mean: 0.53008, std: 0.01294, params: {'n_estimators': 100, 'max_depth': 50}
  mean: 0.51618, std: 0.01632, params: {'n_estimators': 20, 'max_depth': 100}
  mean: 0.52326, std: 0.01403, params: {'n_estimators': 50, 'max_depth': 100}
  mean: 0.52972, std: 0.01189, params: {'n_estimators': 100, 'max_depth': 100}
  mean: 0.51618, std: 0.01632, params: {'n_estimators': 20, 'max_depth': 150}
  mean: 0.52326, std: 0.01403, params: {'n_estimators': 50, 'max_depth': 150}
  mean: 0.52972, std: 0.01189, params: {'n_estimators': 100, 'max_depth': 150}
best score: 0.53008
best params: {'n_estimators': 100, 'max_depth': 50}
```
```
Fitting 10 folds for each of 12 candidates, totalling 120 fits
[Parallel(n_jobs=2)]: Done  46 tasks      | elapsed:  5.1min
[Parallel(n_jobs=2)]: Done 120 out of 120 | elapsed: 44.2min finished
grid scores:
  mean: 0.49246, std: 0.01023, params: {'max_features': 0.02, 'n_estimators': 20, 'max_depth': 50}
  mean: 0.49844, std: 0.00988, params: {'max_features': 0.02, 'n_estimators': 50, 'max_depth': 50}
  mean: 0.50887, std: 0.01593, params: {'max_features': 0.03, 'n_estimators': 20, 'max_depth': 50}
  mean: 0.52013, std: 0.01627, params: {'max_features': 0.03, 'n_estimators': 50, 'max_depth': 50}
  mean: 0.53752, std: 0.01244, params: {'max_features': 0.1, 'n_estimators': 20, 'max_depth': 50}
  mean: 0.54091, std: 0.01068, params: {'max_features': 0.1, 'n_estimators': 50, 'max_depth': 50}
  mean: 0.53761, std: 0.01269, params: {'max_features': 0.2, 'n_estimators': 20, 'max_depth': 50}
  mean: 0.53859, std: 0.01009, params: {'max_features': 0.2, 'n_estimators': 50, 'max_depth': 50}
  mean: 0.53228, std: 0.00530, params: {'max_features': 0.3, 'n_estimators': 20, 'max_depth': 50}
  mean: 0.53077, std: 0.00913, params: {'max_features': 0.3, 'n_estimators': 50, 'max_depth': 50}
  mean: 0.52695, std: 0.00742, params: {'max_features': 0.5, 'n_estimators': 20, 'max_depth': 50}
  mean: 0.52800, std: 0.00910, params: {'max_features': 0.5, 'n_estimators': 50, 'max_depth': 50}
best score: 0.54091
best params: {'max_features': 0.1, 'n_estimators': 50, 'max_depth': 50}
```
```
Fitting 10 folds for each of 10 candidates, totalling 100 fits
[Parallel(n_jobs=2)]: Done  46 tasks      | elapsed:  3.9min
[Parallel(n_jobs=2)]: Done 100 out of 100 | elapsed:  8.7min finished
grid scores:
  mean: 0.51618, std: 0.01632, params: {'n_estimators': 20, 'min_samples_leaf': 1.0}
  mean: 0.52326, std: 0.01403, params: {'n_estimators': 50, 'min_samples_leaf': 1.0}
  mean: 0.51618, std: 0.01632, params: {'n_estimators': 20, 'min_samples_leaf': 1.1}
  mean: 0.52326, std: 0.01403, params: {'n_estimators': 50, 'min_samples_leaf': 1.1}
  mean: 0.51618, std: 0.01632, params: {'n_estimators': 20, 'min_samples_leaf': 1.3}
  mean: 0.52326, std: 0.01403, params: {'n_estimators': 50, 'min_samples_leaf': 1.3}
  mean: 0.51009, std: 0.01191, params: {'n_estimators': 20, 'min_samples_leaf': 1.5}
  mean: 0.52387, std: 0.01257, params: {'n_estimators': 50, 'min_samples_leaf': 1.5}
  mean: 0.50327, std: 0.01382, params: {'n_estimators': 20, 'min_samples_leaf': 2.0}
  mean: 0.50958, std: 0.01438, params: {'n_estimators': 50, 'min_samples_leaf': 2.0}
best score: 0.52387
best params: {'n_estimators': 50, 'min_samples_leaf': 1.5}
```
```
Fitting 10 folds for each of 10 candidates, totalling 100 fits
[Parallel(n_jobs=2)]: Done  46 tasks      | elapsed:  3.8min
[Parallel(n_jobs=2)]: Done 100 out of 100 | elapsed:  8.6min finished
grid scores:
  mean: 0.51342, std: 0.01355, params: {'n_estimators': 20, 'max_depth': 50, 'min_samples_leaf': 1.0}
  mean: 0.52723, std: 0.01247, params: {'n_estimators': 50, 'max_depth': 50, 'min_samples_leaf': 1.0}
  mean: 0.51342, std: 0.01355, params: {'n_estimators': 20, 'max_depth': 50, 'min_samples_leaf': 1.1}
  mean: 0.52723, std: 0.01247, params: {'n_estimators': 50, 'max_depth': 50, 'min_samples_leaf': 1.1}
  mean: 0.51342, std: 0.01355, params: {'n_estimators': 20, 'max_depth': 50, 'min_samples_leaf': 1.3}
  mean: 0.52723, std: 0.01247, params: {'n_estimators': 50, 'max_depth': 50, 'min_samples_leaf': 1.3}
  mean: 0.51223, std: 0.01385, params: {'n_estimators': 20, 'max_depth': 50, 'min_samples_leaf': 1.5}
  mean: 0.52289, std: 0.01240, params: {'n_estimators': 50, 'max_depth': 50, 'min_samples_leaf': 1.5}
  mean: 0.50234, std: 0.01501, params: {'n_estimators': 20, 'max_depth': 50, 'min_samples_leaf': 2.0}
  mean: 0.51029, std: 0.01583, params: {'n_estimators': 50, 'max_depth': 50, 'min_samples_leaf': 2.0}
best score: 0.52723
best params: {'n_estimators': 50, 'max_depth': 50, 'min_samples_leaf': 1.0}
```
```
Fitting 10 folds for each of 3 candidates, totalling 30 fits
[Parallel(n_jobs=2)]: Done  30 out of  30 | elapsed:  7.4min finished
grid scores:
  mean: 0.53880, std: 0.01073, params: {'max_features': 0.07, 'n_estimators': 50, 'criterion': 'gini', 'max_depth': 50}
  mean: 0.54091, std: 0.01068, params: {'max_features': 0.1, 'n_estimators': 50, 'criterion': 'gini', 'max_depth': 50}
  mean: 0.53897, std: 0.01219, params: {'max_features': 0.13, 'n_estimators': 50, 'criterion': 'gini', 'max_depth': 50}
best score: 0.54091
best params: {'max_features': 0.1, 'n_estimators': 50, 'criterion': 'gini', 'max_depth': 50}
```
```
Fitting 10 folds for each of 3 candidates, totalling 30 fits
[Parallel(n_jobs=2)]: Done  30 out of  30 | elapsed:  7.3min finished
grid scores:
  mean: 0.54091, std: 0.01068, params: {'max_features': 0.1, 'n_estimators': 50, 'criterion': 'gini', 'max_depth': 50, 'min_samples_leaf': 1.0}
  mean: 0.54091, std: 0.01068, params: {'max_features': 0.1, 'n_estimators': 50, 'criterion': 'gini', 'max_depth': 50, 'min_samples_leaf': 1.3}
  mean: 0.53488, std: 0.01072, params: {'max_features': 0.1, 'n_estimators': 50, 'criterion': 'gini', 'max_depth': 50, 'min_samples_leaf': 1.5}
best score: 0.54091
best params: {'max_features': 0.1, 'n_estimators': 50, 'criterion': 'gini', 'max_depth': 50, 'min_samples_leaf': 1.0}
```
```
Fitting 10 folds for each of 4 candidates, totalling 40 fits
[Parallel(n_jobs=2)]: Done  40 out of  40 | elapsed: 24.5min finished
grid scores:
  mean: 0.54091, std: 0.01068, params: {'max_features': 0.1, 'n_estimators': 50, 'criterion': 'gini', 'max_depth': 50, 'min_samples_leaf': 1.0}
  mean: 0.53951, std: 0.01028, params: {'max_features': 0.1, 'n_estimators': 100, 'criterion': 'gini', 'max_depth': 50, 'min_samples_leaf': 1.0}
  mean: 0.53869, std: 0.01072, params: {'max_features': 0.1, 'n_estimators': 150, 'criterion': 'gini', 'max_depth': 50, 'min_samples_leaf': 1.0}
  mean: 0.53952, std: 0.00919, params: {'max_features': 0.1, 'n_estimators': 200, 'criterion': 'gini', 'max_depth': 50, 'min_samples_leaf': 1.0}
best score: 0.54091
best params: {'max_features': 0.1, 'n_estimators': 50, 'criterion': 'gini', 'max_depth': 50, 'min_samples_leaf': 1.0}
```

```
### RandomForestRegressor
```python
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(random_state=1, n_estimators=10, n_jobs=1)
```
```
Fitting 10 folds for each of 3 candidates, totalling 30 fits
[Parallel(n_jobs=2)]: Done  30 out of  30 | elapsed: 31.5min finished
grid scores:
  mean: 0.56925, std: 0.01039, params: {'n_estimators': 10, 'criterion': 'mse'}
  mean: 0.58127, std: 0.00800, params: {'n_estimators': 20, 'criterion': 'mse'}
  mean: 0.59008, std: 0.00779, params: {'n_estimators': 50, 'criterion': 'mse'}
best score: 0.59008
best params: {'n_estimators': 50, 'criterion': 'mse'}
```
```
Fitting 10 folds for each of 10 candidates, totalling 100 fits
grid scores:
  mean: 0.42019, std: 0.00781, params: {'n_estimators': 10, 'criterion': 'mse', 'max_depth': 3}
  mean: 0.42095, std: 0.00748, params: {'n_estimators': 20, 'criterion': 'mse', 'max_depth': 3}
  mean: 0.47116, std: 0.00939, params: {'n_estimators': 10, 'criterion': 'mse', 'max_depth': 4}
  mean: 0.47136, std: 0.00946, params: {'n_estimators': 20, 'criterion': 'mse', 'max_depth': 4}
  mean: 0.49639, std: 0.00920, params: {'n_estimators': 10, 'criterion': 'mse', 'max_depth': 5}
  mean: 0.49736, std: 0.00932, params: {'n_estimators': 20, 'criterion': 'mse', 'max_depth': 5}
  mean: 0.50815, std: 0.01132, params: {'n_estimators': 10, 'criterion': 'mse', 'max_depth': 7}
  mean: 0.51206, std: 0.00977, params: {'n_estimators': 20, 'criterion': 'mse', 'max_depth': 7}
  mean: 0.53389, std: 0.00926, params: {'n_estimators': 10, 'criterion': 'mse', 'max_depth': 10}
  mean: 0.53589, std: 0.00884, params: {'n_estimators': 20, 'criterion': 'mse', 'max_depth': 10}
best score: 0.53589
best params: {'n_estimators': 20, 'criterion': 'mse', 'max_depth': 10}
```
```
Fitting 10 folds for each of 4 candidates, totalling 40 fits
grid scores:
  mean: 0.57604, std: 0.00655, params: {'n_estimators': 20, 'criterion': 'mse', 'max_depth': 20}
  mean: 0.58317, std: 0.00646, params: {'n_estimators': 20, 'criterion': 'mse', 'max_depth': 50}
  mean: 0.58127, std: 0.00800, params: {'n_estimators': 20, 'criterion': 'mse', 'max_depth': 100}
  mean: 0.58127, std: 0.00800, params: {'n_estimators': 20, 'criterion': 'mse', 'max_depth': 200}
best score: 0.58317
best params: {'n_estimators': 20, 'criterion': 'mse', 'max_depth': 50}
```
```
Fitting 10 folds for each of 4 candidates, totalling 40 fits
grid scores:
  mean: 0.49596, std: 0.00791, params: {'max_features': 0.02, 'n_estimators': 50, 'criterion': 'mse', 'max_depth': 50}
  mean: 0.52116, std: 0.00762, params: {'max_features': 0.03, 'n_estimators': 50, 'criterion': 'mse', 'max_depth': 50}
  mean: 0.57648, std: 0.00465, params: {'max_features': 0.1, 'n_estimators': 50, 'criterion': 'mse', 'max_depth': 50}
  mean: 0.58956, std: 0.00799, params: {'max_features': 0.2, 'n_estimators': 50, 'criterion': 'mse', 'max_depth': 50}
best score: 0.58956
best params: {'max_features': 0.2, 'n_estimators': 50, 'criterion': 'mse', 'max_depth': 50}
```
```
Fitting 10 folds for each of 4 candidates, totalling 40 fits
[Parallel(n_jobs=2)]: Done  40 out of  40 | elapsed: 47.3min finished
grid scores:
  mean: 0.59056, std: 0.00727, params: {'max_features': 0.3, 'n_estimators': 50, 'criterion': 'mse', 'max_depth': 50}
  mean: 0.59077, std: 0.00655, params: {'max_features': 0.5, 'n_estimators': 50, 'criterion': 'mse', 'max_depth': 50}
  mean: 0.59099, std: 0.00730, params: {'max_features': 0.7, 'n_estimators': 50, 'criterion': 'mse', 'max_depth': 50}
  mean: 0.58974, std: 0.00875, params: {'max_features': 0.9, 'n_estimators': 50, 'criterion': 'mse', 'max_depth': 50}
best score: 0.59099
best params: {'max_features': 0.7, 'n_estimators': 50, 'criterion': 'mse', 'max_depth': 50}
```
```
Fitting 10 folds for each of 3 candidates, totalling 30 fits
[Parallel(n_jobs=2)]: Done  30 out of  30 | elapsed: 18.7min finished
grid scores:
  mean: 0.59056, std: 0.00727, params: {'max_features': 0.3, 'n_estimators': 50, 'criterion': 'mse', 'max_depth': 50, 'min_samples_leaf': 1.0}
  mean: 0.59056, std: 0.00727, params: {'max_features': 0.3, 'n_estimators': 50, 'criterion': 'mse', 'max_depth': 50, 'min_samples_leaf': 1.3}
  mean: 0.58740, std: 0.00757, params: {'max_features': 0.3, 'n_estimators': 50, 'criterion': 'mse', 'max_depth': 50, 'min_samples_leaf': 2.0}
best score: 0.59056
best params: {'max_features': 0.3, 'n_estimators': 50, 'criterion': 'mse', 'max_depth': 50, 'min_samples_leaf': 1.0}
```

