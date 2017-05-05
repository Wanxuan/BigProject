learn_rate = [0.00001, 0.0001, 0.001, 0.1, 0.2, 0.3]
momentum = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9]
param_grid = dict(learn_rate=learn_rate, momentum=momentum)
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1)
grid_result = grid.fit(x, y)

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_param_))
for params, mean_score, scores in grid_result.grid_scores_:
  print("%f (%f) with: %r" % (scores.mean(), scores.std(), params))
