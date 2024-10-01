import numpy as np

best_val_metrics = np.array([0.01749697, 0.01515842, 0.01671512, 0.0175197 , 0.01729757])
best_test_metrics = np.array([0.0133121 , 0.01151167, 0.01262056, 0.0132021 , 0.0133246])

val_mean = np.mean(best_val_metrics)
val_std = np.std(best_val_metrics)

test_mean = np.mean(best_test_metrics)
test_std = np.std(best_test_metrics)

print(f'Val Mean: {val_mean}, Val Std: {val_std}')
print(f'Test Mean: {test_mean}, Test Std: {test_std}')