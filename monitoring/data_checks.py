def simple_drift_check(train_mean, live_mean, threshold=0.2):
    return abs(train_mean - live_mean) > threshold
