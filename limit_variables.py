def limit_variables(X, value_range):
    for i in range(len(X)):
        if X[i] < value_range[0]:
            X[i] = value_range[0]
        elif X[i] > value_range[1]:
            X[i] = value_range[1]
    return X