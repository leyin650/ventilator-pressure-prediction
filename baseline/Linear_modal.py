import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from sklearn.model_selection import RepeatedKFold
import numpy as np
from sklearn.model_selection import KFold


def read_train_data():  # Read files
    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')
    sub = pd.read_csv('sample_submission.csv')
    return train, test, sub


def check_data(dataset):  # Examine the data set for various information
    print(dataset.head())
    print(dataset.describe())
    print(dataset.columns.isnull())
    print(dataset.time_step.value_counts())
    print(len(dataset['breath_id'].unique()))


def draw_breath(dataset):  # Looking at the images of the first 400 data points, we guessed that there was an equal amount of data per breath
    dataset['breath_id'][:400].plot()


def get_breath_lengths(dataset):  # Based on a guess, it works out that there are 80 data points per breath (same id)
    breath_lengths = dataset[['id', 'breath_id']].groupby('breath_id').count()['id']
    breath_lengths.unique()
    BREATH_LENGTH = breath_lengths.unique()[0]
    return BREATH_LENGTH


def cal_std(dataset):  # Calculate the standard deviation of R and C for the same breath, we find that the standard deviation results in 0
    r_c_std_in_breaths = dataset[['breath_id', 'R', 'C']].groupby('breath_id').std()  # , so we know 80 data for the same id breath,R and C unchanged
    print(r_c_std_in_breaths['R'].unique())
    print(r_c_std_in_breaths['C'].unique())


def sum_breath(dataset, x):  # Here we count the total number of breaths in the data set
    r_values = dataset[['breath_id', x]].groupby('breath_id').mean()[x]
    print(r_values)
    print()
    print('Unique values:')
    print(r_values.value_counts())
    r_unique = np.sort(r_values.unique()).astype(int)
    return r_unique


def draw_RC(dataset, R, C):  # Combine different R and C cases in pairs, a total of 9 cases, and then count, observe the image
    rc_values = np.array([
        [r, c, len(dataset[(dataset['R'] == r) & (dataset['C'] == c)]) // BREATH_LENGTH]
        for r in R
        for c in C
    ])
    x = range(len(rc_values))
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(x, rc_values[:, 2], width=0.8, align='center')
    ax.set_xticks(x)
    ax.set_xticklabels([str(r) + '_' + str(c) for r, c in rc_values[:, :2]])
    ax.set_xlabel('R_C')
    ax.set_ylabel('numbers')
    plt.show()


def draw_timestep(dataset, x, y):  # Take any two existing Breath_id here, and you can see that the lines in the timestep image almost coincide
    breath1 = dataset[dataset['breath_id'] == x]  # To prove that the difference in time steps of different breaths is likely to be very small
    breath2 = dataset[dataset['breath_id'] == y]  # , so we can assume that step = total time/number

    x = range(BREATH_LENGTH)
    t1 = breath1['time_step']
    t2 = breath2['time_step']
    plt.plot(x, t1)
    plt.plot(x, t2, ls=':')
    plt.show()
    breath_timestep = (max(t1) - min(t1)) / BREATH_LENGTH

    return breath_timestep


def draw_pressure_uin(dataset, number):  # Drawing the images of pressure and u_in, we found that the fluctuation of these two groups of data is close
    plt.plot(dataset.pressure[:number])  # , so we can try to fit them
    plt.plot(dataset.u_in[:number])
    plt.show()


def Zscore_Normalization(dataset, column):  # Zscore normalization
    a = dataset[column].tolist()
    mean_a = np.mean(a)
    std_a = np.std(a)
    a = (a - mean_a) / std_a

    n = 80  # Data slicing, using small pieces of data for training step by step, to avoid memory insufficiency caused by loading all data into the memory at once
    a = [a[i:i + n] for i in range(0, len(a), n)]
    return a


def Zscore_Denormalization(dataset, column, predict):  # Zscore inverse normalization, restore y axis coordinates
    a = dataset[column].tolist()
    mean_a = np.mean(a)
    std_a = np.std(a)
    test_predictions = predict * std_a + mean_a

    #print(test_predictions.shape)
    #print(test_predictions[0])
    return test_predictions


def Minmax_Normalization(dataset, column):  # Minmax normalization
    a = dataset[column].tolist()
    tst_minimo = np.min(a)
    tst_maximo = np.max(a)
    a -= tst_minimo
    a /= tst_maximo

    n = 80
    a = [a[i:i + n] for i in range(0, len(a), n)]
    return a


def Minmax_Denormalization(dataset, column, predict):  # Minmax inverse normalization
    minimo = np.min(dataset[column].tolist())
    maximo = np.max(dataset[column].tolist())
    test_predictions = predict * maximo + minimo

    #print(test_predictions.shape)
    #print(test_predictions[0])
    return test_predictions


def Model_Linear(X, train_y):  # Linear regression model
    lin_reg = LinearRegression()
    lin_reg.fit(X, train_y)
    train_predictions = lin_reg.predict(X)
    plt.plot(train_predictions[1])
    return lin_reg


def create_pressure_list(test_predictions):  # The model's prediction results on the test set are converted from a multidimensional array to a one-dimensional array
    test_predictions = np.array(test_predictions)  # ,the pressure values for each breathing cycle are then placed in the list "pressure"
    samples, _ = test_predictions.shape
    pressure = []
    test_predictions = list(test_predictions)

    for signal in range(samples):
        breath_pressure = test_predictions[signal - 1]
        pressure.extend(breath_pressure)

    #print(len(pressure))
    return pressure


train, test, sub = read_train_data()
train['in_amount'] = train['time_step'] * train['u_in']
check_data(train)

draw_breath(train)

BREATH_LENGTH = get_breath_lengths(train)

cal_std(train)

r_unique = sum_breath(train, 'R')
c_unique = sum_breath(train, 'C')

draw_RC(train, r_unique, c_unique)

breath_timestep = draw_timestep(train, 5, 6)

draw_pressure_uin(train, 1000)

# Generate a new column in_amount
train['in_amount'] = train['time_step'] * train['u_in']
test['in_amount'] = test['time_step'] * test['u_in']


def compute_model_minimax(train, k_folds=5):
    kf = KFold(n_splits=k_folds, shuffle=True)
    mse_scores = []
    rmse_scores = []
    mae_scores = []
    r2_scores = []
    adjusted_r2_scores = []
    for train_index, test_index in kf.split(train):
        train_z = train.iloc[train_index]
        test_z = train.iloc[test_index]
        h = test_z["pressure"].tolist()

        X_z = Minmax_Normalization(train_z, 'u_in')
        train_y_z = Minmax_Normalization(train_z, 'pressure')
        X_test_z = Minmax_Normalization(test_z, 'u_in')
        lin_reg_minimax = Model_Linear(X_z, train_y_z)
        test_predictions_z = lin_reg_minimax.predict(X_test_z)

        test_predictions_z = Minmax_Denormalization(train_z, 'pressure', test_predictions_z)

        test_predictions = np.array(test_predictions_z)
        samplesnum, m = test_predictions.shape

        predict = create_pressure_list(test_predictions_z)

        mse_scores.append(mean_squared_error(h, predict))
        rmse_scores.append(np.sqrt(mean_squared_error(h, predict)))
        mae_scores.append(mean_absolute_error(h, predict))
        r2_scores.append(r2_score(h, predict))
        n = samplesnum
        p = 2
        adjusted_r2_scores.append(1 - ((1 - r2_scores[-1]) * (n - 1)) / (n - p - 1))
    modelkey = []
    modelkey.append(np.mean(mse_scores))
    modelkey.append(np.mean(rmse_scores))
    modelkey.append(np.mean(mae_scores))
    modelkey.append(np.mean(r2_scores))
    modelkey.append(np.mean(adjusted_r2_scores))
    print()
    print("MSE: {:.4f} std+/- {:.4f}".format(np.mean(mse_scores), np.std(mse_scores)))
    print("RMSE: {:.4f} std+/- {:.4f}".format(np.mean(rmse_scores), np.std(rmse_scores)))
    print("MAE: {:.4f} std+/- {:.4f}".format(np.mean(mae_scores), np.std(mae_scores)))
    print("R2: {:.4f} std+/- {:.4f}".format(np.mean(r2_scores), np.std(r2_scores)))
    print("Adjusted R2: {:.4f} std+/- {:.4f}".format(np.mean(adjusted_r2_scores), np.std(adjusted_r2_scores)))
    print()

    return modelkey


def compute_model_minimaxin_amount(train, k_folds=5):
    kf = KFold(n_splits=k_folds, shuffle=True)
    mse_scores = []
    rmse_scores = []
    mae_scores = []
    r2_scores = []
    adjusted_r2_scores = []
    for train_index, test_index in kf.split(train):
        train_z = train.iloc[train_index]
        test_z = train.iloc[test_index]
        h = test_z["pressure"].tolist()

        X_z = Minmax_Normalization(train_z, 'in_amount')
        train_y_z = Minmax_Normalization(train_z, 'pressure')
        X_test_z = Minmax_Normalization(test_z, 'in_amount')
        lin_reg_minimaxIN = Model_Linear(X_z, train_y_z)
        test_predictions_z = lin_reg_minimaxIN.predict(X_test_z)

        test_predictions_z = Minmax_Denormalization(train_z, 'pressure', test_predictions_z)

        test_predictions = np.array(test_predictions_z)
        samplesnum, m = test_predictions.shape

        predict = create_pressure_list(test_predictions_z)

        mse_scores.append(mean_squared_error(h, predict))
        rmse_scores.append(np.sqrt(mean_squared_error(h, predict)))
        mae_scores.append(mean_absolute_error(h, predict))
        r2_scores.append(r2_score(h, predict))
        n = samplesnum
        p = 2
        adjusted_r2_scores.append(1 - ((1 - r2_scores[-1]) * (n - 1)) / (n - p - 1))
    modelkey = []
    modelkey.append(np.mean(mse_scores))
    modelkey.append(np.mean(rmse_scores))
    modelkey.append(np.mean(mae_scores))
    modelkey.append(np.mean(r2_scores))
    modelkey.append(np.mean(adjusted_r2_scores))
    print("MSE: {:.4f} std+/- {:.4f}".format(np.mean(mse_scores), np.std(mse_scores)))
    print("RMSE: {:.4f} std+/- {:.4f}".format(np.mean(rmse_scores), np.std(rmse_scores)))
    print("MAE: {:.4f} std+/- {:.4f}".format(np.mean(mae_scores), np.std(mae_scores)))
    print("R2: {:.4f} std+/- {:.4f}".format(np.mean(r2_scores), np.std(r2_scores)))
    print("Adjusted R2: {:.4f} std+/- {:.4f}".format(np.mean(adjusted_r2_scores), np.std(adjusted_r2_scores)))
    print()

    return modelkey


def compute_model_zscore(train, k_folds=5):
    kf = KFold(n_splits=k_folds, shuffle=True)
    mse_scores = []
    rmse_scores = []
    mae_scores = []
    r2_scores = []
    adjusted_r2_scores = []
    for train_index, test_index in kf.split(train):
        train_z = train.iloc[train_index]
        test_z = train.iloc[test_index]
        h = test_z["pressure"].tolist()

        X_z = Zscore_Normalization(train_z, 'u_in')
        train_y_z = Zscore_Normalization(train_z, 'pressure')
        X_test_z = Zscore_Normalization(test_z, 'u_in')
        lin_reg_Zscore = Model_Linear(X_z, train_y_z)
        test_predictions_z = lin_reg_Zscore.predict(X_test_z)

        test_predictions_z = Zscore_Denormalization(train_z, 'pressure', test_predictions_z)

        test_predictions = np.array(test_predictions_z)
        samplesnum, m = test_predictions.shape

        predict = create_pressure_list(test_predictions_z)

        mse_scores.append(mean_squared_error(h, predict))
        rmse_scores.append(np.sqrt(mean_squared_error(h, predict)))
        mae_scores.append(mean_absolute_error(h, predict))
        r2_scores.append(r2_score(h, predict))
        n = samplesnum
        p = 2
        adjusted_r2_scores.append(1 - ((1 - r2_scores[-1]) * (n - 1)) / (n - p - 1))
    modelkey = []
    modelkey.append(np.mean(mse_scores))
    modelkey.append(np.mean(rmse_scores))
    modelkey.append(np.mean(mae_scores))
    modelkey.append(np.mean(r2_scores))
    modelkey.append(np.mean(adjusted_r2_scores))
    print("MSE: {:.4f} std+/- {:.4f}".format(np.mean(mse_scores), np.std(mse_scores)))
    print("RMSE: {:.4f} std+/- {:.4f}".format(np.mean(rmse_scores), np.std(rmse_scores)))
    print("MAE: {:.4f} std+/- {:.4f}".format(np.mean(mae_scores), np.std(mae_scores)))
    print("R2: {:.4f} std+/- {:.4f}".format(np.mean(r2_scores), np.std(r2_scores)))
    print("Adjusted R2: {:.4f} std+/- {:.4f}".format(np.mean(adjusted_r2_scores), np.std(adjusted_r2_scores)))
    print()

    return modelkey


def create_list(x1, x2, x3):
    lst = [x1, x2, x3]

    transposed = list(zip(*lst))
    m = []
    for i in transposed:
        m.append(list(i))
    return m


def plotsho(L):
    mse = L[0]
    rmse = L[1]
    mae = L[2]
    r2 = L[3]
    adj_r2 = L[4]

    # tag
    labels = ['minimax', 'minimaxin_amount', 'z_score']

    # The location of each indicator
    x = np.arange(len(labels))

    # Width of each indicator
    width = 0.15

    # Draw a bar chart
    fig, ax = plt.subplots()
    ax.bar(x - 2 * width, mse, width, label='MSE')
    ax.bar(x - width, rmse, width, label='RMSE')
    ax.bar(x, mae, width, label='MAE')
    ax.bar(x + width, r2, width, label='R2')
    ax.bar(x + 2 * width, adj_r2, width, label='Adjusted-R2')

    # Add tags, titles, and legends
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel('model')
    ax.set_ylabel('Scores')
    ax.set_title('compare three models')
    plt.ylim(-2, None)
    ax.legend()

    fig.tight_layout()

    plt.show()


m = compute_model_minimax(train)
n = compute_model_minimaxin_amount(train)
l = compute_model_zscore(train)
L = create_list(m, n, l)

plotsho(L)
print("minimax", m)
print("minimax_in_amount", n)
print("z_score", l)
print()

# z_score model
X = Zscore_Normalization(train, 'u_in')
train_y = Zscore_Normalization(train, 'pressure')
X_test = Zscore_Normalization(test, 'u_in')

lin_reg = Model_Linear(X, train_y)
test_predictions = lin_reg.predict(X_test)
plt.plot(test_predictions[0])
# print("z_score")
test_predictions = Zscore_Denormalization(train, 'pressure', test_predictions)
plt.plot(test_predictions[0])

pressure1 = create_pressure_list(test_predictions)

# minimax model

X1 = Minmax_Normalization(train, 'u_in')
train_y1 = Minmax_Normalization(train, 'pressure')
X_test1 = Minmax_Normalization(test, 'u_in')
lin_reg1 = Model_Linear(X1, train_y1)
test_predictions1 = lin_reg1.predict(X_test1)
plt.plot(test_predictions1[0])
# print("minimax")
test_predictions1 = Minmax_Denormalization(train, 'pressure', test_predictions1)
plt.plot(test_predictions1[0])
pressure2 = create_pressure_list(test_predictions1)

# minimax_inamount model

X2 = Minmax_Normalization(train, 'in_amount')
train_y2 = Minmax_Normalization(train, 'pressure')
X_test2 = Minmax_Normalization(test, 'in_amount')
lin_reg2 = Model_Linear(X2, train_y2)
test_predictions2 = lin_reg2.predict(X_test2)
plt.plot(test_predictions2[0])
# print("minimaxin_amount")
test_predictions2 = Minmax_Denormalization(train, 'pressure', test_predictions2)
plt.plot(test_predictions2[0])
pressure3 = create_pressure_list(test_predictions1)

PRE = np.sum([pressure3, pressure1, pressure2], axis=0)
PRE = np.divide(PRE, 3)
PRE = PRE.tolist()
# Average the three linear models and output the results
sub.pressure = PRE

print(sub.head())
sub.to_csv('submission_regression.csv', index=False)
