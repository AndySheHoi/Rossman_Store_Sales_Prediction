import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from math import sqrt
from EDA import EDA

# import the dataset
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
store = pd.read_csv('store.csv')

# =============================================================================
# EDA
# =============================================================================
train, test, store = EDA.eda(train, test, store)


# =============================================================================
# Feature Engineering
# =============================================================================
store.isnull().sum()


# convert types of categorical values to numbers
store.head()
store['StoreType'] = store['StoreType'].astype('category').cat.codes
store['Assortment'] = store['Assortment'].astype('category').cat.codes
train["StateHoliday"] = train["StateHoliday"].astype('category').cat.codes

merged = pd.merge(train, store, on='Store', how='left')

# remove NaNs
merged.isnull().sum()
merged.fillna(0, inplace=True)

# split datetime values
merged['Date'] = pd.to_datetime(merged['Date'])
merged['Year'] = merged.Date.dt.year
merged['Month'] = merged.Date.dt.month
merged['Day'] = merged.Date.dt.day
merged['Week'] = merged.Date.dt.week

# Number of months that competition has existed for
merged['MonthsCompetitionOpen'] = 12 * (merged['Year'] - merged['CompetitionOpenSinceYear']) + (merged['Month'] - merged['CompetitionOpenSinceMonth'])
merged.loc[merged['CompetitionOpenSinceYear'] == 0, 'MonthsCompetitionOpen'] = 0

# Number of weeks that promotion has existed for
merged['WeeksPromoOpen'] = 12 * (merged['Year'] - merged['Promo2SinceYear']) + (merged['Date'].dt.weekofyear - merged['Promo2SinceWeek'])
merged.loc[merged['Promo2SinceYear'] == 0, 'WeeksPromoOpen'] = 0

toInt_list = [
                'CompetitionOpenSinceMonth',
                'CompetitionOpenSinceYear',
                'Promo2SinceWeek', 
                'Promo2SinceYear', 
                'MonthsCompetitionOpen', 
                'WeeksPromoOpen'
             ]

merged[toInt_list] = merged[toInt_list].astype(int)

med_store = train.groupby('Store')[['Sales', 'Customers', 'SalesPerCustomer']].median()
med_store.rename(columns=lambda x: 'Med' + x, inplace=True)

store = pd.merge(med_store.reset_index(), store, on='Store')
merged = pd.merge(med_store.reset_index(), merged, on='Store')

merged.hist(figsize=(20,20))
plt.show()


# =============================================================================
# Model Building and Evaluation
# =============================================================================

y = np.log(merged['Sales'])
X = merged.drop(columns = ['Date', 'Sales', 'Open', 'SchoolHoliday', 'SalesPerCustomer', 
                           'PromoInterval', 'MonthsCompetitionOpen', 'WeeksPromoOpen'])

sc = StandardScaler()
X_scaled =pd.DataFrame(sc.fit_transform(X))

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.20, random_state=10)

def plot_importance(model):
    k = list(zip(X, model.feature_importances_))
    k.sort(key=lambda tup: tup[1])

    labels, vals = zip(*k)
    
    plt.barh(np.arange(len(X)), vals, align='center')
    plt.yticks(np.arange(len(X)), labels)
    
param ={
            'n_estimators': [100,500, 1000,1500],
            'max_depth':[2,4,6,8]
        }

xgboost_tree = xgb.XGBRegressor(
    eta = 0.1,
    min_child_weight = 2,
    subsample = 0.8,
    colsample_bytree = 0.8,
    tree_method = 'exact',
    reg_alpha = 0.05,
    silent = 0,
    random_state = 1023
)

grid = GridSearchCV(estimator=xgboost_tree,param_grid=param,cv=5,  verbose=1, n_jobs=-1,scoring='neg_mean_squared_error')
    
grid_result = grid.fit(X_train, y_train)
best_params = grid_result.best_params_

print('Best Params :',best_params)

pred = grid_result.predict(X_test)
print('Root Mean squared error {}'.format(sqrt(mean_squared_error(np.exp(y_test), np.exp(pred)))))