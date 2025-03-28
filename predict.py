import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score, make_scorer
import numpy as np
from sklearn.feature_selection import RFECV
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.model_selection import KFold,StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from catboost  import CatBoostClassifier, Pool, EShapCalcType, EFeaturesSelectionAlgorithm

# Path of training data
training_data_path = "train_data.csv"

# Path of testing data
testing_data_path = "test_data.csv"

# Load the training data
data = pd.read_csv(training_data_path) #訓練資料
data_test = pd.read_csv(testing_data_path) #測試資料
id = data_test['id']
X_test = data_test.iloc[:, 6:] #測試資料數值特徵
X_test = X_test.drop(['season'], axis=1)
data['home_team_win'] = data['home_team_win'].astype(int)

X = data.iloc[:, 8:] #訓練資料數值特徵
X = X.drop(['season', 'home_team_season', 'away_team_season'], axis=1)
y = data['home_team_win']


X_train_set, X_test_set, y_train_set, y_test_set = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)  
X_train, X_val, y_train, y_val = train_test_split(X_train_set, y_train_set, test_size=0.2, stratify=y_train_set, random_state=42)

folder = StratifiedKFold(n_splits=5,random_state=0,shuffle=True)

g_models = [] #一堆小g-
G_models = [] #一堆大G-

#get best model using XXX model
def init_rf(X, y):
    para_grid_rf = {
        'n_estimators': [200, 100, 400, 300],
        'criterion': ["gini"],
        'max_depth': [2, 3, 4, 5, 6, 7]
    }
    rf = RandomForestClassifier(n_estimators=300, criterion='gini', random_state=42, max_depth=5, n_jobs=-1, verbose=0, class_weight='balanced')
    grid_search_rf = GridSearchCV(
    estimator=rf,
    param_grid=para_grid_rf,
    scoring='accuracy', 
    verbose=0,
    cv=5,
    return_train_score=True,
    n_jobs=-1
    # refit=False
    )
    grid_search_rf.fit(X, y)
    best_model = grid_search_rf.best_estimator_
    return best_model

def init_xgb(X, y, X_val, y_val):
    para_grid_xgb = {
        'max_depth' : [3, 5, 8],
        'learning_rate' : [0.01, 0.05],
        'n_estimators' : [100, 300, 500, 1000] ,
        #  'min_child_weight' : [0, 2, 5],
        #  'max_delta_step' : [ 0, 0.2, 0.6],
        #  'subsample' : [0.6, 0.8, 0.85],
        'reg_alpha' : [0, 0.25, 0.5],
        'reg_lambda' : [0.2, 0.4, 0.6 ]
    }
    xgb = XGBClassifier(reg_lambda=0.2, reg_alpha=0.5, max_delta_step=0.2, early_stopping_rounds=400, min_child_weight= 0, verbosity=3 ,n_estimators=3000, learning_rate= 0.01, objective="binary:logistic", max_depth=3, subsample=0.8,  eval_metric='error')
    grid_search_xgb = GridSearchCV(xgb, param_grid=para_grid_xgb, scoring='accuracy' , cv=5, n_jobs=-1)

    temp_results = pd.DataFrame()
    for i, model in enumerate(g_models):
        column_name = f'model_{i+1}_prob'
        temp_results[column_name] = model.predict_proba(X_val)[:, 1]
    grid_search_xgb.fit(X, y, eval_set=[(temp_results, y_val)])
    best_model = grid_search_xgb.best_estimator_
    return best_model

def init_cat(X, y, X_val, y_val):
    para_grid_cat = {
        'depth': [3, 4, 5, 6],
        'loss_function': ['Logloss', 'CrossEntropy'],
        'l2_leaf_reg': np.logspace(-20, -19, 3),
        'leaf_estimation_iterations': [10],
        'eval_metric': ['Accuracy'],
        'use_best_model': ['True'],
        'logging_level':['Silent'],
        'random_seed': [42]
    }
    cat = CatBoostClassifier(
        iterations=10000,
        learning_rate=0.1,
        depth=4,
        loss_function='Logloss',
        eval_metric='Accuracy',
        # cat_features=categorical_columns,
        random_seed=42,
        # verbose=50,
        l2_leaf_reg=0.2,
        early_stopping_rounds=300
    )    
    grid_search_cat = GridSearchCV(cat, param_grid=para_grid_cat, scoring='accuracy' , cv=5, n_jobs=-1)

    temp_results = pd.DataFrame()
    for i, model in enumerate(g_models):
        column_name = f'model_{i+1}_prob'
        temp_results[column_name] = model.predict_proba(X_val)[:, 1]
        
    grid_search_cat.fit(X, y, eval_set=[(temp_results, y_val)], use_best_model=True)
    best_model = grid_search_cat.best_estimator_
    return best_model

def init_mlp(X, y):
    para_grid_mlp = {
    'hidden_layer_sizes': [(100,200,100), (100,100), (200,300,200)],
    'activation': ['tanh', 'relu'],
    'alpha': np.logspace(-3, 3, 7),
    }
    mlp = MLPClassifier(hidden_layer_sizes=(300,500,300), activation='tanh', solver='adam', alpha=10.0, learning_rate_init=0.001, max_iter=500, early_stopping=True, random_state=42, learning_rate='adaptive', shuffle=True)
 
    grid_search_mlp = GridSearchCV(mlp, para_grid_mlp, cv=3, scoring='accuracy', verbose=3, n_jobs=-1)

    grid_search_mlp.fit(X, y)
    best_model = grid_search_mlp.best_estimator_
    return best_model


model_names = ['rf', 'xgb', 'cat', 'mlp']

# models_dict = {
#     'R': {'init': rf, 'grid_search': grid_search_rf, 'para_grid': para_grid_rf},
#     'X': {'init': xgb, 'grid_search': grid_search_xgb, 'para_grid': para_grid_xgb},
#     'M': {'init': mlp, 'grid_search': grid_search_mlp, 'para_grid': para_grid_mlp},
#     'C': {'init': cat, 'grid_search': grid_search_cat, 'para_grid': para_grid_cat}
# }


# -----------------------------------------------------------------part A------------------------------
for train_idx, test_idx in folder.split(X_train, y_train):
    
    X_train_fold, X_test_fold = X_train.iloc[train_idx], X_train.iloc[test_idx]
    y_train_fold, y_test_fold = y_train.iloc[train_idx], y_train.iloc[test_idx]
    
    best_model = init_rf(X_train_fold, y_train_fold)
    g_models.append(best_model)
  
    y_val_pred = best_model.predict(X_test_fold)
    val_accuracy = accuracy_score(y_test_fold, y_val_pred)
    print("Validation Accuracy: ", val_accuracy)
    

# 初始化一個空的 DataFrame 用於存儲預測結果
results_df = pd.DataFrame({'home_team_win': y_val})




# 將每個模型對 X_val 的預測結果添加到 DataFrame
for i, model in enumerate(g_models):
    column_name = f'model_{i+1}_prob'
    results_df[column_name] = model.predict_proba(X_val)[:, 1]  # 預測為 1 的機率

results_df.to_csv('today1212/model_predictions.csv', index=False)

print("預測結果已儲存為 model_predictions.csv")

# -----------------------------------------------------------------part B------------------------------

z = results_df.drop('home_team_win', axis=1)

for i, name in enumerate(model_names):
    
    # g_search = dict['grid_search'].copy()
    if name == 'rf':
        best_G = init_rf(z, y_val)
    elif name == 'xgb':
        best_G = init_xgb(z, y_val, X_test_set, y_test_set)
    elif name == 'cat':
        best_G = init_cat(z, y_val, X_test_set, y_test_set)
    else:
        best_G = init_mlp(z, y_val)

    G_models.append(best_G)
 
    


results_test = pd.DataFrame({'home_team_win': y_test_set})

for i, model in enumerate(g_models):
    model.fit(X_train_set, y_train_set)
    column_name = f'model_{i+1}_prob'
    results_test[column_name] = model.predict_proba(X_test_set)[:, 1]

for i, model in enumerate(G_models):
    column_name = f'model_{i+6}_prob'
    results_test[column_name] = model.predict_proba(results_test.iloc[:, 1:6])[:, 1]
    
results_test.to_csv('today1213/model_predictions_test.csv', index=False)


# -----------------------------------------------------------------part C------------------------------

G = LogisticRegression(class_weight='balanced')
para_grid_G = {
    'penalty': ['l1', 'l2'],
    'C': [0.01, 0.1, 1, 10, 100, 150, 200],
    'solver': ['liblinear'],
    'max_iter': [100, 50, 30, 15]
}
z_test = results_test.drop('home_team_win', axis=1)
grid_search_G = GridSearchCV(G, para_grid_G, cv=10, scoring='accuracy')
grid_search_G.fit(z_test, y_test_set)
best_para_G = grid_search_G.best_params_
best_score_G = grid_search_G.best_score_
best_G = grid_search_G.best_estimator_
print("best_para_G:", best_para_G)
print("best_score_G:", best_score_G)

results = pd.DataFrame()

for i, model in enumerate(g_models):
    model.fit(X, y)
    column_name = f'model_{i+1}_prob'
    results[column_name] = model.predict_proba(X_test)[:, 1]

for i, model in enumerate(G_models):
    temp_results = pd.DataFrame()
    for j, g in enumerate(g_models):
        column_name = f'model_{j+1}_prob'
        temp_results[column_name] = g.predict_proba(X)[:, 1]
        
    if model_names[i] != 'xgb' and model_names[i] != 'cat':
        model.fit(temp_results, y)
    # model.fit(X, y)
    column_name = f'model_{i+6}_prob'
    results[column_name] = model.predict_proba(results.iloc[:, :5])[:, 1]



y_test_pred = best_G.predict(results)
y_test_pred_binary = [True if pred >= 0.5 else False for pred in y_test_pred]

# Create a DataFrame for the predictions and save to CSV
output_file='prediction.csv'  #改存檔路徑
output = pd.DataFrame({'id': id, 'home_team_win': y_test_pred_binary})
output.to_csv(output_file, index=False)
print(f'Predictions saved to {output_file}')


