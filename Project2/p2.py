#!/usr/bin/env python
# coding: utf-8

# ___
# 
# <p style="text-align: center;"><img src="https://docs.google.com/uc?id=1lY0Uj5R04yMY3-ZppPWxqCr5pvBLYPnV" class="img-fluid" alt="CLRSWY"></p>
# 
# ___

# # WELCOME!

# In this project, you must apply EDA processes for the development of predictive models. Handling outliers, domain knowledge and feature engineering will be challenges.
# 
# Also, this project aims to improve your ability to implement algorithms for Multi-Class Classification. Thus, you will have the opportunity to implement many algorithms commonly used for Multi-Class Classification problems.
# 
# Before diving into the project, please take a look at the determines and tasks.

# # Determines

# The 2012 US Army Anthropometric Survey (ANSUR II) was executed by the Natick Soldier Research, Development and Engineering Center (NSRDEC) from October 2010 to April 2012 and is comprised of personnel representing the total US Army force to include the US Army Active Duty, Reserves, and National Guard. In addition to the anthropometric and demographic data described below, the ANSUR II database also consists of 3D whole body, foot, and head scans of Soldier participants. These 3D data are not publicly available out of respect for the privacy of ANSUR II participants. The data from this survey are used for a wide range of equipment design, sizing, and tariffing applications within the military and has many potential commercial, industrial, and academic applications.
# 
# The ANSUR II working databases contain 93 anthropometric measurements which were directly measured, and 15 demographic/administrative variables explained below. The ANSUR II Male working database contains a total sample of 4,082 subjects. The ANSUR II Female working database contains a total sample of 1,986 subjects.
# 
# 
# DATA DICT:
# https://data.world/datamil/ansur-ii-data-dictionary/workspace/file?filename=ANSUR+II+Databases+Overview.pdf
# 
# ---
# 
# To achieve high prediction success, you must understand the data well and develop different approaches that can affect the dependent variable.
# 
# Firstly, try to understand the dataset column by column using pandas module. Do research within the scope of domain (body scales, and race characteristics) knowledge on the internet to get to know the data set in the fastest way.
# 
# You will implement ***Logistic Regression, Support Vector Machine, XGBoost, Random Forest*** algorithms. Also, evaluate the success of your models with appropriate performance metrics.
# 
# At the end of the project, choose the most successful model and try to enhance the scores with ***SMOTE*** make it ready to deploy. Furthermore, use ***SHAP*** to explain how the best model you choose works.

# # Tasks

# #### 1. Exploratory Data Analysis (EDA)
# - Import Libraries, Load Dataset, Exploring Data
# 
#     *i. Import Libraries*
#     
#     *ii. Ingest Data *
#     
#     *iii. Explore Data*
#     
#     *iv. Outlier Detection*
#     
#     *v.  Drop unnecessary features*
# 
# #### 2. Data Preprocessing
# - Scale (if needed)
# - Separete the data frame for evaluation purposes
# 
# #### 3. Multi-class Classification
# - Import libraries
# - Implement SVM Classifer
# - Implement Decision Tree Classifier
# - Implement Random Forest Classifer
# - Implement XGBoost Classifer
# - Compare The Models
# 
# 

# # EDA
# (Waad and Noof)
# - Drop unnecessary colums
# - Drop DODRace class if value count below 500 (we assume that our data model can't learn if it is below 500)

# ## Import Libraries
# Besides Numpy and Pandas, you need to import the necessary modules for data visualization, data preprocessing, Model building and tuning.
# 
# *Note: Check out the course materials.*

# In[46]:


#import important libraries 
import sklearn
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.tree import plot_tree
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import make_column_transformer
from sklearn.ensemble import RandomForestClassifier
from scikitplot.metrics import plot_roc, precision_recall_curve,plot_precision_recall
import xgboost as xgb
from scipy.stats import uniform, randint
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.model_selection import cross_validate, GridSearchCV ,cross_val_score, cross_validate ,RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score,
    precision_score,
    make_scorer,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    average_precision_score,
    roc_curve,
    auc,
    roc_auc_score,
)


#since we have a huge number of features this line show all rows and columns 
pd.set_option("display.max_rows", 200)
pd.set_option("display.max_columns", 80000)

plt.rcParams["figure.figsize"] = (7, 4)
import warnings

warnings.filterwarnings("ignore")
warnings.warn("this will not show")
pd.set_option("display.float_format", lambda x: "%.3f" % x)


# ## Ingest Data from links below and make a dataframe
# - Soldiers Male : https://query.data.world/s/h3pbhckz5ck4rc7qmt2wlknlnn7esr
# - Soldiers Female : https://query.data.world/s/sq27zz4hawg32yfxksqwijxmpwmynq

# In[2]:


# import dataframes
df_male = pd.read_csv("ANSUR II MALE Public.csv", encoding="ISO-8859-1")
df_female = pd.read_csv("ANSUR II FEMALE Public.csv", encoding="ISO-8859-1")
# merge dataframes
df = pd.concat([df_male, df_female])
df = df.reset_index()
df

# Copy dataframe for manipulating
df_copy = df.copy()


# In[3]:


df_copy


# In[4]:


# Height and weight are repeated
## drop the self_reported
## Convert the unit
df_copy.loc[:, ["weightkg", "Weightlbs", "Heightin", "stature"]]
df_copy["weightkg"] = df_copy["weightkg"] / 10
df_copy


# ## Explore Data

# In[5]:


df_copy.describe(include="O").T


# In[6]:


df_copy["WritingPreference"].value_counts()


# In[7]:


p = pd.crosstab( df_copy.Component,df_copy.DODRace, margins=True, margins_name="Total", normalize='index')
p


# In[8]:


p.plot(kind='bar', stacked=True)
plt.title('DODRace vs Branch')
plt.xlabel('Branch')
plt.ylabel('Race Ratio')
plt.show()


# In[9]:


df_copy["SubjectsBirthLocation"]


# In[10]:


df_copy = df_copy.drop(
    [
        "Ethnicity",
        "PrimaryMOS",
        "SubjectNumericRace",
        "Weightlbs",
        "Heightin",
        "Installation",
        "subjectid",
        "SubjectId",
        "Date",
        "Branch",
        "Component",
        "index",
    ],
    axis=1,
)


# Ethnicity Because many missing values >50%
# "PrimaryMOS", "SubjectsBirthLocation" becuase have many unique values
# "SubjectNumericRace" it may mislead the conclusion [It is similar to the target
# Installation it is about the place that the measurments ocour and we do not need it
# Drop ID number since there is no benefit of it during modeling


# In[11]:


# Look for DODRace with >= 500 observations
df_copy.groupby("DODRace").count()


# In[12]:


# Just DODRace with > 500 observations
df_copy = df_copy.query("DODRace in [1, 2, 3]")
df_copy.groupby("DODRace").count()
df_copy


# In[13]:


df_copy["DODRace"] = df_copy.DODRace.map(
    {
        1: "White",
        2: "Black",
        3: "Hispanic",
    }
)


# In[14]:


df_copy["DODRace"]


# In[15]:


#check missing values
df_copy.isnull().sum().sum()


# In[16]:


# Check duplicates
df.duplicated().sum()


# In[17]:


# Calculate the correlation matrix
correlation_matrix = df_copy.corr(numeric_only=True)

pd.set_option("display.max_rows", None)
correlation_matrix
# Display the correlation matrix with colors
# print(correlation_matrix.style.background_gradient(cmap='coolwarm'))


def color_red(val):
    if (val > 0.90 and val < 1) or (val < -0.90 and val > -1):
        color = "red"
    else:
        color = "black"
    return f"color: {color}"


pd.DataFrame(correlation_matrix).corr().style.applymap(color_red)


# In[18]:


plt.figure(figsize=(20, 16), dpi=200)
sns.heatmap(df_copy.corr(numeric_only=True), vmin=-1, vmax=1);


# In[19]:


df_copy.info(max_cols=50)


# In[20]:


df_copy['SubjectsBirthLocation'].value_counts()


# In[21]:


df_copy.shape


# # DATA Preprocessing
# - In this step we divide our data to X(Features) and y(Target) then ,
# - To train and evaluation purposes we create train and test sets,
# - Lastly, scale our data if features not in same scale. Why?

# In[22]:


# Encode nominal categorical features
categorical  = ["Gender", "WritingPreference", "SubjectsBirthLocation"]


# In[23]:


df_copy["SubjectsBirthLocation"].nunique()


# In[24]:


column_trans = make_column_transformer(
    (OneHotEncoder(handle_unknown="ignore", sparse=False), categorical),
    remainder="passthrough",
    verbose_feature_names_out=False,
)  # MinMaxScaler()

column_trans = column_trans.set_output(transform="pandas")


# In[25]:


df_copy.shape


# In[26]:


# Split the data
X = df_copy.drop(["DODRace"], axis=1)
y = df_copy["DODRace"]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)


# In[27]:


y_train


# # Modelling
# - Fit the model with train dataset
# - Get predict from vanilla model on both train and test sets to examine if there is over/underfitting   
# - Apply GridseachCV for both hyperparemeter tuning and sanity test of our model.
# - Use hyperparameters that you find from gridsearch and make final prediction and evaluate the result according to chosen metric.

# 
# ## 1. Logistic model

# ### Vanilla Logistic Model

# In[28]:


sc = StandardScaler()

lr = LogisticRegression()

operations = [("encoder", column_trans), ("scaler", sc), ("logistic", lr)]

pipe_model = Pipeline(steps=operations)

pipe_model.fit(X_train, y_train)


# In[29]:


def eval_metric(model, X_train, y_train, X_test, y_test):
    y_train_pred = model.predict(X_train)
    y_pred = model.predict(X_test)

    print("Test_Set")
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    print()
    print("Train_Set")
    print(confusion_matrix(y_train, y_train_pred))
    print(classification_report(y_train, y_train_pred))


# In[30]:


eval_metric(pipe_model, X_train, y_train, X_test, y_test)


# ### Cross validation

# In[31]:


operations = [("encoder", column_trans), ("scaler", sc), ("logistic", lr)]

pipe_model = Pipeline(steps=operations)

scores = cross_validate(
    pipe_model,
    X_train,
    y_train,
    scoring=["accuracy", "precision_weighted", "recall_weighted", "f1_weighted"],
    cv=10,
    return_train_score=True,
    verbose = 3,
)
df_scores = pd.DataFrame(scores, index=range(1, 11))
df_scores.mean()


# In[32]:


# 'white':1, 'black':2, 'hispanic': 3
f1_hispanic = make_scorer(f1_score, average=None, labels=['Hispanic'])
precision_hispanic = make_scorer(precision_score, average=None, labels=['Hispanic'])
recall_hispanic = make_scorer(recall_score, average=None, labels=['Hispanic'])


scoring = {
    "f1_hispanic": f1_hispanic,
    "precision_hispanic": precision_hispanic,
    "recall_hispanic": recall_hispanic,
}


# In[33]:


operations = [("encoder", column_trans), ("scaler", sc), ("logistic", lr)]

pipe_model = Pipeline(steps=operations)

scores = cross_validate(
    pipe_model, X_train, y_train, scoring=scoring, cv=10, return_train_score=True ,verbose = 3
)
df_scores = pd.DataFrame(scores, index=range(1, 11))
df_scores.mean()


# # LR with balancing 

# In[34]:


lr_b = LogisticRegression(class_weight='balanced')

operations = [("encoder", column_trans), ("scaler", sc), ("logistic", lr_b)]

pipe_model_b = Pipeline(steps=operations)

pipe_model_b.fit(X_train, y_train)


# In[35]:


eval_metric(pipe_model_b, X_train, y_train, X_test, y_test)


# ### Cross validation

# In[36]:


operations = [("encoder", column_trans), ("scaler", sc), ("logistic", lr)]

pipe_model = Pipeline(steps=operations)

scores = cross_validate(
    pipe_model,
    X_train,
    y_train,
    scoring=["accuracy", "precision_weighted", "recall_weighted", "f1_weighted"],
    cv=10,
    return_train_score=True,
    verbose = 3,
)
df_scores = pd.DataFrame(scores, index=range(1, 11))
df_scores.mean()


# In[37]:


operations = [("encoder", column_trans), ("scaler", sc), ("logistic", lr_b)]

pipe_model = Pipeline(steps=operations)

scores = cross_validate(
    pipe_model, X_train, y_train, scoring=scoring, cv=10, return_train_score=True ,verbose = 3
)
df_scores = pd.DataFrame(scores, index=range(1, 11))
df_scores.mean()


# ### Logistic Model GridsearchCV

# In[38]:


operations = [("encoder", column_trans), ("scaler", sc), ("logistic", lr)]
pipe_model = Pipeline(steps=operations)
param_grid = {
    "logistic__C": [0.01, 0.1, 1, 100],  # 100, 1000
    "logistic__penalty": ["l1", "l2"],
    "logistic__solver": ["liblinear", "saga"],
    "logistic__class_weight": ["balanced", None],
}

grid_search = GridSearchCV(
    pipe_model, param_grid, cv=5, scoring=f1_hispanic, return_train_score=True,verbose =3
)
grid_search.fit(X_train, y_train)


# In[39]:


grid_search.best_estimator_


# In[40]:


pd.DataFrame(grid_search.cv_results_).loc[
    grid_search.best_index_, ["mean_test_score", "mean_train_score"]
]


# In[41]:


eval_metric(grid_search, X_train, y_train, X_test, y_test)


# In[ ]:





# In[42]:


ConfusionMatrixDisplay.from_estimator(grid_search, X_test, y_test);


# In[43]:


operations = [("OneHotEncoder", column_trans), ("logistic", LogisticRegression(C=0.01, class_weight='balanced', solver='saga')
)]

model = Pipeline(steps=operations)

model.fit(X_train, y_train)

y_pred_proba = model.predict_proba(X_test)

plot_roc(y_test, y_pred_proba)
plt.show();


# In[44]:


y_pred = grid_search.predict(X_test)


# In[48]:


operations = [
    ("encoder", column_trans),
    ("scaler", StandardScaler()),
    ("logistic", LogisticRegression(C=0.1, class_weight="balanced", solver="saga")),
]

grid_search = Pipeline(steps=operations)

grid_search.fit(X_train, y_train)

y_pred_proba = grid_search.predict_proba(X_test)

plot_precision_recall(y_test, y_pred_proba)
plt.show();


# In[49]:


y_test_dummies = pd.get_dummies(y_test).values  # we do that for the sake of the average_precision_score function.

average_precision_score(y_test_dummies[:, 1], y_pred_proba[:, 1])


# In[50]:


lr_AP = average_precision_score(y_test_dummies[:, 1], y_pred_proba[:, 1])
lr_f1 = f1_score(y_test, y_pred, average=None, labels=["Hispanic"])
lr_recall = recall_score(y_test, y_pred, average=None, labels=["Hispanic"])


# ## 2. SVC

# ### Vanilla SVC model

# In[51]:


operations = [("encoder", column_trans), ("scaler", StandardScaler()),
              ("SVC", SVC())]

SVM_pipe_model = Pipeline(steps=operations)


# In[52]:


SVM_pipe_model.fit(X_train, y_train)

eval_metric(SVM_pipe_model, X_train, y_train, X_test, y_test)


# ### cross validation

# In[53]:


scores = cross_validate(SVM_pipe_model,
                        X_train,
                        y_train,
                        scoring=scoring,
                        cv = 5,
                        return_train_score=True)

df_scores = pd.DataFrame(scores, index = range(1, 6))
df_scores.mean()[2:]


# ## svm with balancing

# In[54]:


operations = [("encoder", column_trans), ("scaler", StandardScaler()),
              ("SVC", SVC(class_weight='balanced'))]

SVM_pipe_model_b = Pipeline(steps=operations)

SVM_pipe_model_b.fit(X_train, y_train)

eval_metric(SVM_pipe_model_b, X_train, y_train, X_test, y_test)


# In[55]:


scores = cross_validate(SVM_pipe_model_b,
                        X_train,
                        y_train,
                        scoring=scoring,
                        cv = 10,
                        return_train_score=True)

df_scores = pd.DataFrame(scores, index = range(1, 11))
df_scores.mean()[2:]


# In[ ]:





# ###  SVC Model GridsearchCV

# In[56]:


param_grid = {'SVC__C': [ 1.50,2.00,2.25],
              'SVC__gamma': ["scale", "auto", 0.2, 0.3],
              'SVC__kernel': ['rbf', 'linear'],
              'SVC__class_weight': ["balanced"]}
             

operations = [("encoder", column_trans), ("scaler", sc),
              ("SVC", SVC(probability=True))]
SVM_pipe_model_gs = Pipeline(steps=operations)

SVM_pipe_model_grid = GridSearchCV(
    SVM_pipe_model_gs, param_grid, cv=5, scoring=recall_hispanic, return_train_score=True,verbose =3
)
SVM_pipe_model_grid.fit(X_train, y_train)


# In[57]:


SVM_pipe_model_grid.best_estimator_


# In[58]:


SVM_pipe_model_grid.best_params_


# In[59]:


pd.DataFrame(SVM_pipe_model_grid.cv_results_).loc[SVM_pipe_model_grid.best_index_, ["mean_test_score", "mean_train_score"]]


# In[60]:


eval_metric(SVM_pipe_model_grid, X_train, y_train, X_test, y_test)


# In[61]:


y_pred = SVM_pipe_model_grid.predict(X_test)
y_pred


# In[62]:


SVM_pipe_model_grid.predict(X_test)


# In[63]:


ConfusionMatrixDisplay.from_estimator(SVM_pipe_model_grid, X_test, y_test);


# In[64]:


decision_function = SVM_pipe_model_grid.decision_function(X_test)
decision_function


# In[65]:


predict_probe = SVM_pipe_model_grid.predict_proba(X_test)[:,1]
predict_probe


# In[66]:


operations = [
    ("OneHotEncoder", column_trans),
    ("svc", SVC(C=0.01, class_weight='balanced', kernel='linear')),
]

model = Pipeline(steps=operations)

model.fit(X_train, y_train)

# decision_function = model.decision_function(X_test)

plot_precision_recall(y_test, decision_function)
plt.show();


# In[67]:


y_test_dummies = pd.get_dummies(y_test).values


# In[68]:


average_precision_score(y_test_dummies[:, 1], decision_function[:, 1])


# In[69]:


svm_AP = average_precision_score(y_test_dummies[:, 1], y_pred_proba[:, 1])
svm_f1 = f1_score(y_test, y_pred, average=None, labels=["Hispanic"])
snm_recall = recall_score(y_test, y_pred, average=None, labels=["Hispanic"])


# In[ ]:





# ## 3. RF

# ### Vanilla RF Model

# In[70]:


column_trans = make_column_transformer(
                        (OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), categorical),
                            remainder='passthrough',verbose_feature_names_out=False)

column_trans=column_trans.set_output(transform="pandas")


# In[71]:


operations = [("encoder", column_trans),
              ("RF_model", RandomForestClassifier(random_state=101))]  # max_depth=3

v_model = Pipeline(steps=operations)

v_model.fit(X_train, y_train)


# In[72]:


eval_metric(v_model, X_train, y_train, X_test, y_test)


# ### cross validation

# In[73]:


operations_rf = [
    ("OrdinalEncoder", column_trans),
    ("RF_model", RandomForestClassifier( random_state=101)),
]

model = Pipeline(steps=operations_rf)

scores = cross_validate(
    model, X_train, y_train, scoring=scoring, cv=5, n_jobs=-1, return_train_score=True
)
df_scores = pd.DataFrame(scores, index=range(1, 6))
df_scores.mean()[2:]


# ## RF - with balancing

# In[74]:


operations = [("encoder", column_trans),
              ("RF_model", RandomForestClassifier(class_weight="balanced", random_state=101))]

pipe_model = Pipeline(steps=operations)

pipe_model.fit(X_train, y_train)


# In[75]:


eval_metric(pipe_model, X_train, y_train, X_test, y_test)


# In[76]:


operations = [("encoder", column_trans),
              ("RF_model", RandomForestClassifier(max_depth=5, class_weight="balanced", random_state=101))]

pipe_model = Pipeline(steps=operations)

pipe_model.fit(X_train, y_train)


# In[77]:


eval_metric(pipe_model, X_train, y_train, X_test, y_test)


# ### cross validation

# In[78]:


operations = [("encoder", column_trans),
              ("RF_model", RandomForestClassifier(class_weight="balanced", random_state=101))]


model = Pipeline(steps=operations)

scores = cross_validate(model, 
                        X_train, 
                        y_train, 
                        scoring=scoring, 
                        cv = 10,
                        return_train_score=True,
                        verbose =3)
df_scores = pd.DataFrame(scores, index = range(1, 11))
df_scores.mean()[2:]


# In[79]:


y_pred_probe = pipe_model.predict_proba(X_test)
    
plot_roc(y_test, y_pred_probe)
plt.show();


# In[80]:


pipe_model["RF_model"].feature_importances_


# In[81]:


features = pipe_model["encoder"].get_feature_names_out()
features


# In[82]:


rf_feature_imp = pd.DataFrame(data=pipe_model["RF_model"].feature_importances_, 
                              index = features, #index=X.columns
                              columns=["Feature Importance"])

rf_feature_imp = rf_feature_imp.sort_values("Feature Importance", ascending=False)
rf_feature_imp


# In[83]:


plt.figure(figsize=(14, 30), dpi=200)
ax = sns.barplot(x=rf_feature_imp["Feature Importance"], 
                 y=rf_feature_imp.index)

ax.bar_label(ax.containers[0],fmt="%.3f")
plt.title("Feature Importance for Random Forest")
plt.show()


# In[84]:


def report_model(model, number_of_tree):
    model_pred = model.predict(X_test)
    model_train_pred = model.predict(X_train)
    print('')
    print("Test Set")
    print(confusion_matrix(y_test, model_pred))
    print('')
    print(classification_report(y_test,model_pred))
    print('')
    print("Train Set")
    print(confusion_matrix(y_train, model_train_pred))
    print('')
    print(classification_report(y_train,model_train_pred))
    plt.figure(figsize=(12,8),dpi=100)
    plot_tree(model["RF_model"].estimators_[number_of_tree],
              feature_names=features, #features_names=X.columns
              class_names=df.species.unique(),
              filled = True,
              fontsize = 8);


# In[85]:


RF_model = RandomForestClassifier(random_state=101, 
                                  max_samples=0.5)

operations = [("encoder", column_trans), 
              ("RF_model", RF_model)]

pruned_tree = Pipeline(steps=operations)

pruned_tree.fit(X_train,y_train)


# In[86]:


eval_metric(pruned_tree, X_train, y_train, X_test, y_test)


# ### RF Model GridsearchCV

# In[87]:


param_grid = {
    "RF_model__n_estimators": [300,400],
    "RF_model__max_depth": [2, 3],
    'RF_model__min_samples_split':[18,22],
    'RF_model__max_features': ['auto', 20]
}


# In[88]:


RF_model = RandomForestClassifier(class_weight="balanced", random_state=101)

operations = [("encoder", column_trans), 
              ("RF_model", RF_model)]

rg_model = Pipeline(steps=operations)

rf_grid_model = GridSearchCV(estimator=rg_model,
                             param_grid = param_grid,
                             scoring=recall_hispanic,
                             n_jobs = -1,
                             verbose=2)


# In[89]:


rf_grid_model.fit(X_train, y_train)


# In[90]:


rf_grid_model.best_estimator_


# In[91]:


rf_grid_model.best_params_


# In[92]:


rf_grid_model.best_score_


# In[93]:


eval_metric(rf_grid_model, X_train, y_train, X_test, y_test)


# In[94]:


rf_grid_model.best_score_


# In[95]:


operations_rf = [
    ("OrdinalEncoder", column_trans),
    (
        "RF_model",
        RandomForestClassifier(
            class_weight="balanced", max_depth=2, n_estimators=400, random_state=101
        ),
    ),
]

model = Pipeline(steps=operations_rf)

model.fit(X_train, y_train)

y_pred_proba = model.predict_proba(X_test)

plot_precision_recall(y_test, y_pred_proba)
plt.show();


# In[96]:


average_precision_score(y_test_dummies[:, 1], y_pred_proba[:, 1])


# In[97]:


rf_AP = average_precision_score(y_test_dummies[:, 1], y_pred_proba[:, 1])
rf_f1 = f1_score(y_test, y_pred, average=None, labels=["Hispanic"])
rf_recall = recall_score(y_test, y_pred, average=None, labels=["Hispanic"])


# ## 4. XGBoost

# ### Vanilla XGBoost Model

# In[98]:


# Encode nominal categorical features
# We will do ordinal even though they are nominal since ordinal is better in Tree-based algorithms


# In[99]:


column_trans = make_column_transformer(
    (OrdinalEncoder(handle_unknown= 'use_encoded_value', unknown_value=-1), categorical),
    remainder="passthrough",
    verbose_feature_names_out=False,
)  

column_trans = column_trans.set_output(transform="pandas")


# In[100]:


# Expected: [0 1 2], got ['Black' 'Hispanic' 'White']
df_copy2 = df_copy.copy()
#df_copy["DODRace"] = df_copy.DODRace.map({0: "Black", 1: "Hispanic", 2: "White"})
df_copy2["DODRace"] = df_copy2.DODRace.map({"Black": 0, "Hispanic": 1, "White": 2})

df_copy2["DODRace"]


# In[101]:


df_copy2["DODRace"]


# In[102]:


# Split the data
X = df_copy2.drop(["DODRace"], axis=1)
y = df_copy2["DODRace"]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)


# In[103]:


xgb_classifier = XGBClassifier()

# Create a pipeline
pipe_model = Pipeline([("preprocessor", column_trans), ("xgboost", xgb_classifier)])

pipe_model.fit(X_train, y_train)


# In[104]:


eval_metric(pipe_model, X_train, y_train, X_test, y_test)


# ### CV XGBoost Model

# In[105]:


# 'white':2, 'black':0, 'hispanic': 1
f1_hispanic = make_scorer(f1_score, average=None, labels=[1])
precision_hispanic = make_scorer(precision_score, average=None, labels=[1])
recall_hispanic = make_scorer(recall_score, average=None, labels=[1])


scoring = {
    "f1_hispanic": f1_hispanic,
    "precision_hispanic": precision_hispanic,
    "recall_hispanic": recall_hispanic,
}


# In[106]:


xgb_classifier = XGBClassifier()

# Create a pipeline
pipe_model = Pipeline([("preprocessor", column_trans), ("xgboost", xgb_classifier)])

pipe_model.fit(X_train, y_train)

scores = cross_validate(
    pipe_model, X_train, y_train, scoring=scoring, cv=5, return_train_score=True
)
df_scores = pd.DataFrame(scores, index=range(1, 6))
df_scores.mean()


# In[107]:


eval_metric(pipe_model, X_train, y_train, X_test, y_test)


# ### Random Serach

# In[108]:


# Define the hyperparameter space


param_grid = {"xgboost__n_estimators":[50,80],
              "xgboost__max_depth":[3,4],
              "xgboost__learning_rate": [0.8, 0.5],
              "xgboost__subsample":[0.5, 1],
}
xgb_classifier = XGBClassifier()
# Create a pipeline
pipe_model = Pipeline([
    ('preprocessor', column_trans),
    ('xgboost', xgb_classifier)
])


# Create the random search object
random_search = RandomizedSearchCV(
    estimator=pipe_model,
    param_distributions=param_grid,
    n_iter=1000,
    cv=10,
    verbose=2,
    return_train_score=True,
    scoring=f1_hispanic,
    n_jobs=-1,
)

# Fit the random search model
random_search.fit(X_train, y_train)



# In[109]:


# Print the best hyperparameters
print(random_search.best_params_)


# In[110]:


pd.DataFrame(random_search.cv_results_).loc[
    random_search.best_index_, ["mean_test_score", "mean_train_score"]
]


# In[111]:


eval_metric(random_search, X_train, y_train, X_test, y_test)


# In[112]:


random_search.best_score_


# In[113]:


operations_xgb = [
    ("OrdinalEncoder", column_trans),
    (
        "XGB_model",
        XGBClassifier(
            colsample_bytree=0.8,
            learning_rate=0.05,
            max_depth=2,
            n_estimators=20,
            subsample=0.8,
            random_state=101,
        ),
    ),
]

model = Pipeline(steps=operations_xgb)

model.fit(X_train, y_train)

y_pred_proba = model.predict_proba(X_test)

plot_precision_recall(y_test, y_pred_proba)
plt.show()


# In[114]:


y_test_xgb_dummies = pd.get_dummies(y_test).values


# In[115]:


y_pred = random_search.predict(X_test)

xgb_AP = average_precision_score(y_test_xgb_dummies[:, 1], y_pred_proba[:, 1])
xgb_f1 = f1_score(y_test, y_pred, average=None, labels=[1])
xgb_recall = recall_score(y_test, y_pred, average=None, labels=[1])


# # comparing models 

# In[116]:


compare = pd.DataFrame(
    {
        "Model": ["Logistic Regression", "SVM", "Random Forest", "XGBoost"],
        "F1": [lr_f1[0], svm_f1[0], rf_f1[0], xgb_f1[0]],
        "Recall": [lr_recall[0], snm_recall[0], rf_recall[0], xgb_recall[0]],
        "AP": [lr_AP, svm_AP, rf_AP, xgb_AP],
    }
)


plt.figure(figsize=(14, 10))
plt.subplot(311)
compare = compare.sort_values(by="F1", ascending=False)
ax = sns.barplot(x="F1", y="Model", data=compare, palette="Blues_d")
ax.bar_label(ax.containers[0], fmt="%.3f")

plt.subplot(312)
compare = compare.sort_values(by="Recall", ascending=False)
ax = sns.barplot(x="Recall", y="Model", data=compare, palette="Blues_d")
ax.bar_label(ax.containers[0], fmt="%.3f")

plt.subplot(313)
compare = compare.sort_values(by="AP", ascending=False)
ax = sns.barplot(x="AP", y="Model", data=compare, palette="Blues_d")
ax.bar_label(ax.containers[0], fmt="%.3f")
plt.show();


# ---
# ---

# ---
# ---

# # SMOTE
# https://machinelearningmastery.com/smote-oversampling-for-imbalanced-classification/

# ##  Smote implement

# In[117]:


from collections import Counter
from imblearn.under_sampling import ClusterCentroids
from imblearn.over_sampling import SMOTE


# In[118]:


column_trans = make_column_transformer(
    (OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
    remainder=StandardScaler(),
)


# ## Logistic Regression "Under Sampling"

# In[119]:


X_train_encoded = column_trans.fit_transform(X_train)
X_test_encoded = column_trans.transform(X_test) # We shouldn't apply fit_transform to the TEST data.

cc = ClusterCentroids(random_state=0)
X_resampled, y_resampled = cc.fit_resample(X_train_encoded, y_train)
print(sorted(Counter(y_resampled).items()))


# In[127]:


svm_resampled = SVC(C=1.5, class_weight='balanced',
                                    kernel='linear')

svm_resampled.fit(X_resampled, y_resampled)


# In[128]:


def eval_metric_(model, X_train, y_train, X_test, y_test):
    y_train_pred = model.predict(X_resampled)
    y_pred = model.predict(X_test_encoded)

    print("Test_Set")
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    print()
    print("Train_Set")
    print(confusion_matrix(y_resampled, y_train_pred))
    print(classification_report(y_resampled, y_train_pred))


# In[129]:


eval_metric_(svm_resampled, X_resampled, y_resampled, X_test, y_test)


# ## Logistic Regression "Over Sampling"

# ## **Remember that while SMOTE can be helpful in many situations, it's not a one-size-fits-all solution, and its effectiveness can vary depending on the specific problem and dataset. It's important to carefully evaluate the performance of your model after using SMOTE and consider other techniques such as adjusting class weights or using different sampling strategies if necessary.**

# In[130]:


#Using Somte to over sampling the data

X_train_encoded = column_trans.fit_transform(X_train)
X_test_encoded = column_trans.transform(X_test) # We shouldn't apply fit_transform to the TEST data.

# Apply SMOTE to the training data only
smote = SMOTE(sampling_strategy='auto', random_state=101)
X_resampled_Over, y_resampled_Over = smote.fit_resample(X_train_encoded, y_train)

print(sorted(Counter(y_resampled).items()))


# In[131]:


# Implement the LogisticRegression model with the balanced data

svm_resampled = SVC(C=0.5, class_weight='balanced',
                                    kernel='linear')

svm_resampled.fit(X_resampled, y_resampled)


# In[133]:


eval_metric_(svm_resampled, X_resampled_Over, y_resampled_Over, X_test, y_test)


# # Before the Deployment
# - Choose the model that works best based on your chosen metric
# - For final step, fit the best model with whole dataset to get better performance.
# - And your model ready to deploy, dump your model and scaler.

# In[134]:


column_trans_final = make_column_transformer(
    (OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
    remainder=StandardScaler(),
)

operations_final = [
    ("OneHotEncoder", column_trans_final),
    (
        "log",
        LogisticRegression(class_weight="balanced", max_iter=10000, random_state=101),
    ),
]

final_model = Pipeline(steps=operations_final)


# In[135]:


final_model.fit(X, y)


# In[136]:


X[X.Gender == "Male"].describe()


# In[137]:


male_mean_human = X[X.Gender == "Male"].describe(include="all").loc["mean"]
male_mean_human


# In[138]:


male_mean_human["Gender"] = "Male"
male_mean_human["SubjectsBirthLocation"] = "Texas"
male_mean_human["WritingPreference"] = "Left hand"


# In[139]:


pd.DataFrame(male_mean_human).T


# In[140]:


final_model.predict(pd.DataFrame(male_mean_human).T)


# In[141]:


from sklearn.metrics import matthews_corrcoef

y_pred = final_model.predict(X_test)

matthews_corrcoef(y_test, y_pred)


# In[142]:


from sklearn.metrics import cohen_kappa_score

cohen_kappa_score(y_test, y_pred)


# ___
# 
# <p style="text-align: center;"><img src="https://docs.google.com/uc?id=1lY0Uj5R04yMY3-ZppPWxqCr5pvBLYPnV" class="img-fluid" alt="CLRSWY"></p>
# 
# ___
