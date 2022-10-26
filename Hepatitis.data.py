# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn.ensemble import StackingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix
import seaborn as sns

sns.set_theme()



df = pd.read_csv(r"C:/Users/User/OneDrive/Desktop/datasetler/hepatitis.csv")


'Data haqqında ilkin informasiya'
df.info()

'Data haqqında statistik informasiya'
describe = df.describe()

'Data sütun adları'
s=df.columns


'Class sütunu üzrə dəyərlər'
df['class'].value_counts()

'Bu sütunda 1-lər 0-la, 2-ləri isə 1-lə əvəz edirik'
df['class'].replace({1:0, 2:1}, inplace = True)

'Datatını x və y-ə ayırırıq'
x = df.drop('class', axis = 1)
y = df[['class']]

'Feature-lər arasındakı əlaqə'
x_corr = x.corr()

'Datanı train və test setlərə ayırırıq'
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 1)


'Targetdə inbalancı(1 və 0-ların sayında) aradan qaldırırıq,Oversamling edirik'
y_train.value_counts()

sm = SMOTE(sampling_strategy = 0.75, random_state = 1)
x_tr, y_tr = sm.fit_resample(x_train, y_train)

y_tr.value_counts()

'Datanı StandartScaler edirik'
sc = StandardScaler()
x_train = pd.DataFrame(sc.fit_transform(x_tr), columns = x.columns)
x_test = pd.DataFrame(sc.transform(x_test), columns = x.columns)


'Modelin öyrənməsinə təsir edən 5 əsas feature'
lg = LogisticRegression()
rfe = RFE(lg, n_features_to_select = 5)
selector = rfe.fit(x_tr, y_tr)

print(selector.support_)
print(selector.ranking_)
x_tr.columns

'StackingClassifier Modeli qururuq'
estimators1 = [('svm', SVC()),
 ('nb', GaussianNB()),
 ('lr', LogisticRegression(random_state=1)), 
 ('dt', DecisionTreeClassifier())]


clf = StackingClassifier(estimators = estimators1, final_estimator = LogisticRegression(random_state = 1))

clf.fit(x_tr, y_tr)
y_pred = clf.predict(x_test)

result_ = pd.DataFrame()
result_['y_test'] = y_test['class']
result_['y_pred'] = y_pred

'Performance Metrics-ləri hesablayırıq'
accuracy = accuracy_score(y_test, y_pred)

precision =  precision_score(y_test, y_pred)

f1_score = f1_score(y_test, y_pred)

confusion_matrix = confusion_matrix(y_test, y_pred)

roc_auc_score = roc_auc_score(y_test, y_pred)

'Müqayisə edirik'
accuracy = [0.82]
precision = [0.82]
f1_score = [0.90]
roc_auc_score = [0.5]

x = list(zip(accuracy,precision,f1_score,roc_auc_score))
result = pd.DataFrame(x, columns= ('accuracy', 'precision', 'f1_score', 'roc_auc_score'))

_____________________________________________________

'Data üzərində analizlər'

'cins və age sütunları üzrə xəstə və sağlam olanlar'
crs1 = pd.crosstab(index = [df['sex'], df['age']],
                   columns= df['class'])

"xəstə və sağlam olanların ortalama yaşı"
pt1 = pd.pivot_table(df,index = ['class'],
                     values='age',aggfunc='mean')


'Displot qururuq'
sns.set_theme()
sns.displot(data = df, x= 'sgot', col = 'class', hue = 'sex', kde = True)

