#50
import pandas as pd
from sklearn.model_selection import train_test_split
from functools import reduce
# 2. 事例の抽出
news_corpora = pd.read_csv('NewsAggregatorDataset/newsCorpora.csv',sep='\t',header=None)
news_corpora.columns = ['ID','TITLE','URL','PUBLISHER','CATEGORY','STORY','HOSTNAME','TIMESTAMP']
publisher = ['Reuters', 'Huffington Post', 'Businessweek', 'Contactmusic.com', 'Daily Mail']
ls_is_specified = [news_corpora.PUBLISHER == p for p in publisher]
is_specified =reduce(lambda a, b: a | b, ls_is_specified)
df = news_corpora[is_specified]
#  3. 並び替え
df = df.sample(frac=1) # 全てをサンプリングするので、並び替えと等価
# 4.保存
train_df, valid_test_df = train_test_split(df, test_size=0.2) # 8:2
valid_df, test_df = train_test_split(valid_test_df, test_size=0.5) # 8:1:1
train_df.to_csv('train.txt', columns = ['CATEGORY','TITLE'], sep='\t',header=False, index=False)
valid_df.to_csv('valid.txt', columns = ['CATEGORY','TITLE'], sep='\t',header=False, index=False)
test_df.to_csv('test.txt', columns = ['CATEGORY','TITLE'], sep='\t',header=False, index=False)
#  事例数の確認
df['CATEGORY'].value_counts()
#51
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(train_df['TITLE'])
X_valid = vectorizer.transform(valid_df['TITLE'])
X_test = vectorizer.transform(test_df['TITLE'])
np.savetxt('train.feature.txt', X_train.toarray(), fmt='%d') # スパース行列から密行列に変換
np.savetxt('valid.feature.txt', X_valid.toarray(), fmt='%d')
np.savetxt('test.feature.txt', X_test.toarray(), fmt='%d')
#52
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression()
clf.fit(X_train, train_df['CATEGORY'])
#53
dic = {'b':'business', 't':'science and technology', 'e' : 'entertainment', 'm' : 'health'}
def predict(text):
    text = [text]
    X = vectorizer.transform(text)
    ls_proba = clf.predict_proba(X)
    for proba in ls_proba:
        for c, p in zip(clf.classes_, proba):
            print (dic[c]+':',p)
s = train_df.iloc[0]['TITLE']
print(s)
predict(s)
#54
from sklearn.metrics import accuracy_score
y_train_pred = clf.predict(X_train)
y_test_pred = clf.predict(X_test)
y_train = train_df['CATEGORY']
y_test = test_df['CATEGORY']
print (accuracy_score(y_train, y_train_pred))
print (accuracy_score(y_test, y_test_pred))
#55
from sklearn.metrics import confusion_matrix
print (confusion_matrix(y_train, y_train_pred, labels=['b','t','e','m']))
print (confusion_matrix(y_test, y_test_pred, labels=['b','t','e','m']))
#56
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
print (precision_score(y_test, y_test_pred, average=None, labels=['b','t','e','m']))
print (recall_score(y_test, y_test_pred, average=None, labels=['b','t','e','m']))
print (f1_score(y_test, y_test_pred, average=None, labels=['b','t','e','m']))
print (precision_score(y_test, y_test_pred, average='micro', labels=['b','t','e','m']))
print (recall_score(y_test, y_test_pred, average='micro', labels=['b','t','e','m']))
print (f1_score(y_test, y_test_pred, average='micro', labels=['b','t','e','m']))
print (precision_score(y_test, y_test_pred, average='macro', labels=['b','t','e','m']))
print (recall_score(y_test, y_test_pred, average='macro', labels=['b','t','e','m']))
print (f1_score(y_test, y_test_pred, average='macro', labels=['b','t','e','m']))
#57
names = np.array(vectorizer.get_feature_names())
labels=['b','t','e','m']
for c, coef in zip(clf.classes_, clf.coef_): # カテゴリ毎に表示する
    idx = np.argsort(coef)[::-1]
    print (dic[c])
    print (names[idx][:10]) # 重みの高い特徴量トップ10
    print (names[idx][-10:][::-1]) # 重みの低い特徴量トップ10
#58
import matplotlib.pyplot as plt
def calc_scores(c):
    y_train = train_df['CATEGORY']
    y_valid = valid_df['CATEGORY']
    y_test = test_df['CATEGORY']
    
    clf = LogisticRegression(C=c)
    clf.fit(X_train, y_train)

    y_train_pred = clf.predict(X_train)
    y_valid_pred = clf.predict(X_valid)
    y_test_pred = clf.predict(X_test)
    
    scores = []
    scores.append(accuracy_score(y_train, y_train_pred))
    scores.append(accuracy_score(y_valid, y_valid_pred))
    scores.append(accuracy_score(y_test, y_test_pred))
    return scores

C = np.logspace(-5, 4, 10, base=10)
scores = []
for c in C:
    scores.append(calc_scores(c))
scores = np.array(scores).T
labels = ['train', 'valid', 'test']

for score, label in zip(scores,labels):
    plt.plot(C, score, label=label)
plt.ylim(0, 1.1)
plt.xscale('log')
plt.xlabel('C', fontsize = 14)
plt.ylabel('Accuracy', fontsize = 14)
plt.tick_params(labelsize=14)
plt.grid(True)
plt.legend()
#59
import itertools
def calc_scores(C,solver,class_weight):
    y_train = train_df['CATEGORY']
    y_valid = valid_df['CATEGORY']
    y_test = test_df['CATEGORY']
    
    clf = LogisticRegression(C=C, solver=solver, class_weight=class_weight)
    clf.fit(X_train, y_train)

    y_train_pred = clf.predict(X_train)
    y_valid_pred = clf.predict(X_valid)
    y_test_pred = clf.predict(X_test)
    
    scores = []
    scores.append(accuracy_score(y_train, y_train_pred))
    scores.append(accuracy_score(y_valid, y_valid_pred))
    scores.append(accuracy_score(y_test, y_test_pred))
    return scores

C = np.logspace(-5, 4, 10, base=10)
solver = ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
class_weight = [None, 'balanced']
best_parameter = None
best_scores = None
max_valid_score = 0
for c, s, w in itertools.product(C, solver, class_weight):
    print(c, s, w)
    scores = calc_scores(c, s, w)
    #print (scores)
    if scores[1] > max_valid_score:
        max_valid_score = scores[1]
        best_parameter = [c, s, w]
        best_scores = scores
print ('best patameter: ', best_parameter)
print ('best scores: ', best_scores)
print ('test accuracy: ', best_scores[2])
