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

from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(train_df['TITLE'])
X_valid = vectorizer.transform(valid_df['TITLE'])
X_test = vectorizer.transform(test_df['TITLE'])
np.savetxt('train.feature.txt', X_train.toarray(), fmt='%d') # スパース行列から密行列に変換
np.savetxt('valid.feature.txt', X_valid.toarray(), fmt='%d')
np.savetxt('test.feature.txt', X_test.toarray(), fmt='%d')

from sklearn.linear_model import LogisticRegression
clf = LogisticRegression()
clf.fit(X_train, train_df['CATEGORY'])

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
