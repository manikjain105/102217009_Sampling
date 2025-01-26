import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


Z = 1.96 
p = 0.5   
E = 0.05 
S = 5     
C = 10    

df = pd.read_csv("Creditcard_data.csv")
X = df.drop(['Class'], axis=1)  
y = df['Class']


smote = SMOTE()
X_balanced, y_balanced = smote.fit_resample(X, y)


n_simple = int((Z**2 * p * (1-p)) / E**2)
n_stratified = int((Z**2 * p * (1-p)) / (E/S)**2)
n_cluster = int((Z**2 * p * (1-p)) / (E/C)**2)
n_systematic = len(X_balanced) // 20  

models = {
    'RandomForest': RandomForestClassifier(),
    'GradientBoosting': GradientBoostingClassifier(),
    'LogisticRegression': LogisticRegression(),
    'SVM': SVC(),
    'KNeighbors': KNeighborsClassifier()
}

sample_simple = X_balanced.sample(n=n_simple, random_state=42)
strata = pd.qcut(y_balanced, q=S, labels=False, duplicates='drop')
sample_stratified = X_balanced.groupby(strata, group_keys=False).apply(lambda x: x.sample(min(len(x), n_stratified // S), random_state=42))
kmeans = KMeans(n_clusters=C, random_state=42)
clusters = kmeans.fit_predict(X_balanced)
X_balanced['cluster'] = clusters
sample_cluster = X_balanced.groupby('cluster', group_keys=False).apply(lambda x: x.sample(min(len(x), n_cluster // C), random_state=42), include_groups=False)
interval = len(X_balanced) // n_systematic
indices = np.arange(0, len(X_balanced), step=interval)
sample_systematic = X_balanced.iloc[indices]
sample_multistage = sample_stratified.groupby(strata, group_keys=False).apply(lambda x: x.sample(min(len(x), n_stratified // S // 2), random_state=42))

results = []
sample_names = ['Simple Random', 'Stratified', 'Cluster', 'Systematic', 'Multistage']
samples = [sample_simple, sample_stratified, sample_cluster, sample_systematic, sample_multistage]

for sample_name, sample in zip(sample_names, samples):
    X_sample = sample.drop(['cluster'], axis=1, errors='ignore')
    y_sample = y_balanced.loc[X_sample.index]
    X_train, X_test, y_train, y_test = train_test_split(X_sample, y_sample, test_size=0.2, random_state=42)
    for model_name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        results.append({'Model': model_name, 'Sampling Technique': sample_name, 'Accuracy': accuracy})

df_results = pd.DataFrame(results)
pivot_table = df_results.pivot(index='Model', columns='Sampling Technique', values='Accuracy')

pivot_table.to_csv('results.csv')
print(pivot_table)
