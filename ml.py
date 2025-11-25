# Cell 1.1 - Dependent & independent variables
import pandas as pd

df = pd.read_csv('/content/sample_data/MallCustomers.csv')
X = df.drop(columns=['Spending Score (1-100)'])
y = df['Spending Score (1-100)']

print("X.shape:", X.shape, "y.shape:", y.shape)
X.head()



# Cell 1.2 - Missing values detection and simple imputation
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

df = pd.read_csv('/content/sample_data/real_estate.csv')

df.replace(['na', 'NA', 'N/A', 'HURLEY', '?', 'none', 'None'], np.nan, inplace=True)

num_cols = df.select_dtypes(include='number').columns
if len(num_cols):
    df[num_cols] = SimpleImputer(strategy='mean').fit_transform(df[num_cols])

cat_cols = df.select_dtypes(include='object').columns
for c in cat_cols:
    if df[c].isnull().any():
        df[c].fillna(df[c].mode().iloc[0], inplace=True)

print(df.isnull().sum())




# Cell 1.3 - Label encoding and OneHot encoding
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

df = pd.read_csv('/content/sample_data/MallCustomers.csv')

le = LabelEncoder()
df['Gender_le'] = le.fit_transform(df['Gender'])

ct = ColumnTransformer([('oh', OneHotEncoder(drop='first', sparse_output=False), ['Gender'])], remainder='drop')
encoded = ct.fit_transform(df[['Gender']])

print("Label Encoded:", df['Gender_le'].head().tolist())
print("OneHot shape:", encoded.shape)




# Cell 1.4 - Feature scaling examples
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

df = pd.read_csv('/content/sample_data/MallCustomers.csv')
X_num = df.select_dtypes(include='number')

std = StandardScaler().fit_transform(X_num)
mm = MinMaxScaler().fit_transform(X_num)

print("Std scaled first row:", std[0])
print("MinMax scaled first row:", mm[0])




# Cell 1.5 - Outlier detection & removal using IQR
import pandas as pd
df = pd.read_csv('/content/sample_data/MallCustomers.csv')
X_num = df.select_dtypes(include='number')

def remove_outliers_iqr(df_num):
    clean = df_num.copy()
    for col in df_num.columns:
        Q1 = clean[col].quantile(0.25)
        Q3 = clean[col].quantile(0.75)
        IQR = Q3 - Q1
        low = Q1 - 1.5 * IQR
        high = Q3 + 1.5 * IQR
        clean = clean[(clean[col] >= low) & (clean[col] <= high)]
    return clean

clean_df = remove_outliers_iqr(X_num)

print("Before:", X_num.shape[0], "After:", clean_df.shape[0])
clean_df.head()




# Linear Regression using sklearn diabetes dataset
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset and select one feature (for simple linear regression)
X, y = load_diabetes(return_X_y=True)
X = X[:, [2]]   # using feature index 2 for demonstration

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1
)

# Fit model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Results
print("Coefficient:", model.coef_[0])
print("Intercept:", model.intercept_)
print("MSE:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))




# SVM classifier on Iris (Setosa vs Versicolor) - simple, minimal
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# load data and use first two features for clarity
iris = datasets.load_iris()
X = iris.data[:, :2]
y = iris.target

# keep only classes 0 and 1 (binary)
mask = y != 2
X, y = X[mask], y[mask]

# train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# train linear SVM
clf = SVC(kernel='linear', C=1.0)
clf.fit(X_train, y_train)

# predict and report
y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=iris.target_names[:2]))




# K-Means clustering on MallCustomers dataset (Annual Income vs Spending Score)
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# load dataset
df = pd.read_csv('/content/sample_data/MallCustomers.csv')

# select 2 features for clear visualization
X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

# scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# apply k-means
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
labels = kmeans.fit_predict(X_scaled)

# plot clusters
plt.scatter(X_scaled[:,0], X_scaled[:,1], c=labels)
plt.xlabel("Annual Income (scaled)")
plt.ylabel("Spending Score (scaled)")
plt.title("K-Means Clustering")
plt.show()




# KNN classifier on Iris - simple and minimal
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

iris = load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

print("KNN accuracy:", accuracy_score(y_test, y_pred))
# example single prediction (shows class name)
print("Example prediction (first test sample):", iris.target_names[knn.predict([X_test[0]])[0]])




# Decision Tree classifier on the Iris dataset (simple & minimal)
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# load data
iris = load_iris()
X, y = iris.data, iris.target

# train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# train decision tree
dt = DecisionTreeClassifier(max_depth=4, random_state=42)
dt.fit(X_train, y_train)

# predict and report
y_pred = dt.predict(X_test)
print("Decision Tree accuracy:", accuracy_score(y_test, y_pred))

# optional: feature importances (brief)
print("Feature importances:", dt.feature_importances_)



# Cell 7.2 - BaggingClassifier (simple & minimal)
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

iris = load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

bag = BaggingClassifier(estimator=DecisionTreeClassifier(), n_estimators=30, random_state=42)
bag.fit(X_train, y_train)

y_pred = bag.predict(X_test)
print("Bagging accuracy:", accuracy_score(y_test, y_pred))



# Cell 7.2 - BaggingClassifier (simple & minimal)
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

iris = load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

bag = BaggingClassifier(estimator=DecisionTreeClassifier(), n_estimators=30, random_state=42)
bag.fit(X_train, y_train)

y_pred = bag.predict(X_test)
print("Bagging accuracy:", accuracy_score(y_test, y_pred))



# Cell 7.3 - VotingClassifier (hard voting) using KNN, DecisionTree, and SVC
from sklearn.ensemble import VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

iris = load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

clf1 = KNeighborsClassifier(n_neighbors=5)
clf2 = DecisionTreeClassifier(max_depth=4, random_state=42)
clf3 = SVC(kernel='rbf', probability=True, random_state=42)

voting = VotingClassifier(estimators=[('knn', clf1), ('dt', clf2), ('svc', clf3)], voting='hard')
voting.fit(X_train, y_train)

y_pred = voting.predict(X_test)
print("Voting ensemble accuracy:", accuracy_score(y_test, y_pred))

