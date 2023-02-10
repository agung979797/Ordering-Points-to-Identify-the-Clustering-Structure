# Import Libraries
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
from matplotlib import gridspec 
from sklearn.cluster import OPTICS, cluster_optics_dbscan 
from sklearn.preprocessing import normalize, StandardScaler 


#! EXPLORATORY DATA ANALYSIS
data = pd.read_csv('data_input/Mall_Customers.csv') 
data.head()

#?? DISTRIBUSI AGE VS. SPENDING SCORE
plt.figure(figsize=(10,8))
sns.scatterplot(data=data, x="Age", y="Spending Score (1-100)", hue="Gender")

# DISTRIBUSI ANNUAL INCOME VS. SPENDING SCORE
plt.figure(figsize=(10,8))
sns.scatterplot(data=data, x="Spending Score (1-100)", y="Annual Income (k$)", hue="Gender")

# NORMALISASI DATA
#?? Untuk tujuan analisis selanjutnya, kolom CustomerID dan Gender tidak akan diikutsertakan mengingat perhitungan clustering hanya membutuhkan kolom-kolom numerik. Selain itu, kita juga melakukan normalisasi terhadap data dikarenakan rentang nilai pada setiap kolom berbeda-beda.

drop_features = ['CustomerID', 'Gender'] 
data = data.drop(drop_features, axis = 1) 
data.fillna(method ='ffill', inplace = True) 

# Melakukan proses scaling untuk standarisasi nilai 
scaler = StandardScaler() 
X_scaled = scaler.fit_transform(data) 

# Melakukan normalisasi untuk membuat normal distribusi
X_normalized = normalize(X_scaled) 

# Converting numpy ke dalam data frame
X_normalized = pd.DataFrame(X_normalized) 

# Melakukan rename kolom
X_normalized.columns = data.columns 

X_normalized.head() 

#! MEMBUAT MODEL CLUSTERING
optics_model = OPTICS(min_samples = 10, xi = 0.05) 

optics_model.fit(X_normalized)

# Membuat label berdasarkan model DBSCAN dengan eps = 0.5 
labels1 = cluster_optics_dbscan(reachability = optics_model.reachability_, 
    core_distances = optics_model.core_distances_, 
    ordering = optics_model.ordering_, eps = 0.5) 

# Membuat label berdasarkan model DBSCAN dengan eps = 0.5  eps = 0.25 
labels2 = cluster_optics_dbscan(reachability = optics_model.reachability_, 
    core_distances = optics_model.core_distances_, 
    ordering = optics_model.ordering_, eps = 0.25) 

space = np.arange(len(X_normalized)) 

# Menyimpan reachability distance dari setiap titik /point
reachability = optics_model.reachability_[optics_model.ordering_] 

# Menyimpan label cluster dari setiap titik/point 
labels = optics_model.labels_[optics_model.ordering_] 

print(labels)
print(labels1)
print(labels2)

# Membuat framework untuk tujuan visualisasi 
plt.figure(figsize =(15, 10)) 
G = gridspec.GridSpec(2, 3) 
ax1 = plt.subplot(G[0, :]) 
ax2 = plt.subplot(G[1, 0]) 
ax3 = plt.subplot(G[1, 1]) 
ax4 = plt.subplot(G[1, 2]) 

# Visualisasi Plot Reachability-Distance  
colors = ['c.', 'b.', 'r.', 'y.', 'g.'] 
for Class, colour in zip(range(0, 5), colors): 
    Xk = space[labels == Class] 
    Rk = reachability[labels == Class] 
    ax1.plot(Xk, Rk, colour, alpha = 0.3) 
ax1.plot(space[labels == -1], reachability[labels == -1], 'k.', alpha = 0.3) 
ax1.plot(space, np.full_like(space, 2., dtype = float), 'k-', alpha = 0.5) 
ax1.plot(space, np.full_like(space, 0.5, dtype = float), 'k-.', alpha = 0.5) 
ax1.set_ylabel('Reachability Distance') 
ax1.set_title('Reachability Plot') 

#! Visualisasi OPTICS Clustering 
colors = ['c.', 'b.', 'r.', 'y.', 'g.'] 
for Class, colour in zip(range(0, 5), colors): 
    Xk = X_normalized[optics_model.labels_ == Class] 
    ax2.plot(Xk.iloc[:, 1], Xk.iloc[:, 2], colour, alpha = 0.3) 

ax2.plot(X_normalized.iloc[optics_model.labels_ == -1, 0], 
        X_normalized.iloc[optics_model.labels_ == -1, 1], 
    'k+', alpha = 0.1) 
ax2.set_title('OPTICS Clustering') 

# Visualisasi DBSCAN Clustering dengan eps = 0.5 
colors = ['c.', 'b.', 'r.', 'y.', 'g.'] 
for Class, colour in zip(range(0, 6), colors): 
    Xk = X_normalized[labels1 == Class] 
    ax3.plot(Xk.iloc[:, 1], Xk.iloc[:, 2], colour, alpha = 0.3, marker ='.') 

ax3.plot(X_normalized.iloc[labels1 == -1, 0], 
        X_normalized.iloc[labels1 == -1, 1], 
    'k+', alpha = 0.1) 
ax3.set_title('DBSCAN clustering with eps = 0.5') 

# Visualisasi DBSCAN Clustering dengan eps = 0.25 
colors = ['c.', 'y.', 'm.', 'g.'] 
for Class, colour in zip(range(0, 4), colors): 
    Xk = X_normalized.iloc[labels2 == Class] 
    ax4.plot(Xk.iloc[:, 1], Xk.iloc[:, 2], colour, alpha = 0.3) 

ax4.plot(X_normalized.iloc[labels2 == -1, 0], 
        X_normalized.iloc[labels2 == -1, 1], 
    'k+', alpha = 0.1) 
ax4.set_title('DBSCAN Clustering with eps = 0.25') 

plt.tight_layout() 
plt.show()