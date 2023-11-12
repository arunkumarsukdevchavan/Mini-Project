# Mini-Project
# ANALYSIS OF THE DETAILS OF A PERSON
# Aim:
 Analysis of the details of a person.
 
# ALGORITHM:
Step:1  Importing necessary packages.

Step:2  Read the data set.

Step:3  Execute the methods.

Step:4  Run the program.

Step:5  Get the output.

# CODE AND OUTPUT:
Name : ARUN KUMAR SUKDEV CHAVAN

Reg.no : 212222230013
```python
import pandas as pd
df = pd.read_csv("addresses.csv")

df.head(4)
```
![image](https://github.com/arunkumarsukdevchavan/Mini-Project/assets/118343978/abafe400-162e-41c1-a9aa-6c8bf73c7ea0)

```python
df.info()
```
![image](https://github.com/arunkumarsukdevchavan/Mini-Project/assets/118343978/f60842f0-b12a-40f7-9159-2d65c83fd91f)


```python
df.dropna(how='all').shape
```
![image](https://github.com/arunkumarsukdevchavan/Mini-Project/assets/118343978/cdcc51b5-14ad-40f5-bc4a-69f6bb96663b)

```python
df.fillna(0)
```
![image](https://github.com/arunkumarsukdevchavan/Mini-Project/assets/118343978/d96f95af-1842-4773-8564-8d7cf3df3c73)


```python
df.fillna(method='bfill')
```
![image](https://github.com/arunkumarsukdevchavan/Mini-Project/assets/118343978/9ce7c68c-a10a-42bc-a487-e1a97e410e76)


```python
df.duplicated()
```
![image](https://github.com/arunkumarsukdevchavan/Mini-Project/assets/118343978/3f5c39e8-7414-4ab8-8caf-e64479ef2283)


```python
exp = [13,23,28,12,5,9,31,26,10,19,22,24,29,4,25,30]
af=pd.DataFrame(exp)
af
```
![image](https://github.com/arunkumarsukdevchavan/Mini-Project/assets/118343978/3e860a65-cd01-4c88-b07e-16efd148078b)


```pyhton
q1=af.quantile(0.25)
q2=af.quantile(0.5)
q3=af.quantile(0.75)
iqr=q3-q1

low=q1-1.5*iqr
low
```
![image](https://github.com/arunkumarsukdevchavan/Mini-Project/assets/118343978/aa1aeef6-ec2e-4f09-ae30-da2ca241ef79)

```python
high=q1+1.5*iqr
high
```
![image](https://github.com/arunkumarsukdevchavan/Mini-Project/assets/118343978/263dc6a4-edd3-4021-a27b-2d9d5a7a7a77)


```python
sns.boxplot(data=af)
```
![image](https://github.com/arunkumarsukdevchavan/Mini-Project/assets/118343978/f4578606-5d9c-4e4f-a4c1-aa9536d0f0fa)


```python
import pandas as pd
import matplotlib.pyplot as plt


data = pd.read_csv("addresses.csv")


plt.figure(figsize=(8, 4))
data['Desig'].value_counts().plot(kind='bar')
plt.title('Distribution of Desig')
plt.xlabel('Desig')
plt.ylabel('Count')
plt.show()
```
![image](https://github.com/arunkumarsukdevchavan/Mini-Project/assets/118343978/1dfe55bc-648b-4af4-a21a-bf85ce425ab3)


```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.pairplot(data, hue="Desig")
plt.show()
```
![image](https://github.com/arunkumarsukdevchavan/Mini-Project/assets/118343978/62508099-0339-4228-9ce2-586a62e66012)


```python
correlation_matrix = data.corr()
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()
```
![image](https://github.com/arunkumarsukdevchavan/Mini-Project/assets/118343978/f4d33d15-c94d-4d62-845f-b558989a0f65)


```python
import pandas as pd
from sklearn.preprocessing import StandardScaler

numerical_features = ['ID']
scaler = StandardScaler()
data[numerical_features] = scaler.fit_transform(data[numerical_features])
data
```
![image](https://github.com/arunkumarsukdevchavan/Mini-Project/assets/118343978/b68bc293-0db6-4550-895c-42287e81cc21)


```python
from sklearn.preprocessing import MaxAbsScaler

scaler = MaxAbsScaler()
columns_to_scale = ['ID']
data[columns_to_scale] = scaler.fit_transform(data[columns_to_scale])
data
```
![image](https://github.com/arunkumarsukdevchavan/Mini-Project/assets/118343978/8a467c0c-f09a-4e3a-ae2c-8ea0a1454372)


```python
from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()
data[['ID']] = scaler.fit_transform(data[['ID']])
data
```
![image](https://github.com/arunkumarsukdevchavan/Mini-Project/assets/118343978/499dbe08-06eb-4dd9-98f1-4b83e867ef8d)


```python
data.skew()
```
![image](https://github.com/arunkumarsukdevchavan/Mini-Project/assets/118343978/2d58891f-811d-4ef7-afc2-c8fb0dbd390f)


```python
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import scipy.stats as stats
import numpy as np

np.log(df["ID"])
```
![image](https://github.com/arunkumarsukdevchavan/Mini-Project/assets/118343978/35d34428-d967-4714-a1fa-d9ec1648cc34)


```python
np.sqrt(df["ID"])
```
![image](https://github.com/arunkumarsukdevchavan/Mini-Project/assets/118343978/da54ca4d-1063-42d2-ba8a-f08dfaa9a140)


```python
sm.qqplot(df['ID'],line='45')
plt.show()
```
![image](https://github.com/arunkumarsukdevchavan/Mini-Project/assets/118343978/af7de6ba-8e5e-409a-8f0c-2a5b0620d379)


```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(8, 4))
plt.hist(data['ID'], bins=10, color='skyblue', edgecolor='black')
plt.title('Position')
plt.xlabel('ID')
plt.ylabel('Frequency')
plt.show()
```
![image](https://github.com/arunkumarsukdevchavan/Mini-Project/assets/118343978/6c9de798-fe91-47f1-b7a9-0c82832a7053)


```python
plt.figure(figsize=(8, 4))
sns.boxplot(data=data, x='ID', color='lightcoral')
plt.title('Position Boxplot')
plt.xlabel('ID')
plt.show()
```
![image](https://github.com/arunkumarsukdevchavan/Mini-Project/assets/118343978/5e775b59-db14-429a-85f7-c94dc5ec630d)


```python
plt.figure(figsize=(10, 4))
sns.countplot(data=data, x='Desig', palette='Set2')
plt.title('Desig Counts')
plt.xlabel('Desig')
plt.ylabel('Count')
plt.show()
```
![image](https://github.com/arunkumarsukdevchavan/Mini-Project/assets/118343978/cde5d5f3-3cbb-4adf-ba3d-285b2ed8f02f)




# Result:
Hence the program to analyze the data set using data science is applied sucessfully.

