```python
from sklearn.datasets import fetch_california_housing
```


```python
cali_data = fetch_california_housing(as_frame=True)
```


```python
print( cali_data.DESCR)
```

    .. _california_housing_dataset:
    
    California Housing dataset
    --------------------------
    
    **Data Set Characteristics:**
    
        :Number of Instances: 20640
    
        :Number of Attributes: 8 numeric, predictive attributes and the target
    
        :Attribute Information:
            - MedInc        median income in block group
            - HouseAge      median house age in block group
            - AveRooms      average number of rooms per household
            - AveBedrms     average number of bedrooms per household
            - Population    block group population
            - AveOccup      average number of household members
            - Latitude      block group latitude
            - Longitude     block group longitude
    
        :Missing Attribute Values: None
    
    This dataset was obtained from the StatLib repository.
    https://www.dcc.fc.up.pt/~ltorgo/Regression/cal_housing.html
    
    The target variable is the median house value for California districts,
    expressed in hundreds of thousands of dollars ($100,000).
    
    This dataset was derived from the 1990 U.S. census, using one row per census
    block group. A block group is the smallest geographical unit for which the U.S.
    Census Bureau publishes sample data (a block group typically has a population
    of 600 to 3,000 people).
    
    An household is a group of people residing within a home. Since the average
    number of rooms and bedrooms in this dataset are provided per household, these
    columns may take surpinsingly large values for block groups with few households
    and many empty houses, such as vacation resorts.
    
    It can be downloaded/loaded using the
    :func:`sklearn.datasets.fetch_california_housing` function.
    
    .. topic:: References
    
        - Pace, R. Kelley and Ronald Barry, Sparse Spatial Autoregressions,
          Statistics and Probability Letters, 33 (1997) 291-297
    
    


```python
print(cali_data.data.head())
```

       MedInc  HouseAge  AveRooms  AveBedrms  Population  AveOccup  Latitude  \
    0  8.3252      41.0  6.984127   1.023810       322.0  2.555556     37.88   
    1  8.3014      21.0  6.238137   0.971880      2401.0  2.109842     37.86   
    2  7.2574      52.0  8.288136   1.073446       496.0  2.802260     37.85   
    3  5.6431      52.0  5.817352   1.073059       558.0  2.547945     37.85   
    4  3.8462      52.0  6.281853   1.081081       565.0  2.181467     37.85   
    
       Longitude  
    0    -122.23  
    1    -122.22  
    2    -122.24  
    3    -122.25  
    4    -122.25  
    


```python
from sklearn.model_selection import train_test_split
training_data = cali_data.data
target_value = cali_data.target
X_train, X_test, y_train, y_test = train_test_split(training_data, target_value, test_size= 0.2, random_state= 5)
```


```python
import sklearn
```


```python
from sklearn.linear_model import LinearRegression
```


```python
linear_regressor = LinearRegression()
```


```python
linear_regressor.fit(X_train,y_train)
```




    LinearRegression()




```python
y_train_predict = linear_regressor.predict(X_test)
```


```python
import matplotlib.pyplot as plt
```


```python
plt.title('Actual vs Predicted California Housing price')
plt.xlabel('Actual price ($1000s)')
plt.ylabel('Predicted price ($1000s)')
plt.scatter(y_test,y_train_predict)
plt.plot([0, 5], [0, 5], "r-")

plt.show()
```


    
![png](output_11_0.png)
    



```python
import numpy as np
```


```python
np.array([1,2,3])

```




    array([1, 2, 3])




```python
datalar = 'https://raw.githubusercontent.com/cfdonmez/Building-Data-Science-Solutions-with-Anaconda/main/Chapter04/mars_temp_data_f.txt'
```


```python
arr_1 = np.genfromtxt(datalar, delimiter=',')
```


```python
arr_1 > 20
```




    array([[ True, False, False, False,  True,  True,  True, False, False,
            False],
           [False, False, False, False,  True, False, False, False, False,
            False],
           [False, False, False, False, False, False, False, False,  True,
            False],
           [False, False, False, False, False, False, False, False, False,
            False],
           [False, False, False, False, False, False, False,  True,  True,
            False],
           [False, False,  True, False, False, False, False, False, False,
             True],
           [False, False, False, False,  True, False, False, False, False,
            False],
           [ True,  True, False, False,  True, False,  True,  True, False,
            False],
           [False, False, False, False, False, False, False, False, False,
            False],
           [False, False, False, False, False, False, False, False, False,
            False],
           [False,  True, False, False, False,  True, False, False, False,
            False],
           [False, False, False, False, False, False, False, False, False,
             True],
           [False, False, False, False, False,  True, False, False, False,
            False],
           [False, False, False, False, False, False, False, False, False,
            False],
           [False, False, False, False, False, False, False, False, False,
            False],
           [False, False,  True, False, False, False, False,  True, False,
             True],
           [ True, False, False, False, False, False, False, False,  True,
            False],
           [False, False,  True, False, False, False, False, False, False,
            False],
           [False, False, False, False, False, False, False, False,  True,
            False],
           [ True, False, False, False, False, False, False, False, False,
            False],
           [False, False, False, False,  True, False, False, False, False,
             True],
           [False, False, False, False, False, False, False, False,  True,
            False],
           [False,  True, False,  True, False, False, False, False, False,
            False],
           [False,  True, False, False,  True, False, False, False, False,
             True]])




```python
arr_conv_cel = (arr_1 - 32) * .5566
```


```python
arr_conv_cel[arr_conv_cel>10]
```




    array([], dtype=float64)




```python
datalar = 'https://github.com/cfdonmez/kuzenodev/blob/45e634787ed9a13851cd3288ccfcc513917bcc6d/Breast%20Cancer%20Prediction.csv'
```


```python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
```


```python
dataset = pd.read_csv('BreastCancerPrediction.csv')

```


```python
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
```


```python
dataset.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Sample code number</th>
      <th>Clump Thickness</th>
      <th>Uniformity of Cell Size</th>
      <th>Uniformity of Cell Shape</th>
      <th>Marginal Adhesion</th>
      <th>Single Epithelial Cell Size</th>
      <th>Bare Nuclei</th>
      <th>Bland Chromatin</th>
      <th>Normal Nucleoli</th>
      <th>Mitoses</th>
      <th>Class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1000025</td>
      <td>5</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1002945</td>
      <td>5</td>
      <td>4</td>
      <td>4</td>
      <td>5</td>
      <td>7</td>
      <td>10</td>
      <td>3</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1015425</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1016277</td>
      <td>6</td>
      <td>8</td>
      <td>8</td>
      <td>1</td>
      <td>3</td>
      <td>4</td>
      <td>3</td>
      <td>7</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1017023</td>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>2</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>




```python
dataset.isnull().sum()
```




    Sample code number             0
    Clump Thickness                0
    Uniformity of Cell Size        0
    Uniformity of Cell Shape       0
    Marginal Adhesion              0
    Single Epithelial Cell Size    0
    Bare Nuclei                    0
    Bland Chromatin                0
    Normal Nucleoli                0
    Mitoses                        0
    Class                          0
    dtype: int64




```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 43)
```


```python
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
```


```python
from xgboost import XGBClassifier

```


```python
classifier = XGBClassifier()
```


```python
X_train.shape
```




    (546, 10)




```python
y_train.shape
```




    (546,)




```python
classifier.fit(X_train, y_train)
```




    XGBClassifier()




```python
from sklearn.metrics import confusion_matrix, accuracy_score
y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)
```

    [[86  2]
     [ 6 43]]
    




    0.9416058394160584




```python
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
print("Accuracy: {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))
```

    Accuracy: 97.26 %
    Standard Deviation: 1.68 %
    


```python

```
