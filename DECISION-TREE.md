```python
import pandas as pd
import numpy as np
```


```python
df=pd.read_csv('E:\\7th sem\\ML-LAB\\ML_lab_7th-Sem-main\\ML_lab_7th-Sem-main\\zoo.csv')
```


```python
df
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
      <th>animal_name</th>
      <th>hair</th>
      <th>feathers</th>
      <th>eggs</th>
      <th>milk</th>
      <th>airborne</th>
      <th>aquatic</th>
      <th>predator</th>
      <th>toothed</th>
      <th>backbone</th>
      <th>breathes</th>
      <th>venomous</th>
      <th>fins</th>
      <th>legs</th>
      <th>tail</th>
      <th>domestic</th>
      <th>catsize</th>
      <th>class_type</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>aardvark</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>antelope</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>4</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>bass</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>4</td>
    </tr>
    <tr>
      <th>3</th>
      <td>bear</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>boar</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>4</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>96</th>
      <td>wallaby</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>97</th>
      <td>wasp</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>6</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>6</td>
    </tr>
    <tr>
      <th>98</th>
      <td>wolf</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>4</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>99</th>
      <td>worm</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>7</td>
    </tr>
    <tr>
      <th>100</th>
      <td>wren</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
<p>101 rows × 18 columns</p>
</div>




```python
labels=df.columns.tolist()

```


```python
labels
```




    ['animal_name',
     'hair',
     'feathers',
     'eggs',
     'milk',
     'airborne',
     'aquatic',
     'predator',
     'toothed',
     'backbone',
     'breathes',
     'venomous',
     'fins',
     'legs',
     'tail',
     'domestic',
     'catsize',
     'class_type']




```python
x=df[labels[1:-1]]
y=df[labels[-1:]]
```


```python
x
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
      <th>hair</th>
      <th>feathers</th>
      <th>eggs</th>
      <th>milk</th>
      <th>airborne</th>
      <th>aquatic</th>
      <th>predator</th>
      <th>toothed</th>
      <th>backbone</th>
      <th>breathes</th>
      <th>venomous</th>
      <th>fins</th>
      <th>legs</th>
      <th>tail</th>
      <th>domestic</th>
      <th>catsize</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>4</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>4</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>96</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>97</th>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>6</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>98</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>4</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>99</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>100</th>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>101 rows × 16 columns</p>
</div>




```python
y
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
      <th>class_type</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th>96</th>
      <td>1</td>
    </tr>
    <tr>
      <th>97</th>
      <td>6</td>
    </tr>
    <tr>
      <th>98</th>
      <td>1</td>
    </tr>
    <tr>
      <th>99</th>
      <td>7</td>
    </tr>
    <tr>
      <th>100</th>
      <td>2</td>
    </tr>
  </tbody>
</table>
<p>101 rows × 1 columns</p>
</div>




```python
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
```


```python
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25)
```


```python
clf=DecisionTreeClassifier()
clf.fit(x_train,y_train)
y_pred=clf.predict(x_test)
```


```python
y_pred
```




    array([1, 4, 1, 1, 4, 4, 4, 7, 6, 5, 1, 2, 6, 1, 2, 2, 2, 6, 1, 2, 1, 1,
           4, 1, 6, 2], dtype=int64)




```python
print(accuracy_score(y_test,y_pred))
```

    1.0
    


```python
print(classification_report(y_test,y_pred))
```

                  precision    recall  f1-score   support
    
               1       1.00      1.00      1.00         9
               2       1.00      1.00      1.00         6
               4       1.00      1.00      1.00         5
               5       1.00      1.00      1.00         1
               6       1.00      1.00      1.00         4
               7       1.00      1.00      1.00         1
    
        accuracy                           1.00        26
       macro avg       1.00      1.00      1.00        26
    weighted avg       1.00      1.00      1.00        26
    
    


```python
print(confusion_matrix(y_test,y_pred))
```

    [[9 0 0 0 0 0]
     [0 6 0 0 0 0]
     [0 0 5 0 0 0]
     [0 0 0 1 0 0]
     [0 0 0 0 4 0]
     [0 0 0 0 0 1]]
    


```python
actual_class=list(y_test['class_type'])
```


```python
actual_class
```




    [1, 4, 1, 1, 4, 4, 4, 7, 6, 5, 1, 2, 6, 1, 2, 2, 2, 6, 1, 2, 1, 1, 4, 1, 6, 2]




```python
pred_class=list(y_pred)
```


```python
pred_class
```




    [1, 4, 1, 1, 4, 4, 4, 7, 6, 5, 1, 2, 6, 1, 2, 2, 2, 6, 1, 2, 1, 1, 4, 1, 6, 2]




```python
for i in range(len(actual_class)):
    print("Actual class =",actual_class[i]," Predicted class = ",pred_class[i])
```

    Actual class = 1  Predicted class =  1
    Actual class = 4  Predicted class =  4
    Actual class = 1  Predicted class =  1
    Actual class = 1  Predicted class =  1
    Actual class = 4  Predicted class =  4
    Actual class = 4  Predicted class =  4
    Actual class = 4  Predicted class =  4
    Actual class = 7  Predicted class =  7
    Actual class = 6  Predicted class =  6
    Actual class = 5  Predicted class =  5
    Actual class = 1  Predicted class =  1
    Actual class = 2  Predicted class =  2
    Actual class = 6  Predicted class =  6
    Actual class = 1  Predicted class =  1
    Actual class = 2  Predicted class =  2
    Actual class = 2  Predicted class =  2
    Actual class = 2  Predicted class =  2
    Actual class = 6  Predicted class =  6
    Actual class = 1  Predicted class =  1
    Actual class = 2  Predicted class =  2
    Actual class = 1  Predicted class =  1
    Actual class = 1  Predicted class =  1
    Actual class = 4  Predicted class =  4
    Actual class = 1  Predicted class =  1
    Actual class = 6  Predicted class =  6
    Actual class = 2  Predicted class =  2
    


```python

```
