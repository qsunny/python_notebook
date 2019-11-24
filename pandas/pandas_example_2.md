

```python
import numpy as np
import pandas as pd
s = pd.Series([1,3,5,np.nan,6,8])
s
```




    0    1.0
    1    3.0
    2    5.0
    3    NaN
    4    6.0
    5    8.0
    dtype: float64




```python
dates = pd.date_range('20190101',periods=6)
dates
```




    DatetimeIndex(['2019-01-01', '2019-01-02', '2019-01-03', '2019-01-04',
                   '2019-01-05', '2019-01-06'],
                  dtype='datetime64[ns]', freq='D')




```python
df = pd.DataFrame(np.random.randn(6,4),index=dates,columns=list('ABCD'))
df
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2019-01-01</th>
      <td>1.107656</td>
      <td>0.915809</td>
      <td>-0.184408</td>
      <td>-0.317617</td>
    </tr>
    <tr>
      <th>2019-01-02</th>
      <td>-0.760998</td>
      <td>0.110140</td>
      <td>-1.406992</td>
      <td>-0.430301</td>
    </tr>
    <tr>
      <th>2019-01-03</th>
      <td>-0.493568</td>
      <td>0.611598</td>
      <td>-0.149030</td>
      <td>0.212389</td>
    </tr>
    <tr>
      <th>2019-01-04</th>
      <td>1.418031</td>
      <td>1.010258</td>
      <td>0.446002</td>
      <td>0.991972</td>
    </tr>
    <tr>
      <th>2019-01-05</th>
      <td>-0.316527</td>
      <td>1.789683</td>
      <td>-1.924230</td>
      <td>-0.437339</td>
    </tr>
    <tr>
      <th>2019-01-06</th>
      <td>0.603124</td>
      <td>-0.981076</td>
      <td>-0.857155</td>
      <td>-1.427457</td>
    </tr>
  </tbody>
</table>
</div>




```python
df2 = pd.DataFrame({'A': 1.,
      'B': pd.Timestamp('20190102'),
      'C': pd.Series(1, index=list(range(4)), dtype='float32'),
      'D': np.array([3] * 4, dtype='int32'),
      'E': pd.Categorical(["test", "train", "test", "train"]),
      'F': 'foo'})
df2
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
      <th>E</th>
      <th>F</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.0</td>
      <td>2019-01-02</td>
      <td>1.0</td>
      <td>3</td>
      <td>test</td>
      <td>foo</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.0</td>
      <td>2019-01-02</td>
      <td>1.0</td>
      <td>3</td>
      <td>train</td>
      <td>foo</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.0</td>
      <td>2019-01-02</td>
      <td>1.0</td>
      <td>3</td>
      <td>test</td>
      <td>foo</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.0</td>
      <td>2019-01-02</td>
      <td>1.0</td>
      <td>3</td>
      <td>train</td>
      <td>foo</td>
    </tr>
  </tbody>
</table>
</div>




```python
df2.dtypes
```




    A           float64
    B    datetime64[ns]
    C           float32
    D             int32
    E          category
    F            object
    dtype: object




```python
df.head()

```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2019-01-01</th>
      <td>1.107656</td>
      <td>0.915809</td>
      <td>-0.184408</td>
      <td>-0.317617</td>
    </tr>
    <tr>
      <th>2019-01-02</th>
      <td>-0.760998</td>
      <td>0.110140</td>
      <td>-1.406992</td>
      <td>-0.430301</td>
    </tr>
    <tr>
      <th>2019-01-03</th>
      <td>-0.493568</td>
      <td>0.611598</td>
      <td>-0.149030</td>
      <td>0.212389</td>
    </tr>
    <tr>
      <th>2019-01-04</th>
      <td>1.418031</td>
      <td>1.010258</td>
      <td>0.446002</td>
      <td>0.991972</td>
    </tr>
    <tr>
      <th>2019-01-05</th>
      <td>-0.316527</td>
      <td>1.789683</td>
      <td>-1.924230</td>
      <td>-0.437339</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.tail(2)
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2019-01-05</th>
      <td>-0.316527</td>
      <td>1.789683</td>
      <td>-1.924230</td>
      <td>-0.437339</td>
    </tr>
    <tr>
      <th>2019-01-06</th>
      <td>0.603124</td>
      <td>-0.981076</td>
      <td>-0.857155</td>
      <td>-1.427457</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.index
```




    DatetimeIndex(['2019-01-01', '2019-01-02', '2019-01-03', '2019-01-04',
                   '2019-01-05', '2019-01-06'],
                  dtype='datetime64[ns]', freq='D')




```python
df.columns
```




    Index(['A', 'B', 'C', 'D'], dtype='object')




```python
df.describe()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>6.000000</td>
      <td>6.000000</td>
      <td>6.000000</td>
      <td>6.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.259620</td>
      <td>0.576069</td>
      <td>-0.679302</td>
      <td>-0.234726</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.907742</td>
      <td>0.940197</td>
      <td>0.883564</td>
      <td>0.801683</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-0.760998</td>
      <td>-0.981076</td>
      <td>-1.924230</td>
      <td>-1.427457</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>-0.449307</td>
      <td>0.235504</td>
      <td>-1.269533</td>
      <td>-0.435580</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.143298</td>
      <td>0.763703</td>
      <td>-0.520782</td>
      <td>-0.373959</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.981523</td>
      <td>0.986646</td>
      <td>-0.157874</td>
      <td>0.079888</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1.418031</td>
      <td>1.789683</td>
      <td>0.446002</td>
      <td>0.991972</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.T
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>2019-01-01 00:00:00</th>
      <th>2019-01-02 00:00:00</th>
      <th>2019-01-03 00:00:00</th>
      <th>2019-01-04 00:00:00</th>
      <th>2019-01-05 00:00:00</th>
      <th>2019-01-06 00:00:00</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>A</th>
      <td>1.107656</td>
      <td>-0.760998</td>
      <td>-0.493568</td>
      <td>1.418031</td>
      <td>-0.316527</td>
      <td>0.603124</td>
    </tr>
    <tr>
      <th>B</th>
      <td>0.915809</td>
      <td>0.110140</td>
      <td>0.611598</td>
      <td>1.010258</td>
      <td>1.789683</td>
      <td>-0.981076</td>
    </tr>
    <tr>
      <th>C</th>
      <td>-0.184408</td>
      <td>-1.406992</td>
      <td>-0.149030</td>
      <td>0.446002</td>
      <td>-1.924230</td>
      <td>-0.857155</td>
    </tr>
    <tr>
      <th>D</th>
      <td>-0.317617</td>
      <td>-0.430301</td>
      <td>0.212389</td>
      <td>0.991972</td>
      <td>-0.437339</td>
      <td>-1.427457</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.sort_index(axis=1, ascending=False)
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>D</th>
      <th>C</th>
      <th>B</th>
      <th>A</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2019-01-01</th>
      <td>-0.317617</td>
      <td>-0.184408</td>
      <td>0.915809</td>
      <td>1.107656</td>
    </tr>
    <tr>
      <th>2019-01-02</th>
      <td>-0.430301</td>
      <td>-1.406992</td>
      <td>0.110140</td>
      <td>-0.760998</td>
    </tr>
    <tr>
      <th>2019-01-03</th>
      <td>0.212389</td>
      <td>-0.149030</td>
      <td>0.611598</td>
      <td>-0.493568</td>
    </tr>
    <tr>
      <th>2019-01-04</th>
      <td>0.991972</td>
      <td>0.446002</td>
      <td>1.010258</td>
      <td>1.418031</td>
    </tr>
    <tr>
      <th>2019-01-05</th>
      <td>-0.437339</td>
      <td>-1.924230</td>
      <td>1.789683</td>
      <td>-0.316527</td>
    </tr>
    <tr>
      <th>2019-01-06</th>
      <td>-1.427457</td>
      <td>-0.857155</td>
      <td>-0.981076</td>
      <td>0.603124</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.sort_values(by='B')
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2019-01-06</th>
      <td>0.603124</td>
      <td>-0.981076</td>
      <td>-0.857155</td>
      <td>-1.427457</td>
    </tr>
    <tr>
      <th>2019-01-02</th>
      <td>-0.760998</td>
      <td>0.110140</td>
      <td>-1.406992</td>
      <td>-0.430301</td>
    </tr>
    <tr>
      <th>2019-01-03</th>
      <td>-0.493568</td>
      <td>0.611598</td>
      <td>-0.149030</td>
      <td>0.212389</td>
    </tr>
    <tr>
      <th>2019-01-01</th>
      <td>1.107656</td>
      <td>0.915809</td>
      <td>-0.184408</td>
      <td>-0.317617</td>
    </tr>
    <tr>
      <th>2019-01-04</th>
      <td>1.418031</td>
      <td>1.010258</td>
      <td>0.446002</td>
      <td>0.991972</td>
    </tr>
    <tr>
      <th>2019-01-05</th>
      <td>-0.316527</td>
      <td>1.789683</td>
      <td>-1.924230</td>
      <td>-0.437339</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.iloc[1:3, :]
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2019-01-02</th>
      <td>-0.760998</td>
      <td>0.110140</td>
      <td>-1.406992</td>
      <td>-0.430301</td>
    </tr>
    <tr>
      <th>2019-01-03</th>
      <td>-0.493568</td>
      <td>0.611598</td>
      <td>-0.149030</td>
      <td>0.212389</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.iloc[:, 1:3]
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>B</th>
      <th>C</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2019-01-01</th>
      <td>0.915809</td>
      <td>-0.184408</td>
    </tr>
    <tr>
      <th>2019-01-02</th>
      <td>0.110140</td>
      <td>-1.406992</td>
    </tr>
    <tr>
      <th>2019-01-03</th>
      <td>0.611598</td>
      <td>-0.149030</td>
    </tr>
    <tr>
      <th>2019-01-04</th>
      <td>1.010258</td>
      <td>0.446002</td>
    </tr>
    <tr>
      <th>2019-01-05</th>
      <td>1.789683</td>
      <td>-1.924230</td>
    </tr>
    <tr>
      <th>2019-01-06</th>
      <td>-0.981076</td>
      <td>-0.857155</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.iloc[1, 1]
```




    0.11013976365693386




```python
#等价前面方法
df.iat[1, 1]
```




    0.11013976365693386




```python
df[df.A > 0]
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2019-01-01</th>
      <td>1.107656</td>
      <td>0.915809</td>
      <td>-0.184408</td>
      <td>-0.317617</td>
    </tr>
    <tr>
      <th>2019-01-04</th>
      <td>1.418031</td>
      <td>1.010258</td>
      <td>0.446002</td>
      <td>0.991972</td>
    </tr>
    <tr>
      <th>2019-01-06</th>
      <td>0.603124</td>
      <td>-0.981076</td>
      <td>-0.857155</td>
      <td>-1.427457</td>
    </tr>
  </tbody>
</table>
</div>




```python
df[df > 0]
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2019-01-01</th>
      <td>1.107656</td>
      <td>0.915809</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2019-01-02</th>
      <td>NaN</td>
      <td>0.110140</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2019-01-03</th>
      <td>NaN</td>
      <td>0.611598</td>
      <td>NaN</td>
      <td>0.212389</td>
    </tr>
    <tr>
      <th>2019-01-04</th>
      <td>1.418031</td>
      <td>1.010258</td>
      <td>0.446002</td>
      <td>0.991972</td>
    </tr>
    <tr>
      <th>2019-01-05</th>
      <td>NaN</td>
      <td>1.789683</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2019-01-06</th>
      <td>0.603124</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
df2 = df.copy()
df2['E'] = ['one', 'one', 'two', 'three', 'four', 'three']
print(df2)
df2[df2['E'].isin(['two', 'four'])]
```

                       A         B         C         D      E
    2019-01-01  1.107656  0.915809 -0.184408 -0.317617    one
    2019-01-02 -0.760998  0.110140 -1.406992 -0.430301    one
    2019-01-03 -0.493568  0.611598 -0.149030  0.212389    two
    2019-01-04  1.418031  1.010258  0.446002  0.991972  three
    2019-01-05 -0.316527  1.789683 -1.924230 -0.437339   four
    2019-01-06  0.603124 -0.981076 -0.857155 -1.427457  three
    




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
      <th>E</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2019-01-03</th>
      <td>-0.493568</td>
      <td>0.611598</td>
      <td>-0.14903</td>
      <td>0.212389</td>
      <td>two</td>
    </tr>
    <tr>
      <th>2019-01-05</th>
      <td>-0.316527</td>
      <td>1.789683</td>
      <td>-1.92423</td>
      <td>-0.437339</td>
      <td>four</td>
    </tr>
  </tbody>
</table>
</div>


