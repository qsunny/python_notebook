import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


ts = pd.Series(np.random.randn(1000),
   index=pd.date_range('1/1/2019', periods=1000))
ts = ts.cumsum()
ts.plot()