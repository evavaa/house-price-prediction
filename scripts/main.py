#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from pre_process import *
from visualisation import *
from numeric_calculation import *

# preprocess
fill_health()
fill_house_price()
finalize_hp()
fill_open_space()
crime()
highschool_rank19()
prischool()
school_preprocess()
highschool_weight()
primary_weight()
sum_score()
final_merge()

# plot
plot()

# calculation
minmax_normalise()
cal_pearson()
cal_nmi()
cal_regression_old()
cal_regression_new()

