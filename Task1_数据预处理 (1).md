

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
```


```python
df_data=pd.read_csv('data1.csv',index_col=0)   # 将之前的csv改成 csv utf-8读取，不然报错
```


```python
df_data.head()
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
      <th>custid</th>
      <th>trade_no</th>
      <th>bank_card_no</th>
      <th>low_volume_percent</th>
      <th>middle_volume_percent</th>
      <th>take_amount_in_later_12_month_highest</th>
      <th>trans_amount_increase_rate_lately</th>
      <th>trans_activity_month</th>
      <th>trans_activity_day</th>
      <th>transd_mcc</th>
      <th>...</th>
      <th>loans_max_limit</th>
      <th>loans_avg_limit</th>
      <th>consfin_credit_limit</th>
      <th>consfin_credibility</th>
      <th>consfin_org_count_current</th>
      <th>consfin_product_count</th>
      <th>consfin_max_limit</th>
      <th>consfin_avg_limit</th>
      <th>latest_query_day</th>
      <th>loans_latest_day</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>5</th>
      <td>2791858</td>
      <td>2.018050e+31</td>
      <td>卡号1</td>
      <td>0.01</td>
      <td>0.99</td>
      <td>0</td>
      <td>0.90</td>
      <td>0.55</td>
      <td>0.313</td>
      <td>17.0</td>
      <td>...</td>
      <td>2900.0</td>
      <td>1688.0</td>
      <td>1200.0</td>
      <td>75.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>1200.0</td>
      <td>1200.0</td>
      <td>12.0</td>
      <td>18.0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>534047</td>
      <td>2.018050e+31</td>
      <td>卡号1</td>
      <td>0.02</td>
      <td>0.94</td>
      <td>2000</td>
      <td>1.28</td>
      <td>1.00</td>
      <td>0.458</td>
      <td>19.0</td>
      <td>...</td>
      <td>3500.0</td>
      <td>1758.0</td>
      <td>15100.0</td>
      <td>80.0</td>
      <td>5.0</td>
      <td>6.0</td>
      <td>22800.0</td>
      <td>9360.0</td>
      <td>4.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>12</th>
      <td>2849787</td>
      <td>2.018050e+31</td>
      <td>卡号1</td>
      <td>0.04</td>
      <td>0.96</td>
      <td>0</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>0.114</td>
      <td>13.0</td>
      <td>...</td>
      <td>1600.0</td>
      <td>1250.0</td>
      <td>4200.0</td>
      <td>87.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>4200.0</td>
      <td>4200.0</td>
      <td>2.0</td>
      <td>6.0</td>
    </tr>
    <tr>
      <th>13</th>
      <td>1809708</td>
      <td>2.018050e+31</td>
      <td>卡号1</td>
      <td>0.00</td>
      <td>0.96</td>
      <td>2000</td>
      <td>0.13</td>
      <td>0.57</td>
      <td>0.777</td>
      <td>22.0</td>
      <td>...</td>
      <td>3200.0</td>
      <td>1541.0</td>
      <td>16300.0</td>
      <td>80.0</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>30000.0</td>
      <td>12180.0</td>
      <td>2.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>14</th>
      <td>2499829</td>
      <td>2.018050e+31</td>
      <td>卡号1</td>
      <td>0.01</td>
      <td>0.99</td>
      <td>0</td>
      <td>0.46</td>
      <td>1.00</td>
      <td>0.175</td>
      <td>13.0</td>
      <td>...</td>
      <td>2300.0</td>
      <td>1630.0</td>
      <td>8300.0</td>
      <td>79.0</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>8400.0</td>
      <td>8250.0</td>
      <td>22.0</td>
      <td>120.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 89 columns</p>
</div>




```python
df_data.shape  #有4754个样本，89个特征
```




    (4754, 89)



## 1.对数据类型进行分析


```python
df_data.info()   #查看数据类型
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 4754 entries, 5 to 11992
    Data columns (total 89 columns):
    custid                                        4754 non-null int64
    trade_no                                      4754 non-null float64
    bank_card_no                                  4754 non-null object
    low_volume_percent                            4752 non-null float64
    middle_volume_percent                         4752 non-null float64
    take_amount_in_later_12_month_highest         4754 non-null int64
    trans_amount_increase_rate_lately             4751 non-null float64
    trans_activity_month                          4752 non-null float64
    trans_activity_day                            4752 non-null float64
    transd_mcc                                    4752 non-null float64
    trans_days_interval_filter                    4746 non-null float64
    trans_days_interval                           4752 non-null float64
    regional_mobility                             4752 non-null float64
    student_feature                               1756 non-null float64
    repayment_capability                          4754 non-null int64
    is_high_user                                  4754 non-null int64
    number_of_trans_from_2011                     4752 non-null float64
    first_transaction_time                        4752 non-null float64
    historical_trans_amount                       4754 non-null int64
    historical_trans_day                          4752 non-null float64
    rank_trad_1_month                             4752 non-null float64
    trans_amount_3_month                          4754 non-null int64
    avg_consume_less_12_valid_month               4752 non-null float64
    abs                                           4754 non-null int64
    top_trans_count_last_1_month                  4752 non-null float64
    avg_price_last_12_month                       4754 non-null int64
    avg_price_top_last_12_valid_month             4650 non-null float64
    reg_preference_for_trad                       4752 non-null object
    trans_top_time_last_1_month                   4746 non-null float64
    trans_top_time_last_6_month                   4746 non-null float64
    consume_top_time_last_1_month                 4746 non-null float64
    consume_top_time_last_6_month                 4746 non-null float64
    cross_consume_count_last_1_month              4328 non-null float64
    trans_fail_top_count_enum_last_1_month        4738 non-null float64
    trans_fail_top_count_enum_last_6_month        4738 non-null float64
    trans_fail_top_count_enum_last_12_month       4738 non-null float64
    consume_mini_time_last_1_month                4728 non-null float64
    max_cumulative_consume_later_1_month          4754 non-null int64
    max_consume_count_later_6_month               4746 non-null float64
    railway_consume_count_last_12_month           4742 non-null float64
    pawns_auctions_trusts_consume_last_1_month    4754 non-null int64
    pawns_auctions_trusts_consume_last_6_month    4754 non-null int64
    jewelry_consume_count_last_6_month            4742 non-null float64
    status                                        4754 non-null int64
    source                                        4754 non-null object
    first_transaction_day                         4752 non-null float64
    trans_day_last_12_month                       4752 non-null float64
    id_name                                       4478 non-null object
    apply_score                                   4450 non-null float64
    apply_credibility                             4450 non-null float64
    query_org_count                               4450 non-null float64
    query_finance_count                           4450 non-null float64
    query_cash_count                              4450 non-null float64
    query_sum_count                               4450 non-null float64
    latest_query_time                             4450 non-null object
    latest_one_month_apply                        4450 non-null float64
    latest_three_month_apply                      4450 non-null float64
    latest_six_month_apply                        4450 non-null float64
    loans_score                                   4457 non-null float64
    loans_credibility_behavior                    4457 non-null float64
    loans_count                                   4457 non-null float64
    loans_settle_count                            4457 non-null float64
    loans_overdue_count                           4457 non-null float64
    loans_org_count_behavior                      4457 non-null float64
    consfin_org_count_behavior                    4457 non-null float64
    loans_cash_count                              4457 non-null float64
    latest_one_month_loan                         4457 non-null float64
    latest_three_month_loan                       4457 non-null float64
    latest_six_month_loan                         4457 non-null float64
    history_suc_fee                               4457 non-null float64
    history_fail_fee                              4457 non-null float64
    latest_one_month_suc                          4457 non-null float64
    latest_one_month_fail                         4457 non-null float64
    loans_long_time                               4457 non-null float64
    loans_latest_time                             4457 non-null object
    loans_credit_limit                            4457 non-null float64
    loans_credibility_limit                       4457 non-null float64
    loans_org_count_current                       4457 non-null float64
    loans_product_count                           4457 non-null float64
    loans_max_limit                               4457 non-null float64
    loans_avg_limit                               4457 non-null float64
    consfin_credit_limit                          4457 non-null float64
    consfin_credibility                           4457 non-null float64
    consfin_org_count_current                     4457 non-null float64
    consfin_product_count                         4457 non-null float64
    consfin_max_limit                             4457 non-null float64
    consfin_avg_limit                             4457 non-null float64
    latest_query_day                              4450 non-null float64
    loans_latest_day                              4457 non-null float64
    dtypes: float64(71), int64(12), object(6)
    memory usage: 3.3+ MB
    

对其中的object类型，之后要转化成数值，或者删除。

### 对无关特征进行删除


```python
df_data.describe().T
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
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>custid</th>
      <td>4754.0</td>
      <td>1.690993e+06</td>
      <td>1.034235e+06</td>
      <td>1.140000e+02</td>
      <td>7.593358e+05</td>
      <td>1.634942e+06</td>
      <td>2.597905e+06</td>
      <td>4.004694e+06</td>
    </tr>
    <tr>
      <th>trade_no</th>
      <td>4754.0</td>
      <td>2.018050e+31</td>
      <td>1.812890e+18</td>
      <td>2.018050e+31</td>
      <td>2.018050e+31</td>
      <td>2.018050e+31</td>
      <td>2.018050e+31</td>
      <td>2.018050e+31</td>
    </tr>
    <tr>
      <th>low_volume_percent</th>
      <td>4752.0</td>
      <td>2.180556e-02</td>
      <td>4.152712e-02</td>
      <td>0.000000e+00</td>
      <td>1.000000e-02</td>
      <td>1.000000e-02</td>
      <td>2.000000e-02</td>
      <td>1.000000e+00</td>
    </tr>
    <tr>
      <th>middle_volume_percent</th>
      <td>4752.0</td>
      <td>9.012942e-01</td>
      <td>1.448561e-01</td>
      <td>0.000000e+00</td>
      <td>8.800000e-01</td>
      <td>9.600000e-01</td>
      <td>9.900000e-01</td>
      <td>1.000000e+00</td>
    </tr>
    <tr>
      <th>take_amount_in_later_12_month_highest</th>
      <td>4754.0</td>
      <td>1.940198e+03</td>
      <td>3.923971e+03</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>5.000000e+02</td>
      <td>2.000000e+03</td>
      <td>6.800000e+04</td>
    </tr>
    <tr>
      <th>trans_amount_increase_rate_lately</th>
      <td>4751.0</td>
      <td>1.416067e+01</td>
      <td>6.941805e+02</td>
      <td>0.000000e+00</td>
      <td>6.150000e-01</td>
      <td>9.700000e-01</td>
      <td>1.600000e+00</td>
      <td>4.759674e+04</td>
    </tr>
    <tr>
      <th>trans_activity_month</th>
      <td>4752.0</td>
      <td>8.044108e-01</td>
      <td>1.969205e-01</td>
      <td>1.200000e-01</td>
      <td>6.700000e-01</td>
      <td>8.600000e-01</td>
      <td>1.000000e+00</td>
      <td>1.000000e+00</td>
    </tr>
    <tr>
      <th>trans_activity_day</th>
      <td>4752.0</td>
      <td>3.654253e-01</td>
      <td>1.701962e-01</td>
      <td>3.300000e-02</td>
      <td>2.330000e-01</td>
      <td>3.500000e-01</td>
      <td>4.800000e-01</td>
      <td>9.410000e-01</td>
    </tr>
    <tr>
      <th>transd_mcc</th>
      <td>4752.0</td>
      <td>1.750295e+01</td>
      <td>4.475616e+00</td>
      <td>2.000000e+00</td>
      <td>1.500000e+01</td>
      <td>1.700000e+01</td>
      <td>2.000000e+01</td>
      <td>4.200000e+01</td>
    </tr>
    <tr>
      <th>trans_days_interval_filter</th>
      <td>4746.0</td>
      <td>2.902992e+01</td>
      <td>2.272243e+01</td>
      <td>0.000000e+00</td>
      <td>1.600000e+01</td>
      <td>2.300000e+01</td>
      <td>3.200000e+01</td>
      <td>2.850000e+02</td>
    </tr>
    <tr>
      <th>trans_days_interval</th>
      <td>4752.0</td>
      <td>2.175126e+01</td>
      <td>1.647492e+01</td>
      <td>4.000000e+00</td>
      <td>1.200000e+01</td>
      <td>1.700000e+01</td>
      <td>2.700000e+01</td>
      <td>2.340000e+02</td>
    </tr>
    <tr>
      <th>regional_mobility</th>
      <td>4752.0</td>
      <td>2.678662e+00</td>
      <td>8.903605e-01</td>
      <td>1.000000e+00</td>
      <td>2.000000e+00</td>
      <td>3.000000e+00</td>
      <td>3.000000e+00</td>
      <td>5.000000e+00</td>
    </tr>
    <tr>
      <th>student_feature</th>
      <td>1756.0</td>
      <td>1.001139e+00</td>
      <td>3.373875e-02</td>
      <td>1.000000e+00</td>
      <td>1.000000e+00</td>
      <td>1.000000e+00</td>
      <td>1.000000e+00</td>
      <td>2.000000e+00</td>
    </tr>
    <tr>
      <th>repayment_capability</th>
      <td>4754.0</td>
      <td>1.870201e+04</td>
      <td>5.221783e+04</td>
      <td>0.000000e+00</td>
      <td>8.590000e+03</td>
      <td>1.221000e+04</td>
      <td>1.764750e+04</td>
      <td>2.459390e+06</td>
    </tr>
    <tr>
      <th>is_high_user</th>
      <td>4754.0</td>
      <td>1.114851e-02</td>
      <td>1.050073e-01</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>1.000000e+00</td>
    </tr>
    <tr>
      <th>number_of_trans_from_2011</th>
      <td>4752.0</td>
      <td>2.303388e+01</td>
      <td>1.005784e+01</td>
      <td>1.000000e+00</td>
      <td>1.600000e+01</td>
      <td>2.100000e+01</td>
      <td>2.900000e+01</td>
      <td>8.500000e+01</td>
    </tr>
    <tr>
      <th>first_transaction_time</th>
      <td>4752.0</td>
      <td>2.015109e+07</td>
      <td>1.480487e+04</td>
      <td>2.011010e+07</td>
      <td>2.014102e+07</td>
      <td>2.015111e+07</td>
      <td>2.016083e+07</td>
      <td>2.018011e+07</td>
    </tr>
    <tr>
      <th>historical_trans_amount</th>
      <td>4754.0</td>
      <td>2.307359e+05</td>
      <td>3.204931e+05</td>
      <td>0.000000e+00</td>
      <td>7.949750e+04</td>
      <td>1.623350e+05</td>
      <td>2.985600e+05</td>
      <td>1.360130e+07</td>
    </tr>
    <tr>
      <th>historical_trans_day</th>
      <td>4752.0</td>
      <td>1.761094e+02</td>
      <td>9.968729e+01</td>
      <td>2.000000e+00</td>
      <td>1.020000e+02</td>
      <td>1.600000e+02</td>
      <td>2.310000e+02</td>
      <td>9.070000e+02</td>
    </tr>
    <tr>
      <th>rank_trad_1_month</th>
      <td>4752.0</td>
      <td>4.769255e-01</td>
      <td>2.637694e-01</td>
      <td>5.000000e-02</td>
      <td>3.000000e-01</td>
      <td>4.500000e-01</td>
      <td>6.000000e-01</td>
      <td>1.000000e+00</td>
    </tr>
    <tr>
      <th>trans_amount_3_month</th>
      <td>4754.0</td>
      <td>3.896430e+04</td>
      <td>1.017461e+05</td>
      <td>0.000000e+00</td>
      <td>1.168250e+04</td>
      <td>2.555500e+04</td>
      <td>4.795000e+04</td>
      <td>6.024100e+06</td>
    </tr>
    <tr>
      <th>avg_consume_less_12_valid_month</th>
      <td>4752.0</td>
      <td>6.572601e+00</td>
      <td>1.390723e+00</td>
      <td>0.000000e+00</td>
      <td>6.000000e+00</td>
      <td>7.000000e+00</td>
      <td>7.000000e+00</td>
      <td>1.100000e+01</td>
    </tr>
    <tr>
      <th>abs</th>
      <td>4754.0</td>
      <td>9.344350e+03</td>
      <td>2.700760e+04</td>
      <td>0.000000e+00</td>
      <td>1.290000e+03</td>
      <td>3.345000e+03</td>
      <td>8.067500e+03</td>
      <td>9.184500e+05</td>
    </tr>
    <tr>
      <th>top_trans_count_last_1_month</th>
      <td>4752.0</td>
      <td>3.557449e-01</td>
      <td>3.505951e-01</td>
      <td>5.000000e-02</td>
      <td>8.750000e-02</td>
      <td>2.000000e-01</td>
      <td>6.500000e-01</td>
      <td>1.000000e+00</td>
    </tr>
    <tr>
      <th>avg_price_last_12_month</th>
      <td>4754.0</td>
      <td>1.237089e+03</td>
      <td>7.658736e+02</td>
      <td>0.000000e+00</td>
      <td>9.200000e+02</td>
      <td>1.140000e+03</td>
      <td>1.400000e+03</td>
      <td>2.314000e+04</td>
    </tr>
    <tr>
      <th>avg_price_top_last_12_valid_month</th>
      <td>4650.0</td>
      <td>5.146667e-01</td>
      <td>1.003969e-01</td>
      <td>5.000000e-02</td>
      <td>4.500000e-01</td>
      <td>5.000000e-01</td>
      <td>5.500000e-01</td>
      <td>1.000000e+00</td>
    </tr>
    <tr>
      <th>trans_top_time_last_1_month</th>
      <td>4746.0</td>
      <td>7.134008e+00</td>
      <td>5.318254e+00</td>
      <td>0.000000e+00</td>
      <td>3.250000e+00</td>
      <td>7.000000e+00</td>
      <td>1.000000e+01</td>
      <td>2.700000e+01</td>
    </tr>
    <tr>
      <th>trans_top_time_last_6_month</th>
      <td>4746.0</td>
      <td>2.017467e+01</td>
      <td>1.296298e+01</td>
      <td>0.000000e+00</td>
      <td>1.200000e+01</td>
      <td>1.700000e+01</td>
      <td>2.600000e+01</td>
      <td>1.240000e+02</td>
    </tr>
    <tr>
      <th>consume_top_time_last_1_month</th>
      <td>4746.0</td>
      <td>7.047198e+00</td>
      <td>5.456050e+00</td>
      <td>0.000000e+00</td>
      <td>3.000000e+00</td>
      <td>7.000000e+00</td>
      <td>1.000000e+01</td>
      <td>2.700000e+01</td>
    </tr>
    <tr>
      <th>consume_top_time_last_6_month</th>
      <td>4746.0</td>
      <td>2.064960e+01</td>
      <td>1.312522e+01</td>
      <td>0.000000e+00</td>
      <td>1.200000e+01</td>
      <td>1.800000e+01</td>
      <td>2.600000e+01</td>
      <td>1.510000e+02</td>
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
    </tr>
    <tr>
      <th>loans_score</th>
      <td>4457.0</td>
      <td>5.432060e+02</td>
      <td>6.095427e+01</td>
      <td>4.130000e+02</td>
      <td>4.930000e+02</td>
      <td>5.110000e+02</td>
      <td>6.020000e+02</td>
      <td>6.880000e+02</td>
    </tr>
    <tr>
      <th>loans_credibility_behavior</th>
      <td>4457.0</td>
      <td>7.543864e+01</td>
      <td>2.231822e+00</td>
      <td>5.600000e+01</td>
      <td>7.400000e+01</td>
      <td>7.500000e+01</td>
      <td>7.700000e+01</td>
      <td>8.500000e+01</td>
    </tr>
    <tr>
      <th>loans_count</th>
      <td>4457.0</td>
      <td>3.595221e+01</td>
      <td>2.461436e+01</td>
      <td>1.000000e+00</td>
      <td>1.700000e+01</td>
      <td>3.100000e+01</td>
      <td>5.000000e+01</td>
      <td>1.580000e+02</td>
    </tr>
    <tr>
      <th>loans_settle_count</th>
      <td>4457.0</td>
      <td>3.103994e+01</td>
      <td>2.169407e+01</td>
      <td>0.000000e+00</td>
      <td>1.500000e+01</td>
      <td>2.700000e+01</td>
      <td>4.300000e+01</td>
      <td>1.540000e+02</td>
    </tr>
    <tr>
      <th>loans_overdue_count</th>
      <td>4457.0</td>
      <td>2.308952e+00</td>
      <td>3.152881e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>1.000000e+00</td>
      <td>3.000000e+00</td>
      <td>2.500000e+01</td>
    </tr>
    <tr>
      <th>loans_org_count_behavior</th>
      <td>4457.0</td>
      <td>1.284541e+01</td>
      <td>7.448393e+00</td>
      <td>1.000000e+00</td>
      <td>7.000000e+00</td>
      <td>1.200000e+01</td>
      <td>1.700000e+01</td>
      <td>4.100000e+01</td>
    </tr>
    <tr>
      <th>consfin_org_count_behavior</th>
      <td>4457.0</td>
      <td>4.732331e+00</td>
      <td>2.974596e+00</td>
      <td>0.000000e+00</td>
      <td>2.000000e+00</td>
      <td>4.000000e+00</td>
      <td>7.000000e+00</td>
      <td>1.800000e+01</td>
    </tr>
    <tr>
      <th>loans_cash_count</th>
      <td>4457.0</td>
      <td>8.113081e+00</td>
      <td>5.374465e+00</td>
      <td>0.000000e+00</td>
      <td>4.000000e+00</td>
      <td>7.000000e+00</td>
      <td>1.100000e+01</td>
      <td>3.100000e+01</td>
    </tr>
    <tr>
      <th>latest_one_month_loan</th>
      <td>4457.0</td>
      <td>9.658963e-01</td>
      <td>1.495566e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>1.000000e+00</td>
      <td>1.500000e+01</td>
    </tr>
    <tr>
      <th>latest_three_month_loan</th>
      <td>4457.0</td>
      <td>2.821853e+00</td>
      <td>3.455817e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>2.000000e+00</td>
      <td>4.000000e+00</td>
      <td>5.200000e+01</td>
    </tr>
    <tr>
      <th>latest_six_month_loan</th>
      <td>4457.0</td>
      <td>1.392686e+01</td>
      <td>1.082847e+01</td>
      <td>0.000000e+00</td>
      <td>6.000000e+00</td>
      <td>1.100000e+01</td>
      <td>2.000000e+01</td>
      <td>7.400000e+01</td>
    </tr>
    <tr>
      <th>history_suc_fee</th>
      <td>4457.0</td>
      <td>4.314561e+01</td>
      <td>3.035362e+01</td>
      <td>0.000000e+00</td>
      <td>2.100000e+01</td>
      <td>3.700000e+01</td>
      <td>5.900000e+01</td>
      <td>2.540000e+02</td>
    </tr>
    <tr>
      <th>history_fail_fee</th>
      <td>4457.0</td>
      <td>1.770855e+01</td>
      <td>2.508935e+01</td>
      <td>0.000000e+00</td>
      <td>3.000000e+00</td>
      <td>1.000000e+01</td>
      <td>2.200000e+01</td>
      <td>3.450000e+02</td>
    </tr>
    <tr>
      <th>latest_one_month_suc</th>
      <td>4457.0</td>
      <td>1.224366e+00</td>
      <td>1.944912e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>2.000000e+00</td>
      <td>2.000000e+01</td>
    </tr>
    <tr>
      <th>latest_one_month_fail</th>
      <td>4457.0</td>
      <td>1.311420e+00</td>
      <td>3.893607e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>1.000000e+00</td>
      <td>5.800000e+01</td>
    </tr>
    <tr>
      <th>loans_long_time</th>
      <td>4457.0</td>
      <td>3.351600e+02</td>
      <td>3.577010e+01</td>
      <td>2.600000e+01</td>
      <td>3.290000e+02</td>
      <td>3.490000e+02</td>
      <td>3.560000e+02</td>
      <td>3.600000e+02</td>
    </tr>
    <tr>
      <th>loans_credit_limit</th>
      <td>4457.0</td>
      <td>2.089298e+03</td>
      <td>7.089514e+02</td>
      <td>0.000000e+00</td>
      <td>1.700000e+03</td>
      <td>2.100000e+03</td>
      <td>2.400000e+03</td>
      <td>6.900000e+03</td>
    </tr>
    <tr>
      <th>loans_credibility_limit</th>
      <td>4457.0</td>
      <td>7.199237e+01</td>
      <td>1.085193e+01</td>
      <td>0.000000e+00</td>
      <td>7.200000e+01</td>
      <td>7.400000e+01</td>
      <td>7.500000e+01</td>
      <td>8.900000e+01</td>
    </tr>
    <tr>
      <th>loans_org_count_current</th>
      <td>4457.0</td>
      <td>8.113081e+00</td>
      <td>5.374465e+00</td>
      <td>0.000000e+00</td>
      <td>4.000000e+00</td>
      <td>7.000000e+00</td>
      <td>1.100000e+01</td>
      <td>3.100000e+01</td>
    </tr>
    <tr>
      <th>loans_product_count</th>
      <td>4457.0</td>
      <td>8.685214e+00</td>
      <td>5.759025e+00</td>
      <td>0.000000e+00</td>
      <td>4.000000e+00</td>
      <td>8.000000e+00</td>
      <td>1.200000e+01</td>
      <td>3.200000e+01</td>
    </tr>
    <tr>
      <th>loans_max_limit</th>
      <td>4457.0</td>
      <td>3.390038e+03</td>
      <td>1.474207e+03</td>
      <td>0.000000e+00</td>
      <td>2.300000e+03</td>
      <td>3.100000e+03</td>
      <td>4.300000e+03</td>
      <td>1.000000e+04</td>
    </tr>
    <tr>
      <th>loans_avg_limit</th>
      <td>4457.0</td>
      <td>1.820358e+03</td>
      <td>5.834183e+02</td>
      <td>0.000000e+00</td>
      <td>1.535000e+03</td>
      <td>1.810000e+03</td>
      <td>2.100000e+03</td>
      <td>6.900000e+03</td>
    </tr>
    <tr>
      <th>consfin_credit_limit</th>
      <td>4457.0</td>
      <td>9.187009e+03</td>
      <td>7.371257e+03</td>
      <td>0.000000e+00</td>
      <td>4.800000e+03</td>
      <td>7.700000e+03</td>
      <td>1.170000e+04</td>
      <td>8.710000e+04</td>
    </tr>
    <tr>
      <th>consfin_credibility</th>
      <td>4457.0</td>
      <td>7.604263e+01</td>
      <td>1.453682e+01</td>
      <td>0.000000e+00</td>
      <td>7.700000e+01</td>
      <td>7.900000e+01</td>
      <td>8.000000e+01</td>
      <td>8.700000e+01</td>
    </tr>
    <tr>
      <th>consfin_org_count_current</th>
      <td>4457.0</td>
      <td>4.732331e+00</td>
      <td>2.974596e+00</td>
      <td>0.000000e+00</td>
      <td>2.000000e+00</td>
      <td>4.000000e+00</td>
      <td>7.000000e+00</td>
      <td>1.800000e+01</td>
    </tr>
    <tr>
      <th>consfin_product_count</th>
      <td>4457.0</td>
      <td>5.227507e+00</td>
      <td>3.409292e+00</td>
      <td>0.000000e+00</td>
      <td>3.000000e+00</td>
      <td>5.000000e+00</td>
      <td>7.000000e+00</td>
      <td>2.000000e+01</td>
    </tr>
    <tr>
      <th>consfin_max_limit</th>
      <td>4457.0</td>
      <td>1.615369e+04</td>
      <td>1.430104e+04</td>
      <td>0.000000e+00</td>
      <td>7.800000e+03</td>
      <td>1.380000e+04</td>
      <td>2.040000e+04</td>
      <td>2.664000e+05</td>
    </tr>
    <tr>
      <th>consfin_avg_limit</th>
      <td>4457.0</td>
      <td>8.007697e+03</td>
      <td>5.679419e+03</td>
      <td>0.000000e+00</td>
      <td>4.737000e+03</td>
      <td>7.050000e+03</td>
      <td>1.000000e+04</td>
      <td>8.280000e+04</td>
    </tr>
    <tr>
      <th>latest_query_day</th>
      <td>4450.0</td>
      <td>2.411281e+01</td>
      <td>3.772572e+01</td>
      <td>-2.000000e+00</td>
      <td>5.000000e+00</td>
      <td>1.400000e+01</td>
      <td>2.400000e+01</td>
      <td>3.600000e+02</td>
    </tr>
    <tr>
      <th>loans_latest_day</th>
      <td>4457.0</td>
      <td>5.518151e+01</td>
      <td>5.348641e+01</td>
      <td>-2.000000e+00</td>
      <td>1.000000e+01</td>
      <td>3.600000e+01</td>
      <td>9.100000e+01</td>
      <td>3.230000e+02</td>
    </tr>
  </tbody>
</table>
<p>83 rows × 8 columns</p>
</div>




```python
df_data.describe()['student_feature']
```




    count    1756.000000
    mean        1.001139
    std         0.033739
    min         1.000000
    25%         1.000000
    50%         1.000000
    75%         1.000000
    max         2.000000
    Name: student_feature, dtype: float64




```python
df_data.isnull().sum().sort_values(ascending=False)  #查看缺失值的列
```




    student_feature                               2998
    cross_consume_count_last_1_month               426
    query_org_count                                304
    query_cash_count                               304
    latest_six_month_apply                         304
    latest_three_month_apply                       304
    latest_query_time                              304
    query_sum_count                                304
    latest_one_month_apply                         304
    query_finance_count                            304
    apply_credibility                              304
    apply_score                                    304
    latest_query_day                               304
    latest_one_month_loan                          297
    loans_score                                    297
    loans_credibility_behavior                     297
    loans_count                                    297
    loans_settle_count                             297
    loans_overdue_count                            297
    loans_org_count_behavior                       297
    consfin_org_count_behavior                     297
    loans_cash_count                               297
    loans_latest_day                               297
    latest_three_month_loan                        297
    loans_product_count                            297
    latest_six_month_loan                          297
    consfin_product_count                          297
    consfin_org_count_current                      297
    consfin_max_limit                              297
    consfin_avg_limit                              297
                                                  ... 
    rank_trad_1_month                                2
    trans_days_interval                              2
    trans_day_last_12_month                          2
    transd_mcc                                       2
    number_of_trans_from_2011                        2
    first_transaction_time                           2
    trans_activity_day                               2
    historical_trans_day                             2
    trans_activity_month                             2
    low_volume_percent                               2
    avg_consume_less_12_valid_month                  2
    top_trans_count_last_1_month                     2
    first_transaction_day                            2
    reg_preference_for_trad                          2
    regional_mobility                                2
    take_amount_in_later_12_month_highest            0
    trade_no                                         0
    bank_card_no                                     0
    source                                           0
    repayment_capability                             0
    is_high_user                                     0
    historical_trans_amount                          0
    trans_amount_3_month                             0
    abs                                              0
    avg_price_last_12_month                          0
    max_cumulative_consume_later_1_month             0
    pawns_auctions_trusts_consume_last_1_month       0
    pawns_auctions_trusts_consume_last_6_month       0
    status                                           0
    custid                                           0
    Length: 89, dtype: int64




```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```
