import pandas as pd
import numpy as np

def calculate_n_day_retention(df, n_days=7):
    """
    计算 N 日留存率。
    输入 df 包含: user_id, login_date (datetime类型)
    """
    if df.empty:
        return 0.0
    
    # 1. 预处理：去重（同一个用户同一天多次登录只计一次）
    df = df[['user_id', 'login_date']].drop_duplicates()
    
    # 2. 获取每个用户的首次登录日期 (Cohort)
    first_login = df.groupby('user_id')['login_date'].min().reset_index()
    first_login.columns = ['user_id', 'first_date']
    
    # 3. 合并回原表
    merged = pd.merge(df, first_login, on='user_id')
    
    # 4. 计算登录日期与首日的时间差
    merged['diff'] = (merged['login_date'] - merged['first_date']).dt.days
    
    # 5. 统计初始用户数 (Day 0 的总人数)
    total_users = first_login['user_id'].nunique()
    
    # 6. 统计第 N 天依然活跃的用户数
    retained_users = merged[merged['diff'] == n_days]['user_id'].nunique()
    
    return retained_users / total_users if total_users > 0 else 0.0