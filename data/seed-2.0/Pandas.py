import pandas as pd
import numpy as np

def calculate_n_day_retention(df, n_days=7, mode='first_cohort', cohort_date=None):
    """
    计算 N 日留存率。
    
    参数:
        df: 包含 user_id, login_date (datetime类型) 的 DataFrame
        n_days: 计算几日留存，默认 7
        mode: 计算模式
            - 'first_cohort': 只计算第一个 cohort（最早首登日期）的留存率
            - 'all_cohorts_weighted': 所有 cohort 的加权平均留存率
            - 'all_cohorts_mixed': 所有 cohort 混在一起计算（原代码的方式）
            - 'first_cohort_retained_over_all': 第一个 cohort 留存数 / 所有用户数
        cohort_date: 指定 cohort 日期（首登日期），仅在 mode='first_cohort' 时使用
    
    返回:
        float: N 日留存率
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
    merged['diff'] = (merged['login_date'] - merged['first_date']).dt.days
    
    # 获取第一个 cohort
    first_cohort_date = first_login['first_date'].min()
    first_cohort_users = first_login[first_login['first_date'] == first_cohort_date]['user_id'].unique()
    
    # 计算第一个 cohort 的留存用户数
    first_cohort_merged = merged[merged['user_id'].isin(first_cohort_users)]
    first_cohort_retained = first_cohort_merged[first_cohort_merged['diff'] == n_days]['user_id'].nunique()
    
    if mode == 'first_cohort':
        # 只计算第一个 cohort 的留存率
        total_users = len(first_cohort_users)
        return first_cohort_retained / total_users if total_users > 0 else 0.0
    
    elif mode == 'first_cohort_retained_over_all':
        # 第一个 cohort 留存数 / 所有用户数
        total_users_all = first_login['user_id'].nunique()
        return first_cohort_retained / total_users_all if total_users_all > 0 else 0.0
    
    elif mode == 'all_cohorts_mixed':
        # 所有 cohort 混在一起计算（原代码的方式）
        total_users = first_login['user_id'].nunique()
        retained_users = merged[merged['diff'] == n_days]['user_id'].nunique()
        return retained_users / total_users if total_users > 0 else 0.0
    
    elif mode == 'all_cohorts_weighted':
        # 所有 cohort 的加权平均留存率
        cohort_stats = []
        for cohort_date in first_login['first_date'].unique():
            cohort_users = first_login[first_login['first_date'] == cohort_date]['user_id'].unique()
            cohort_merged = merged[merged['user_id'].isin(cohort_users)]
            retained = cohort_merged[cohort_merged['diff'] == n_days]['user_id'].nunique()
            cohort_stats.append({'users': len(cohort_users), 'retained': retained})
        
        total_retained = sum([s['retained'] for s in cohort_stats])
        total_users = sum([s['users'] for s in cohort_stats])
        return total_retained / total_users if total_users > 0 else 0.0
    
    else:
        raise ValueError(f"Unknown mode: {mode}")

