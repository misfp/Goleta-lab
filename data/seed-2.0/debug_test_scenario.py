import pandas as pd
from datetime import datetime

# 复杂场景的测试数据
users_data = {
    1: ['2024-01-01', '2024-01-01', '2024-01-08', '2024-01-08'],
    2: ['2024-01-01', '2024-01-03', '2024-01-05', '2024-01-09'],
    3: ['2024-01-02', '2024-01-09'],
    4: ['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04', '2024-01-05', '2024-01-06', '2024-01-07', '2024-01-08'],
    5: ['2024-01-02', '2024-01-03'],
    6: ['2024-01-01']
}

# 创建 DataFrame
data = []
for user_id, dates in users_data.items():
    for date in dates:
        data.append({
            'user_id': user_id,
            'login_date': pd.to_datetime(date),
            'action_type': 'login'
        })
df = pd.DataFrame(data)

print("=" * 70)
print("原始数据（去重后）：")
print("=" * 70)
df_unique = df[['user_id', 'login_date']].drop_duplicates()
print(df_unique)

print("\n" + "=" * 70)
print("每个用户的首登日期：")
print("=" * 70)
first_login = df_unique.groupby('user_id')['login_date'].min().reset_index()
first_login.columns = ['user_id', 'first_date']
print(first_login)

print("\n" + "=" * 70)
print("按 Cohort 分组分析：")
print("=" * 70)
merged = pd.merge(df_unique, first_login, on='user_id')
merged['diff'] = (merged['login_date'] - merged['first_date']).dt.days

# 按 cohort 分组
for cohort_date in first_login['first_date'].unique():
    print(f"\n--- Cohort: {cohort_date.date()} ---")
    cohort_users = first_login[first_login['first_date'] == cohort_date]['user_id'].unique()
    print(f"该 cohort 用户: {list(cohort_users)}")
    print(f"该 cohort 总用户数: {len(cohort_users)}")
    
    cohort_data = merged[merged['user_id'].isin(cohort_users)]
    retained = cohort_data[cohort_data['diff'] == 7]['user_id'].nunique()
    print(f"7 日留存用户数: {retained}")
    print(f"7 日留存率: {retained/len(cohort_users):.4f}")

print("\n" + "=" * 70)
print("不同计算方式的结果：")
print("=" * 70)

# 方式 1：所有用户混在一起计算（原代码）
total_all = first_login['user_id'].nunique()
retained_all = merged[merged['diff'] == 7]['user_id'].nunique()
print(f"方式 1（所有用户混合）: {retained_all}/{total_all} = {retained_all/total_all:.4f}")

# 方式 2：只计算第一个 cohort
first_cohort_date = first_login['first_date'].min()
first_cohort_users = first_login[first_login['first_date'] == first_cohort_date]['user_id'].unique()
first_cohort_data = merged[merged['user_id'].isin(first_cohort_users)]
retained_first = first_cohort_data[first_cohort_data['diff'] == 7]['user_id'].nunique()
print(f"方式 2（只第一个 cohort）: {retained_first}/{len(first_cohort_users)} = {retained_first/len(first_cohort_users):.4f}")

# 方式 3：第一个 cohort 留存 / 所有用户
print(f"方式 3（第一个 cohort 留存 / 所有用户）: {retained_first}/{total_all} = {retained_first/total_all:.4f}")

# 方式 4：所有 cohort 加权平均
cohort_stats = []
for cohort_date in first_login['first_date'].unique():
    cohort_users = first_login[first_login['first_date'] == cohort_date]['user_id'].unique()
    cohort_data = merged[merged['user_id'].isin(cohort_users)]
    retained = cohort_data[cohort_data['diff'] == 7]['user_id'].nunique()
    cohort_stats.append({'users': len(cohort_users), 'retained': retained, 'rate': retained/len(cohort_users)})

weighted_avg = sum([s['retained'] for s in cohort_stats]) / sum([s['users'] for s in cohort_stats])
print(f"方式 4（所有 cohort 加权平均）: {weighted_avg:.4f}")

# 方式 5：用户 4 不算留存（因为中间登录了）
print("\n假设：用户 4 不算留存（因为中间几天也登录了）")
retained_all_no_user4 = retained_all - (1 if 4 in merged[merged['diff'] == 7]['user_id'].values else 0)
print(f"方式 5（排除用户 4）: {retained_all_no_user4}/{total_all} = {retained_all_no_user4/total_all:.4f}")
