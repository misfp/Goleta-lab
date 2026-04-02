import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytest
from Pandas import calculate_n_day_retention


def create_test_data(users_data):
    """
    辅助函数：从用户数据字典创建测试 DataFrame
    
    参数:
        users_data: dict {user_id: [login_dates]}
    """
    data = []
    for user_id, dates in users_data.items():
        for date in dates:
            data.append({
                'user_id': user_id,
                'login_date': pd.to_datetime(date),
                'action_type': 'login'
            })
    return pd.DataFrame(data)


class TestCalculateNDayRetention:
    """测试 calculate_n_day_retention 函数"""

    def test_empty_dataframe(self):
        """测试空 DataFrame 返回 0"""
        df = pd.DataFrame(columns=['user_id', 'login_date', 'action_type'])
        assert calculate_n_day_retention(df) == 0.0

    def test_single_user_no_retention(self):
        """测试单个用户只有首日登录，7日留存为 0"""
        users_data = {
            1: ['2024-01-01']
        }
        df = create_test_data(users_data)
        assert calculate_n_day_retention(df, n_days=7) == 0.0

    def test_single_user_with_retention(self):
        """测试单个用户在第 7 天登录，留存率为 1.0"""
        users_data = {
            1: ['2024-01-01', '2024-01-08']
        }
        df = create_test_data(users_data)
        assert calculate_n_day_retention(df, n_days=7) == 1.0

    def test_multiple_users_mixed_retention(self):
        """测试多个用户混合情况"""
        users_data = {
            1: ['2024-01-01', '2024-01-08'],
            2: ['2024-01-01', '2024-01-02', '2024-01-03'],
            3: ['2024-01-01', '2024-01-08', '2024-01-09'],
            4: ['2024-01-01']
        }
        df = create_test_data(users_data)
        assert calculate_n_day_retention(df, n_days=7) == 0.5

    def test_duplicate_dates_same_user(self):
        """测试单个用户同一天有多次登录记录（重复日期），应该只计一次"""
        users_data = {
            1: ['2024-01-01', '2024-01-01', '2024-01-01', '2024-01-08', '2024-01-08']
        }
        df = create_test_data(users_data)
        assert calculate_n_day_retention(df, n_days=7) == 1.0

    def test_non_consecutive_dates(self):
        """测试非连续日期记录，用户在中间几天没登录，但第 7 天登录了"""
        users_data = {
            1: ['2024-01-01', '2024-01-03', '2024-01-08'],
            2: ['2024-01-01', '2024-01-05', '2024-01-06', '2024-01-08']
        }
        df = create_test_data(users_data)
        assert calculate_n_day_retention(df, n_days=7) == 1.0

    def test_user_logs_in_before_day_n(self):
        """测试用户在第 N 天之前登录了，但第 N 天没登录，不计入留存"""
        users_data = {
            1: ['2024-01-01', '2024-01-07'],
            2: ['2024-01-01', '2024-01-08']
        }
        df = create_test_data(users_data)
        assert calculate_n_day_retention(df, n_days=7) == 0.5

    def test_user_logs_in_after_day_n(self):
        """测试用户在第 N 天之后才登录，不计入留存"""
        users_data = {
            1: ['2024-01-01', '2024-01-09'],
            2: ['2024-01-01', '2024-01-08']
        }
        df = create_test_data(users_data)
        assert calculate_n_day_retention(df, n_days=7) == 0.5

    def test_users_with_different_cohort_dates(self):
        """测试多个用户有不同的首登日期（Cohort 不同）"""
        users_data = {
            1: ['2024-01-01', '2024-01-08'],
            2: ['2024-01-02', '2024-01-09'],
            3: ['2024-01-01', '2024-01-02'],
            4: ['2024-01-02']
        }
        df = create_test_data(users_data)
        assert calculate_n_day_retention(df, n_days=7) == 0.5

    def test_n_day_retention_3_day(self):
        """测试 3 日留存率计算"""
        users_data = {
            1: ['2024-01-01', '2024-01-04'],
            2: ['2024-01-01', '2024-01-02', '2024-01-03'],
            3: ['2024-01-01', '2024-01-04']
        }
        df = create_test_data(users_data)
        assert calculate_n_day_retention(df, n_days=3) == pytest.approx(0.666666666)

    def test_complex_scenario(self):
        """复杂场景：包含重复日期、非连续日期、不同首登日期、混合留存情况"""
        users_data = {
            1: ['2024-01-01', '2024-01-01', '2024-01-08', '2024-01-08'],
            2: ['2024-01-01', '2024-01-03', '2024-01-05', '2024-01-09'],
            3: ['2024-01-02', '2024-01-09'],
            4: ['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04', '2024-01-05', '2024-01-06', '2024-01-07', '2024-01-08'],
            5: ['2024-01-02', '2024-01-03'],
            6: ['2024-01-01']
        }
        df = create_test_data(users_data)
        assert calculate_n_day_retention(df, n_days=7) == pytest.approx(0.333333333)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
