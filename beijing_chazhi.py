import pandas as pd
from tqdm import tqdm

# 读取宽表数据
df = pd.read_csv('beijing_pm25.csv')

# 确保 utc_time 是 datetime 类型并设为索引
df['utc_time'] = pd.to_datetime(df['utc_time'])
df.set_index('utc_time', inplace=True)

# 记录原始空值情况
nan_mask = df.isna()

# 对每列（每个站点）插值：仅对缺失长度 <= 5 的间隙进行线性插值
for col in tqdm(df.columns, desc="Interpolating stations"):
    is_na = df[col].isna()
    group = (~is_na).cumsum()
    gap_lengths = is_na.groupby(group).sum()
    # 找到哪些 group 是小于等于5的缺失段
    fillable_groups = gap_lengths[gap_lengths <= 5].index
    mask = is_na & group.isin(fillable_groups)
    # 插值（只对 mask 中 True 的值插）
    df.loc[mask, col] = df[col].interpolate(limit=5, limit_direction='both')[mask]

# 重新检查还有哪些时间点有缺失（长缺失）
rows_with_any_nan = df[df.isna().any(axis=1)].copy()

# 标记这些行中缺失是否连续超过5小时
rows_with_any_nan['gap_group'] = (rows_with_any_nan.index.to_series().diff() != pd.Timedelta(hours=1)).cumsum()

# 找出 gap_group 中连续空缺 > 5 小时的 group
gap_sizes = rows_with_any_nan.groupby('gap_group').size()
long_gap_groups = gap_sizes[gap_sizes > 5].index

# 获取这些 group 的时间索引
to_drop_times = rows_with_any_nan[rows_with_any_nan['gap_group'].isin(long_gap_groups)].index

# 从原数据中删除这些时间点
df_cleaned = df.drop(index=to_drop_times)

# 重置索引并保存
df_cleaned.reset_index(inplace=True)
df_cleaned.to_csv('beijing_pm25_interpolated.csv', index=False)
