import pandas as pd
import sqlite3
import numpy as np
from scipy.spatial.distance import cdist

# ================= 配置路径 =================
DB_PATH = 'switrs.sqlite'  
SENSOR_FILE = 'pems/graph_sensor_locations_bay.csv'  
OUTPUT_FILE = 'pems/pems_bay_incidents_2017.csv'  

# ================= 1. 读取传感器数据并确立边界 =================
print("正在读取传感器位置...")
sensors = pd.read_csv(SENSOR_FILE, header=None)
sensors.columns = ['sensor_id', 'lat', 'lng']  
print("传感器数据预览：")
print(sensors.head())
min_lat, max_lat = sensors['lat'].min() - 0.05, sensors['lat'].max() + 0.05
min_lng, max_lng = sensors['lng'].min() - 0.05, sensors['lng'].max() + 0.05
print(f"筛选范围: Lat [{min_lat:.4f}, {max_lat:.4f}], Lng [{min_lng:.4f}, {max_lng:.4f}]")

# ================= 2. 连接 SQLite 并查询 =================
print("正在连接数据库并执行 SQL 查询...")
conn = sqlite3.connect(DB_PATH)

query = f"""
SELECT 
    case_id, 
    collision_date, 
    collision_time, 
    latitude, 
    longitude, 
    collision_severity,
    pcf_violation_category -- 事故原因，可选
FROM collisions 
WHERE collision_date BETWEEN '2017-01-01' AND '2017-05-31'
  AND latitude BETWEEN {min_lat} AND {max_lat}
  AND longitude BETWEEN {min_lng} AND {max_lng}
  AND latitude IS NOT NULL 
  AND longitude IS NOT NULL
"""

incidents = pd.read_sql_query(query, conn)
conn.close()
print(f"初步筛选出 {len(incidents)} 条事故记录。")

# ================= 3. 数据清洗与最近邻匹配 (Mapping) =================
incidents['timestamp'] = pd.to_datetime(incidents['collision_date'] + ' ' + incidents['collision_time'], errors='coerce')
incidents = incidents[(incidents['latitude'] != 0) & (incidents['longitude'] != 0)]
incident_coords = incidents[['latitude', 'longitude']].values
sensor_coords = sensors[['lat', 'lng']].values

dists = cdist(incident_coords, sensor_coords, metric='euclidean')
closest_sensor_indices = np.argmin(dists, axis=1)
min_distances = np.min(dists, axis=1)
incidents['sensor_id'] = sensors.iloc[closest_sensor_indices]['sensor_id'].values
incidents['dist_to_sensor'] = min_distances
threshold = 0.02
final_incidents = incidents[incidents['dist_to_sensor'] < threshold].copy()

# ================= 4. 处理严重程度 (与论文对齐) =================
def map_severity(val):
    if val in [1, 2]:
        return 'Severe' 
    else:
        return 'Minor'  
final_incidents['paper_type'] = final_incidents['collision_severity'].apply(map_severity)

# ================= 5. 保存结果 =================
final_incidents.to_csv(OUTPUT_FILE, index=False)
print(f"处理完成！最终数据集包含 {len(final_incidents)} 条记录。")
print(f"结果已保存至: {OUTPUT_FILE}")