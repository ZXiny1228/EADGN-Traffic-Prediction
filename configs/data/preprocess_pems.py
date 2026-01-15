import pandas as pd
import numpy as np
import math
import os
from datetime import datetime, timedelta

# ================= 配置区域 =================
DATA_DIR = 'configs/data/pems'
SENSOR_FILE = os.path.join(DATA_DIR, 'graph_sensor_locations_bay.csv')
INCIDENT_FILE = os.path.join(DATA_DIR, 'pems_bay_incidents_2017.csv')
START_TIME = "2017-01-01 00:00:00"
END_TIME = "2017-05-31 23:55:00"
TIME_INTERVAL_MIN = 5
NUM_NODES = 325
TOTAL_STEPS = 52116  
OUTPUT_EQ4 = os.path.join(DATA_DIR, 'PEMS_events_eq4.csv')  
OUTPUT_BASELINE = os.path.join(DATA_DIR, 'PEMS_events_baseline.csv')  

# ================= 辅助函数 =================

def haversine(lat1, lon1, lat2, lon2):
    """计算两点间地理距离 (km)"""
    R = 6371
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))

def get_severity_score(severity_str):
    """将文本严重程度映射为数值"""
    severity_str = str(severity_str).lower()
    if 'fatal' in severity_str: return 1.0  
    if 'severe' in severity_str: return 0.8
    if 'major' in severity_str: return 0.6
    if 'minor' in severity_str: return 0.3
    return 0.2  

def generate_event_matrix(use_eq4=True):
    print(f"Loading data... (Mode: {'EADGN Eq.4' if use_eq4 else 'Baseline Nearest'})")

    # 1. 读取传感器位置
    try:
        sensors = pd.read_csv(SENSOR_FILE, header=None, names=['sensor_id', 'lat', 'lon'])
        if isinstance(sensors.iloc[0]['lat'], str):
            sensors = pd.read_csv(SENSOR_FILE)
    except Exception as e:
        print(f"Error reading sensor file: {e}")
        return

    sensor_locs = sensors[['lat', 'lon']].values
    sensor_ids = sensors['sensor_id'].tolist()
    sensor_id_to_ind = {sid: i for i, sid in enumerate(sensor_ids)}

    # 2. 读取事件数据
    incidents = pd.read_csv(INCIDENT_FILE)
    incidents['timestamp'] = pd.to_datetime(incidents['timestamp'], errors='coerce')
    missing_count = incidents['timestamp'].isna().sum()
    if missing_count > 0:
        print(f"Warning: Found {missing_count} rows with invalid/missing timestamps. Dropping them.")
        incidents.dropna(subset=['timestamp'], inplace=True)
    incidents.reset_index(drop=True, inplace=True)

    # 3. 初始化时间轴
    start_dt = pd.to_datetime(START_TIME)
    event_mx = np.zeros((TOTAL_STEPS, NUM_NODES), dtype=np.float32)
    count_mapped = 0

    # 4. 遍历事件进行映射
    for idx, row in incidents.iterrows():
        try:
            e_time = row['timestamp']
            if pd.isnull(e_time):
                continue

            e_lat = row['latitude']
            e_lon = row['longitude']

            if pd.isnull(e_lat) or pd.isnull(e_lon):
                continue

            severity = get_severity_score(row['collision_severity'])
            time_delta = (e_time - start_dt).total_seconds() / 60.0
            t_idx = int(time_delta // TIME_INTERVAL_MIN)
            if t_idx < 0 or t_idx >= TOTAL_STEPS:
                continue

            target_node_idx = -1

            if not use_eq4:
                provided_sensor_id = row.get('sensor_id')
                if provided_sensor_id in sensor_id_to_ind:
                    target_node_idx = sensor_id_to_ind[provided_sensor_id]
                else:
                    dists = [haversine(e_lat, e_lon, s_lat, s_lon) for s_lat, s_lon in sensor_locs]
                    target_node_idx = np.argmin(dists)
            else:
                w_d = 1.0
                w_theta = 0.5

                scores = []
                for i, (s_lat, s_lon) in enumerate(sensor_locs):
                    dist = haversine(e_lat, e_lon, s_lat, s_lon)
                    alignment_cost = abs((s_lon - e_lon) * 100)
                    final_cost = w_d * dist + w_theta * alignment_cost
                    scores.append(final_cost)

                target_node_idx = np.argmin(scores)

            if target_node_idx != -1:
                duration_steps = 6
                end_step = min(t_idx + duration_steps, TOTAL_STEPS)

                for t in range(t_idx, end_step):
                    event_mx[t, target_node_idx] = max(event_mx[t, target_node_idx], severity)

                count_mapped += 1

        except Exception as e:
            print(f"Skipping row {idx} due to error: {e}")
            continue

    print(f"Processed {count_mapped} incidents.")

    out_path = OUTPUT_EQ4 if use_eq4 else OUTPUT_BASELINE
    df_out = pd.DataFrame(event_mx)
    df_out.columns = [str(sid) for sid in sensor_ids]
    df_out.to_csv(out_path, index=False)
    print(f"Successfully saved event matrix to: {out_path}")

#  ================= 执行 =================
if __name__ == "__main__":
    generate_event_matrix(use_eq4=True)
    generate_event_matrix(use_eq4=False)