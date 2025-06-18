import csv
import numpy as np

log_file = 'log_metrics.csv'

# Chỉ các trường thời gian
time_fields = ['total_time', 'detect_time', 'antispoof_time', 'recognize_time']

data = []
with open(log_file, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        # chỉ lấy những dòng có label là REAL hoặc FAKE
        if row['label'] in ['REAL', 'FAKE']:
            d = {}
            for k in time_fields:
                try:
                    d[k] = float(row[k]) if row[k] not in [None, ''] else None
                except:
                    d[k] = None
            data.append(d)

def stat(lst):
    arr = [x for x in lst if x is not None]
    if not arr:
        return None, None, None
    return np.mean(arr), np.max(arr), np.min(arr)

for field in time_fields:
    values = [d[field] for d in data]
    avg_value, max_value, min_value = stat(values)
    print(f"--- {field} ---")
    if avg_value is not None:
        print(f"Avg {field}: {avg_value:.4f} s")
        print(f"Max {field}: {max_value:.4f} s")
        print(f"Min {field}: {min_value:.4f} s")
        print(f"FPS from avg {field}: {1.0/avg_value:.2f}")
    else:
        print("No valid data.")
    print()
