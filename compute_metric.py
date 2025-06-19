import csv
import numpy as np

log_file = 'log_metrics.csv'

time_fields = ['total_time', 'detect_time', 'antispoof_time', 'recognize_time']
inference_fields = ['detect_time', 'antispoof_time', 'recognize_time']

data = []
with open(log_file, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
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
    else:
        print("No valid data.")
    print()

combined_times = []
for d in data:
    times = [d[k] for k in inference_fields if d[k] is not None]
    if len(times) == len(inference_fields):
        combined_times.append(sum(times))

if combined_times:
    avg_combined = np.mean(combined_times)
    print(f"--- FPS (from avg of detect + antispoof + recognize) ---")
    print(f"Avg Inference Time: {avg_combined:.4f} s")
    print(f"FPS: {1.0 / avg_combined:.2f}")
else:
    print("No valid inference data to calculate FPS.")
