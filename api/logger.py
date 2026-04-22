# logger.py

import json
import os
from datetime import datetime

LOG_FILE = "predictions_log.json"

def log_event(data):
    entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "data": data
    }

    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, "w") as f:
            json.dump([entry], f, indent=4)
    else:
        with open(LOG_FILE, "r+") as f:
            logs = json.load(f)
            logs.append(entry)
            f.seek(0)
            json.dump(logs, f, indent=4)