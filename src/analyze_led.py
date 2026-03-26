import numpy as np
import matplotlib.pyplot as plt
from metavision_core.event_io import EventsIterator


# ----------------------------
# Load raw events
# ----------------------------
# Load your recording
file_path = "led_test2.raw"

# Create iterator
events = EventsIterator(input_path=file_path)

timestamps = []

# Collect timestamps
for evs in events:
    timestamps.extend(evs['t'])

timestamps = np.array(timestamps)

print("Total events:", len(timestamps))


# ----------------------------
# Convert to time signal
# ----------------------------
# Convert microseconds to seconds
time_sec = timestamps * 1e-6

# Make histogram (event count vs time)
bin_width = 0.001  # 1 ms bins
bins = np.arange(time_sec.min(), time_sec.max(), bin_width)

counts, edges = np.histogram(time_sec, bins=bins)


# ----------------------------
# Plot activity
# ----------------------------
plt.plot(edges[:-1], counts)
plt.xlabel("Time (s)")
plt.ylabel("Event Count")
plt.title("Event Activity vs Time")
plt.show()
