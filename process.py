import pandas as pd

def process(file_path):
    
    #Load Data
    raw_data = pd.read_csv(open(file_path))

    time_data = raw_data[['frame_no', 'timestamp']]
    puck_data = raw_data[['rx-green', 'ry-green']]

    # ms to s
    time_data['timestamp'] = time_data['timestamp'] / 1000
    puck_data['rx-green'] /= 100
    puck_data['ry-green'] /= 100

    # Finite Differance
    deltaTime = time_data['timestamp'].diff()
    puck_data['vx-green'] = puck_data['rx-green'].diff() / deltaTime
    puck_data['vy-green'] = puck_data['ry-green'].diff() / deltaTime
    puck_data['ax-green'] = puck_data['vx-green'].diff() / deltaTime
    puck_data['ay-green'] = puck_data['vy-green'].diff() / deltaTime
    time_data['delta'] = deltaTime

    # Splits test runs: (initial velocity is 0 and it falls off the screen at the end)
    capturing = False
    last_zero = 0
    tests = []
    times = []

    for i, row in puck_data.iterrows():
        ay_green = row['ay-green']
        if capturing:
            if ay_green == 0:
                last_zero = i
            elif pd.isna(ay_green):
                print(f"Captured {last_zero} to {i}")
                tests.append(puck_data[last_zero:i-2])
                times.append(time_data[last_zero:i-2])
                capturing = False
        elif ay_green == 0:
            last_zero = i
            capturing = True
            
    # Output results
    results = []
    for i in range(len(tests)):
        results.append(times[i].join(tests[i]))
    return results
        