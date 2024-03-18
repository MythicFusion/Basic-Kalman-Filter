import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import glob

from process import process
from kalman import filter

# Collect all files
test_files = glob.glob('Data\*')
test_data = {}

# Store data by file names
for test_file in test_files:
    test_data[test_file[-6:-4]] = process(test_file)

results = pd.DataFrame(columns=list(test_data.keys())) # DataFrame to store results for later

# Chains relevant information between tests
for key, tests in test_data.items():
    
    # Carry over values
    xState = None
    yState = None
    xCov = None
    yCov = None
    
    for test in tests:
        if len(test) == 0: continue
        estimate, uncertanty, xState, yState, xCov, yCov = filter(test, xState, yState, xCov, yCov)
        
        result = estimate.loc[(len(estimate)-1, 'ax-green')]
        if not pd.isna(result): 
            results.loc['X Acl', key] = result
            results.loc['X Unc', key] = np.sqrt(uncertanty.loc[(len(estimate)-1, 'ax-green')])
            results.loc['Y Acl', key] = estimate.loc[(len(estimate)-1, 'ay-green')]
            results.loc['Y Unc', key] = np.sqrt(uncertanty.loc[(len(estimate)-1, 'ay-green')])

# Data Post Processing to find Kinetic Friction and keep Uncertanty Propagation (MESSY ˙◠˙)
gravity = 9.79284
for col in results.columns:
    xAcl = results.loc['X Acl', col]
    yAcl = results.loc['Y Acl', col]
    xUnc = results.loc['X Unc', col]
    yUnc = results.loc['Y Unc', col]
    
    accel = np.sqrt(xAcl**2 + yAcl**2)
    aUnc = np.sqrt((xAcl**2 * xUnc**2 + yAcl**2 * yUnc**2)/(xAcl**2 + yAcl**2))
    
    angle = np.arctan(yAcl / -xAcl)
    tUnc = np.sqrt((xAcl**2 * xUnc**2 + yAcl**2 * yUnc**2)/(xAcl**2 + yAcl**2)**2)
    
    expected = gravity * np.sin(angle)
    eUnc = tUnc * np.sqrt(gravity**2 * np.cos(angle)**2)
    
    results.loc['Accel', col] = accel
    results.loc['Acl Unc', col] = aUnc
    results.loc['Angle', col] = angle
    results.loc['Ang Unc', col] = tUnc
    results.loc['Expect', col] = expected
    results.loc['E Unc', col] = eUnc
    results.loc['Kinetic', col] = (expected - accel) / (gravity * np.cos(angle))
    results.loc['K Unc', col] = np.sqrt((tUnc**2 * (accel - expected)**2 * np.sin(angle)**2 + (aUnc**2 + eUnc**2)*np.cos(angle)**2) / (gravity**2 * np.cos(angle)**4))

results = results.T
print(results)    
results.to_csv('results.csv')
