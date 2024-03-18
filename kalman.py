import pandas as pd
import numpy as np

# Delta Time can varry so I made a function to get the model for any delta time
def model(dt):
    return np.mat([
        [1, dt, (dt**2)/2],
        [0, 1, dt],
        [0, 0, 1]
    ])

# Initialize values based on X components of data
def xinit(data : pd.DataFrame, pState = None, cov = None):
    
    state = np.mat([
        [data['rx-green'].iloc[0]],
        [data['vx-green'].iloc[0]],
        [data['ax-green'].iloc[0]]
    ])
    if pState is not None: state[2] = pState[2]
    mCov = np.cov(data.filter(like='x-green').transpose().to_numpy())
    if cov is None: cov = mCov
    mVals = np.identity(mCov.shape[0])
    
    return state, cov, mVals, mCov

# Initialize values based on Y components of data
def yinit(data : pd.DataFrame, pState = None, cov = None):
    
    state = np.mat([
        [data['ry-green'].iloc[0]],
        [data['vy-green'].iloc[0]],
        [data['ay-green'].iloc[0]]
    ])
    if pState is not None: state[2] = pState[2]
    mCov = np.cov(data.filter(like='y-green').transpose().to_numpy())
    if cov is None : cov = mCov
    mVals = np.identity(mCov.shape[0])
    
    return state, cov, mVals, mCov
    
# Step the state forward based on current understanding of the system dynamics
def predict(dt, state, cov):
    m = model(dt)
    pState = m * state
    pCov = m * cov * m.T
    return pState, pCov

# Update our current understanding of the system dynamics
def update(pState, pCov, mVals, mCov, new):
    gain = pCov * mVals.T * (mVals * pCov * mVals.T + mCov) ** -1
    state = pState + gain * (mVals * new - mVals * pState)
    cov = (np.identity(pCov.shape[0]) - gain * mVals) * pCov * (np.identity(pCov.shape[0]) - gain * mVals).T + gain * mCov * gain.T
    return state, cov

# Main function of this script. Duplicated for X and Y component sections    
def filter(data : pd.DataFrame, xState = None, yState = None, xCov = None, yCov = None):
    
    state, cov, mVals, mCov = xinit(data, xState, xCov)
    
    xestimate = pd.DataFrame(columns = data.filter(like='x-green').columns)
    xuncertanty = xestimate.copy()
    
    for i in range(data.shape[0]):
        
        new_data = np.mat([
            [data['rx-green'].iloc[i]],
            [data['vx-green'].iloc[i]],
            [data['ax-green'].iloc[i]],
        ])
        
        pState, pCov = predict(data['delta'].iloc[i], state, cov)
        state, cov = update(pState, pCov, mVals, mCov, new_data)
        xestimate.loc[len(xestimate)] = state.T.tolist()[0]
        xuncertanty.loc[len(xuncertanty)] = [cov.item(0), cov.item(4), cov.item(8)]
        
    xState = state.copy()
    xCov = cov.copy()
        
    state, cov, mVals, mCov = yinit(data, yState, yCov)
    
    yestimate = pd.DataFrame(columns = data.filter(like='y-green').columns)
    yuncertanty = yestimate.copy()
    
    for i in range(data.shape[0]):
        
        new_data = np.mat([
            [data['ry-green'].iloc[i]],
            [data['vy-green'].iloc[i]],
            [data['ay-green'].iloc[i]],
        ])
        
        pState, pCov = predict(data['delta'].iloc[i], state, cov)
        state, cov = update(pState, pCov, mVals, mCov, new_data)
        yestimate.loc[len(yestimate)] = state.T.tolist()[0]
        yuncertanty.loc[len(yuncertanty)] = [cov.item(0), cov.item(4), cov.item(8)]
        
    estimate = xestimate.join(yestimate)
    uncertanty = xuncertanty.join(yuncertanty)
    
    return estimate, uncertanty, xState, state, xCov, cov
    