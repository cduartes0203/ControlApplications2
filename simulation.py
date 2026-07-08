# %%
import numpy as np
import pandas as pd
import cvxpy as cp
from scipy.spatial import KDTree
from scipy.interpolate import interp1d
import plotly.graph_objects as go
from scipy.interpolate import CubicSpline
from Functions.Graphs import *
from sklearn.metrics import root_mean_squared_error as RMSE
import optuna
from optuna.samplers import RandomSampler
from optuna.exceptions import TrialPruned
from numba import njit

# %%
path = r'ChargingResistance_SOC.csv' 
df = pd.read_csv(path)
SOC = df.iloc[:,0].values
CR = df.iloc[:,1].values
get_CR = interp1d(SOC, CR, kind='linear', bounds_error=False, fill_value="extrapolate")

path = r'DischargingResistance_SOC.csv' 
df = pd.read_csv(path)
SOC = df.iloc[:,0].values
DR = df.iloc[:,1].values
get_DR = interp1d(SOC, DR, kind='linear', bounds_error=False, fill_value="extrapolate")

path = r'OCV_SOC.csv'
df = pd.read_csv(path)
SOC = df.iloc[:,0].values
OCV = df.iloc[:,1].values
get_OCV = interp1d(SOC, OCV, kind='linear', bounds_error=False, fill_value="extrapolate")

path = r'PMEFC_eta.csv'
df = pd.read_csv(path)
PFC = df.iloc[:,0].values
ETA = df.iloc[:,1].values
get_ETAfc = interp1d(PFC, ETA, kind='linear', bounds_error=False, fill_value="extrapolate")

def GetBatterieParams(soc):
    Rbat = get_DR(soc)
    Uocv = get_OCV(soc)
    return Rbat, Uocv

def GetETAfc(pfc):
    return get_ETAfc(pfc)/100

def GetI_bat(soc,pbat):
    R_bat, U_ocv = GetBatterieParams(soc)
    P_bat_est_watts = (pbat) * 1000.0
    inside_sqrt_est = max(0.01, U_ocv**2 - 4.0 * R_bat * P_bat_est_watts)
    I_bat_est = (U_ocv - np.sqrt(inside_sqrt_est)) / (2.0 * R_bat)
    return I_bat_est

def GetBatterieParams2(soc,pbat):
    Rbat, Uocv = GetBatterieParams(soc)
    InSqrt = max(0.01, Uocv**2 - 4.0 * Rbat * pbat * 1e3)
    Ibat = (Uocv - np.sqrt(InSqrt)) / (2.0 * Rbat)
    UDC = Uocv - Ibat * Rbat
    return Ibat, Rbat, UDC
 
def SelSampler(mode='auto'):    
    if mode == 'auto':
        sampler = None
    elif mode == 'tpe':
        sampler = optuna.samplers.TPESampler(multivariate=True, n_startup_trials=10)
    elif mode == 'random':
        sampler = RandomSampler()
    return sampler

def TractionForce(v,acc):
    cr = 0.0085
    cd = 0.55
    rho = 1.225
    a = 0
    Area = 8.16
    m = 3000
    g = 9.81
    return m*g*cr*np.cos(a) + m*g*np.sin(a) + m*acc + 0.5*rho*Area*cd*(v**2)

def Preq(v, acc):
    Ft= TractionForce(v,acc)
    eta= 1
    return Ft * v / eta

def TurningSpeed(u_control_1, V_cruising, V_turning, k=0.15):
    V_turning = V_turning * 0.9
    angulo_abs = abs(u_control_1)
    speed_drop = (V_cruising - V_turning) * np.exp(-k * angulo_abs)
    return float(V_turning + speed_drop)

def get_linear_model_matrices_all(x_current, u_prev, dt, n_horizon, l=2.5):
    v = np.full(n_horizon, x_current[2])
    psi = np.full(n_horizon, x_current[3])
    a = np.full(n_horizon, u_prev[0])
    delta = np.full(n_horizon, u_prev[1])
    
    beta = np.arctan(0.5 * np.tan(delta))
    cos_psi_beta = np.cos(psi + beta)
    sin_psi_beta = np.sin(psi + beta)
    
    A_list = np.zeros((n_horizon, 4, 4))
    B_list = np.zeros((n_horizon, 4, 2))
    c_list = np.zeros((n_horizon, 4))
    
    dbeta_ddelta = 0.5 * (1 + np.tan(delta)**2) / (1 + (0.5 * np.tan(delta))**2)
    dx_ddelta = -dt * v * sin_psi_beta * dbeta_ddelta
    dy_ddelta = dt * v * cos_psi_beta * dbeta_ddelta
    dpsi_ddelta = dt * (v / l) * np.cos(beta) * dbeta_ddelta
    
    for k in range(n_horizon):
        A_list[k] = [
            [1.0, 0.0, dt * cos_psi_beta[k], -dt * v[k] * sin_psi_beta[k]],
            [0.0, 1.0, dt * sin_psi_beta[k],  dt * v[k] * cos_psi_beta[k]],
            [0.0, 0.0, 1.0,                0.0],
            [0.0, 0.0, dt * np.sin(beta[k])/l, 1.0]
        ]
        B_list[k] = [
            [0.0, dx_ddelta[k]],
            [0.0, dy_ddelta[k]],
            [dt,  0.0],
            [0.0, dpsi_ddelta[k]]
        ]
        
        x_next_nominal = np.array([
            x_current[0] + dt * v[k] * cos_psi_beta[k],
            x_current[1] + dt * v[k] * sin_psi_beta[k],
            v[k] + dt * a[k],
            psi[k] + dt * (v[k] / l) * np.sin(beta[k])
        ])
        c_list[k] = x_next_nominal - A_list[k] @ x_current - B_list[k] @ u_prev
        
    return A_list, B_list, c_list


def SimulateRT(dt=0.1, n_horizon=30, sim_steps=800, track_percentual=1,
               W_X=1, W_Y=1, W_speed=10, W_acc=1.5, W_delta=0.25, W_U0=1, W_U1=2,
               size=1, show=False):
    
    
    V_cruising, V_turning = 16, 6
    acc_max = 2.0
    acc_min = -3
    delta_max = np.deg2rad(30)
    delta_min = -np.deg2rad(30)
    l = 2.5 
    BreakCheck = False



    path = r'DyntheticDataset\RaceTrack5.csv' 
    try:
        df = pd.read_csv(path)
        x_mid = df['x_coords'].values[:] * size
        y_mid = df['y_coords'].values[:] * size
    except FileNotFoundError:
        theta = np.linspace(0, 2*np.pi, 200)
        x_mid = 100 * np.cos(theta) + 100
        y_mid = 100 * np.sin(theta) + 100

    track_points = np.vstack((x_mid, y_mid)).T
    track_tree = KDTree(track_points)

    dx = np.diff(x_mid)
    dy = np.diff(y_mid)
    segment_lengths = np.sqrt(dx**2 + dy**2)
    s_coor = np.insert(np.cumsum(segment_lengths), 0, 0.0)
    track_length = s_coor[-1]

    get_x_at_s = interp1d(s_coor, x_mid, kind='linear', bounds_error=False, fill_value="extrapolate")
    get_y_at_s = interp1d(s_coor, y_mid, kind='linear', bounds_error=False, fill_value="extrapolate")

    

    s_total_traveled = 0.0
    last_current_idx = 0
    x_current = np.array([75, 0, 0.0, 0.0])
    u_prev = np.array([0.0, 0.0])

    t_history, x_history, y_history, v_history = [0.0], [x_current[0]], [x_current[1]], [x_current[2]]
    v_ref_history, acc_history, delta_history, psi_history = [0.0], [0.0], [0.0], [0.0]
    v_horizon = [np.zeros(n_horizon+1)]
    Preq_horizon = [np.zeros(n_horizon)]
    turning_history = [0]

    t_lim = 500 * track_percentual * size
    sim_steps = int(sim_steps * track_percentual * 1.1 * size)
    
    # ==============================================================================
    # --- OPTIMIZED CVXPY PARAMETRIC SETUP (FULLY VECTORIZED PARAMETERS) ---
    # ==============================================================================
    X_cvx = cp.Variable((4, n_horizon + 1))
    U_cvx = cp.Variable((2, n_horizon))
    
    x_init_param = cp.Parameter(4)
    u_prev_param = cp.Parameter(2)
    
    # Combined horizon constraints as stacked parameters 
    # This prevents updating parameters entry-by-entry inside the loop
    x_ref_param = cp.Parameter(n_horizon)
    y_ref_param = cp.Parameter(n_horizon)
    v_ref_param = cp.Parameter(n_horizon)
    W_acc_param = cp.Parameter(nonneg=True)
    
    # Stacking A, B, and c arrays dynamically removes the 1-to-N horizon assignment loops
    A_stacked = cp.Parameter((n_horizon * 4, 4))
    B_stacked = cp.Parameter((n_horizon * 4, 2))
    c_stacked = cp.Parameter((n_horizon * 4))
    
    cost = 0
    constraints = [X_cvx[:, 0] == x_init_param]
    
    for k in range(n_horizon):
        # Extract row slices mapping cleanly across the stacked matrices
        A_k = A_stacked[k*4 : (k+1)*4, :]
        B_k = B_stacked[k*4 : (k+1)*4, :]
        c_k = c_stacked[k*4 : (k+1)*4]
        
        constraints += [X_cvx[:, k+1] == A_k @ X_cvx[:, k] + B_k @ U_cvx[:, k] + c_k]
        
        cost += W_X * cp.square(X_cvx[0, k] - x_ref_param[k])
        cost += W_Y * cp.square(X_cvx[1, k] - y_ref_param[k])
        cost += W_speed * cp.square(X_cvx[2, k] - v_ref_param[k])
        cost += W_acc_param * cp.square(U_cvx[0, k])
        cost += W_delta * cp.square(U_cvx[1, k])
        
        if k == 0:
            cost += W_U0 * cp.square(U_cvx[0, 0] - u_prev_param[0])
            cost += W_U1 * cp.square(U_cvx[1, 0] - u_prev_param[1])
        else:
            cost += W_U0 * cp.square(U_cvx[0, k] - U_cvx[0, k-1])
            cost += W_U1 * cp.square(U_cvx[1, k] - U_cvx[1, k-1])
            
        constraints += [U_cvx[0, k] >= acc_min, U_cvx[0, k] <= acc_max]
        constraints += [U_cvx[1, k] >= delta_min, U_cvx[1, k] <= delta_max]
        constraints += [X_cvx[2, k] >= 0.0, X_cvx[2, k] <= V_cruising]
        
    prob = cp.Problem(cp.Minimize(cost), constraints)
    # ==============================================================================

    for step in range(sim_steps):
        if step % 100 * size == 0 and show: 
            print(f'Step: {step} | Speed: {x_current[2]:.2f} m/s | Distance traveled: {s_total_traveled:.2f} / {track_length*track_percentual:.2f} m')
        _, current_idx = track_tree.query([x_current[0], x_current[1]])
        
        idx_diff = current_idx - last_current_idx
        if idx_diff < -len(x_mid)/2: idx_diff += len(x_mid)
        elif idx_diff > len(x_mid)/2: idx_diff -= len(x_mid)
        if idx_diff > 0:
            s_total_traveled += np.sum(segment_lengths[last_current_idx:current_idx])
        last_current_idx = current_idx

        s_projected = s_coor[current_idx]
        x_ref_horizon = np.zeros(n_horizon)
        y_ref_horizon = np.zeros(n_horizon)
        v_ref_horizon = np.zeros(n_horizon)

        u_prev_deg = np.rad2deg(u_prev[1])
        for k in range(n_horizon):
            s_projected += max(x_current[2], 1.5) * dt 
            s_wrapped = s_projected % track_length
            x_ref_horizon[k] = get_x_at_s(s_wrapped)
            y_ref_horizon[k] = get_y_at_s(s_wrapped)
            
            if s_total_traveled >= track_length * track_percentual:
                v_ref_horizon[k] = 0.0
            else:
                v_ref_horizon[k] = TurningSpeed(u_prev_deg, V_cruising, V_turning, k=0.15)
            
        A_mat, B_mat, c_mat = get_linear_model_matrices_all(x_current, u_prev, dt, n_horizon, l)
        # Update scalar parameters instantly
        W_acc_param.value = 0.0 if s_total_traveled >= track_length * track_percentual else W_acc
        x_init_param.value = x_current
        u_prev_param.value = u_prev
        x_ref_param.value = x_ref_horizon
        y_ref_param.value = y_ref_horizon
        v_ref_param.value = v_ref_horizon

        # --- INSTANT MASS ASSIGNMENT VIA TRANSFORMATION ---
        # Stacking removes the python `for k in range(n_horizon)` setter loop entirely!
        A_stacked.value = A_mat.reshape(n_horizon * 4, 4)
        B_stacked.value = B_mat.reshape(n_horizon * 4, 2)
        c_stacked.value = c_mat.flatten()

        try:
            # Native warm_start dramatically improves MOSEK performance over step updates
            # canon_backend handles problem parsing via SciPy to avoid the _cvxcore error
            prob.solve(
                solver=cp.MOSEK, 
                verbose=False, 
                warm_start=True, 
                canon_backend=cp.SCIPY_CANON_BACKEND
            )
            u_control = U_cvx[:, 0].value
            
            if u_control is None: 
                raise ValueError("Solver returned None")
                
        except Exception:
            u_control = np.array([0.0, u_prev[1]])


        beta_sim = np.arctan(0.5 * np.tan(u_control[1]))
        x_next = x_current[0] + dt * (x_current[2] * np.cos(x_current[3] + beta_sim))
        y_next = x_current[1] + dt * (x_current[2] * np.sin(x_current[3] + beta_sim))
        v_next = x_current[2] + dt * u_control[0]
        psi_next = x_current[3] + dt * ((x_current[2] / l) * np.sin(beta_sim))
        
        turning = abs(u_control[1]) >= np.deg2rad(4.5)
        x_current = np.array([x_next, y_next, v_next, psi_next])
        u_prev = u_control.copy()

        #print(x_current[2], X_cvx[2].value[:4])
        v_h = X_cvx[2].value[1:]
        acc_h = U_cvx[0].value
        #acc_h = np.clip(acc_h,0,np.inf)
        p_horizon = Preq(v_h,acc_h)
        p_horizon = np.clip(p_horizon,0,np.inf)
        for i,_ in enumerate(p_horizon):
            if p_horizon[i] == 0:
                p_horizon[i] = 0
        
        
        t_history.append((step + 1) * dt)
        x_history.append(x_current[0])
        y_history.append(x_current[1])
        v_history.append(x_current[2])
        v_horizon.append(X_cvx.value[2])
        Preq_horizon.append(p_horizon)
        v_ref_history.append(v_ref_horizon[0])
        acc_history.append(u_control[0])
        delta_history.append(np.rad2deg(u_control[1]))
        psi_history.append(np.rad2deg(x_current[3]))
        turning_history.append(1 if turning else 0)
        
        delta_check = np.abs(np.array(delta_history))
        window = int(5/dt)
        Vref_mean = abs(np.mean(np.array(v_ref_history[-window:])))
        V_mean = abs(np.mean(np.array(v_history[-window:])))
        acc_mean = abs(np.mean(np.array(acc_history[-window:])))

        if len(delta_check[delta_check > 26.5]) > 10 :
            BreakCheck = True
            if show: print('Angle Break')
            break
        if s_total_traveled >= track_length * track_percentual * 1.1:
            BreakCheck = True
            if show: print('Length Break')
            break
        elif s_total_traveled >= track_length * track_percentual and x_current[2] < 0.9:
            if show: print('Completed')
            break
        elif t_history[-1] > t_lim:
            BreakCheck = True
            if show: print('Time Break')
            break
        elif t_history[-1] > t_lim and acc_mean < 0.1 and V_mean > 0.9:
            BreakCheck = True
            if show: print('Time and Speed Break')
            break
        elif Vref_mean < 0.01 and acc_mean < 0.1 and x_current[2] > 0.9:
            BreakCheck = True
            if show: print('Vref and Acc Break')
            break
    
    if show: 
        print(f'Time: {t_history[-1]} | Speed: {x_current[2]:.2f} m/s | Distance traveled: {s_total_traveled:.2f} / {track_length*track_percentual:.2f} m')
    t_history = np.array(t_history)
    score = RMSE(np.array(v_ref_history)[t_history < t_lim], np.array(v_history)[t_history < t_lim])
    turning_history.append(1 if turning else 0)


    #df_Vhorizon = pd.DataFrame(v_horizon)
    #df_Vhorizon.iloc[:,0] = t_history
    #names = ['time'] + [f'{i}' for i in range(1,(n_horizon+1))]
    #df_Vhorizon.columns = names

    v_horizon = np.array(v_horizon).T

    df_Vhorizon = pd.DataFrame()
    df_Vhorizon['time'] = t_history
    for i in range(len(v_horizon)):
        df_Vhorizon[f'{i}'] = v_horizon[i]

    return score, BreakCheck, [t_history, v_history, acc_history, delta_history, turning_history, x_history, y_history, x_mid, y_mid, psi_history, v_ref_history, Preq_horizon, df_Vhorizon]

# %%

'''def objective(trial):
    W_X =         trial.suggest_float('W_X', 1, 15,step=0.5)
    W_Y =         trial.suggest_float('W_Y', 1, 15,step=0.5)
    W_speed = trial.suggest_float('W_speed', 0.1, 15,step=0.1)
    W_acc = trial.suggest_float(    'W_acc', 0.1, 15,step=0.1)
    W_delta = trial.suggest_float('W_delta', 0.1, 15,step=0.1)
    W_U0 = trial.suggest_float(      'W_U0', 0.1, 15,step=0.1)
    W_U1 = trial.suggest_float(      'W_U1', 0.1, 15,step=0.1)
    
    score,BreakCheck,_ = SimulateRT(dt=0.25, n_horizon=12, sim_steps=350,track_percentual=0.9,
                                    W_X=W_X, W_Y=W_Y, W_speed=W_speed, W_acc=W_acc, W_delta=W_delta,
                                    W_U0=W_U0, W_U1=W_U1, size=1.5, show=False)
    if BreakCheck:
        raise TrialPruned()
    
    return score

study = optuna.create_study(
    direction='minimize',
    sampler=SelSampler(mode='auto'),
    #pruner=pruner,
    #storage="sqlite:///" + f'Optuna/RT.db', study_name=f'P{0}'
    )

study.optimize(objective, n_trials=500)
best_params = study.best_params
params = list(best_params.values())
print('Erro:', study.best_value, 'parameters: ', params)'''


# %% [markdown]
# Erro: 3.180956239922146 parameters:  [1.0, 1.0, 14.6, 1.25, 10.51, 1.72, 11.87]
# 

# %%
if 'params' in globals(): params = params
else: params = [1.0, 1.0, 14.6, 1.25, 10.51, 1.72, 11.87]
W_X,W_Y,W_speed,W_acc,W_delta,W_U0,W_U1 =  params

score,BreakCheck,data_sim = SimulateRT(dt=0.25, n_horizon=12, sim_steps=1e5,track_percentual=1,
                                    W_X=W_X, W_Y=W_Y, W_speed=W_speed, W_acc=W_acc, W_delta=W_delta,
                                    W_U0=W_U0, W_U1=W_U1, size=1, show=True)
[t_history, v_history, acc_history, delta_history, turning_history, x_history, y_history, x_mid, y_mid, psi_history,
  v_ref_history, Preq_horizon, df_Vhorizon] = data_sim

# %%
ySeries=[v_ref_history, v_history, acc_history, delta_history][:]
xSeries=[t_history for i in range(len(ySeries))][:]
names = ['Reference Speed (m/s)', 'Current Speed (m/s)',
        'Acceleration (m/s²)', 'Stirring Angle (deg)',][:]
data = [x_mid, y_mid, x_history,y_history, t_history, delta_history, psi_history, v_history]

PlotSeriesPLY(xSeries,ySeries,names,title=f'Parameters Over Time|RMSE: {score}')
#PlotSeriesDifScalesPLY(xSeries,ySeries,names,title=f'Parameters Over Time|RMSE: {score}')
PlotMPCTracksPLY(data,width=800,height=500)

# %%
def SimulateEMS(Preq_horizon_raw, dt=1.0, n_horizon=3,
                W_H2=1, W_SoC=1, L_SoC=1, W_FC=1.0):
    # --- Downsample the 0.1s vehicle trajectory data to 1.0s intervals ---
    Preq_ems_input = []
    for step_1s in range(len(Preq_horizon_raw)):
        if step_1s % 4 == 0:
            forecast_vector_30points = Preq_horizon_raw[step_1s]
            forecast_3points_1s = forecast_vector_30points[[3, 7, 11]]
            Preq_ems_input.append(forecast_3points_1s)
            
    Preq_ems_input = np.array(Preq_ems_input) 
    sim_steps = len(Preq_ems_input)

    # --- Power Sources & Vehicle Constants (From Article Table 1) ---
    Q_bat_h = 90.0        # Battery Capacity (Ah)
    Q_bat_s = Q_bat_h * 3600.0 
    eta_dcdc = 0.9      # Unidirectional DC/DC Converter Efficiency
    rho_H2 = 120 * 1e6
    
    SOC_i = 0.6         
    Pfc_i = 0.0        
    x_current = np.array([SOC_i, Pfc_i])
    
    t_history, SOC_history, Pfc_history = [0.0], [x_current[0]], [x_current[1]]
    cur_history = [0.0]
    Preq_history, P_batL_history, P_batU_history = [0.0], [0.0], [0.0]
    dPfc_history, P_bat_history, U_DC_history = [0.0], [0.0], [0.0]

    # ==============================================================================
    # --- FIXED CVXPY COMPLIANT SETUP (ELEMENT-SPECIFIC ALGEBRA) ---
    # ==============================================================================
    X_soc = cp.Variable(n_horizon + 1)
    X_pfc = cp.Variable(n_horizon + 1)
    U_dpfc = cp.Variable(n_horizon) 

    # Numerical scalar parameters
    soc_init_param = cp.Parameter()
    pfc_init_param = cp.Parameter()
    P_req_param = cp.Parameter(n_horizon)  
    Udc_param = cp.Parameter() 
    etaFC_param = cp.Parameter() 
    Ibat_param = cp.Parameter() 

    P_bat_min_param = cp.Parameter()
    P_bat_max_param = cp.Parameter()

    cost = 0
    constraints = [
        X_soc[0] == soc_init_param,
        X_pfc[0] == pfc_init_param,

        X_soc[0] >=  0.4,  
        X_soc[0] <=  0.9,
    ]

    for k in range(n_horizon):
        constraints += [
            X_soc[k+1] == X_soc[k] + (X_pfc[k] * dt * eta_dcdc)/(Udc_param * Q_bat_s) \
                                   + (U_dpfc[k] * dt * eta_dcdc)/(Udc_param * Q_bat_s) \
                                   - (P_req_param[k] * dt)/(Udc_param * Q_bat_s),
            
            X_pfc[k+1] == X_pfc[k] + U_dpfc[k],
        ]

        # Operational limits constraints
        constraints += [
            X_pfc[k+1] >=  0.0,       # Min FC Power (kW)
            X_pfc[k+1] <= 60.0,       # Max FC Power (kW)
            U_dpfc[k]  >= -1.0,       # Rate limit bounds (kW/s)
            U_dpfc[k]  <=  1.0,        
            X_soc[k+1] >=  0.4,       # SoC tracking bounds
            X_soc[k+1] <=  0.9,
        ]
        # Costs
        
        C_H2  = W_H2 * X_pfc[k+1] * 1000 * dt / (etaFC_param * rho_H2)
        C_SOC = W_SoC * (X_soc[k] - X_soc[k+1]) * 47.3
        #L_SOC = L_SoC * cp.square(X_soc[k+1] - X_soc[k]) * 47.3
        cost += C_H2           
        cost += C_SOC           
        #cost += L_SOC           
        #cost += C_FC           
        #cost += W_SoC * cp.square(X_soc[k+1] - 0.6)      
        #cost += W_SoC * cp.square(X_soc[k+1] - X_soc[k]) * 47.3
        #cost += W_FC * cp.square(U_dpfc[k])            

    prob = cp.Problem(cp.Minimize(cost), constraints)

    # ==============================================================================
    # --- EXECUTION SIMULATION LOOP ---
    # ==============================================================================
    print("--- Diagnostics Start ---")

    cost_history = []

    for step in range(sim_steps):
        horizon_req = Preq_ems_input[step]
        if len(horizon_req) < n_horizon:
            horizon_req = np.pad(horizon_req, (0, n_horizon - len(horizon_req)), 'edge')

        horizon_req_kw = horizon_req / 1000.0

        R_bat, U_ocv = GetBatterieParams(x_current[0])
        pbat = horizon_req_kw[0] - x_current[1] * eta_dcdc
        if pbat < 0:
            pbat = 0.0
        I_bat_est = GetI_bat(x_current[0], pbat)
        U_DC = U_ocv - I_bat_est * R_bat 

        # FIX 1: UPDATE ALL PARAMETER VALUES PRIOR TO INVOKING THE SOLVER
        soc_init_param.value = float(x_current[0])
        pfc_init_param.value = float(x_current[1])
        Udc_param.value = float(U_DC)
        P_req_param.value = horizon_req_kw.flatten()
        etaFC_param.value = GetETAfc(x_current[1])
        Ibat_param.value = float(I_bat_est)
        P_batL = (U_DC * -200.0) / 1000.0
        P_batU = (U_DC * 300.0) / 1000.0

        P_bat_min_param.value = float(P_batL)
        P_bat_max_param.value = float(P_batU)

        # FIX 1 (CONTINUED): Solve now that parameters accurately describe the current step
        try:
            prob.solve(
                solver=cp.MOSEK, 
                verbose=False, 
                warm_start=True, 
            )
            u_control = float(U_dpfc[0].value)
            if U_dpfc[0].value is None: raise ValueError
        except Exception:
            u_control = 0.0  

        # --- Nonlinear Plant Physics Update ---
        P_fc_actual = float(x_current[1] + u_control)
        P_bat_actual = float(horizon_req_kw[0] - P_fc_actual * eta_dcdc)
        if P_bat_actual < 0: 
            P_bat_actual = 0.0

        P_bat_watts = P_bat_actual * 1000.0
        
        inside_sqrt = max(0.01, U_ocv**2 - 4.0 * R_bat * P_bat_watts)
        I_bat_actual = (U_ocv - np.sqrt(inside_sqrt)) / (2.0 * R_bat) 

        #U_DC_true = U_ocv - I_bat_actual * R_bat 
        soc_next = x_current[0] - (I_bat_actual * dt) / Q_bat_s

        soc_next = np.clip(soc_next, 0.0, 1.0)

        x_current = np.array([soc_next, P_fc_actual])

        t_history.append((step + 1) * dt)
        Preq_history.append(horizon_req_kw[0])
        cur_history.append(I_bat_actual)
        U_DC_history.append(U_DC)
        P_bat_history.append(P_bat_actual)
        P_batL_history.append(P_batL)
        P_batU_history.append(P_batU)
        Pfc_history.append(x_current[1])
        dPfc_history.append(u_control)
        SOC_history.append(x_current[0])
        
        if step % 10 == 0:
            print(f"Step {step} | Preq: {horizon_req_kw[0]:.2f} kW | "
                  #f"Pfc: {pfc_next_sol:.2f} kW | "
                  f"dPfc: {u_control:.2f} kW | "
                  f"Pbat: {P_bat_actual:.2f} kW |"
                  #f"C_H2: {C_H2_numeric:.3f} | "
                  #f"C_SoC: {C_SOC_numeric:.3f} | "
                  f"Status: {prob.status}")
            
    print("--- Diagnostics End ---\n")
    
    vecs = [t_history, Preq_history, cur_history, U_DC_history, P_bat_history, P_batL_history, P_batU_history, Pfc_history, dPfc_history, SOC_history]
    column_names = ['Time', 'P_req', 'I_bat', 'U_DC', 'P_bat', 'P_batL', 'P_batU', 'P_fc', 'dP_fc', 'SOC']
    df = pd.DataFrame()
    for col, vec in zip(column_names, vecs):
        df[col] = np.array(vec)

    return df

Preq_horizon_data = data_sim[11]
ems = SimulateEMS(Preq_horizon_data[1:], dt=1.0, n_horizon=3,W_H2=1e-5, W_SoC=1111, L_SoC=1, W_FC=896)

# %%
time, P_req, I_bat, U_DC, P_bat, P_batL, P_batU, P_fc, dP_fc, SOC = [ems[col].values for col in ems.columns]
ys = [I_bat]
xs = [time]*len(ys)
PlotSeriesPLY1(xs,ys, names=['I_bat'], title=None, h=300)

# %%
def get_vehicle_matrices(n_horizon, UDC, Q_bat=90, eta_dcdc=0.9, dt=1.0):

    Q_bat = Q_bat * 3600.0 
    A_list  = np.zeros((n_horizon, 2, 2))
    Bu_list = np.zeros((n_horizon, 2, 1))
    Bv_list = np.zeros((n_horizon, 2, 1))
    C_list  = np.zeros((n_horizon, 2, 2))
    D_list  = np.zeros((n_horizon, 2, 1))

    for k in range(n_horizon):

        A_list[k] = np.array([[1.0,  (dt * eta_dcdc)/(UDC*Q_bat)],
                              [0.0,  1.0]])
    
        Bu_list[k] = np.array([[(dt * eta_dcdc)/(UDC*Q_bat)],
                    [1.0]])
        
        Bv_list[k] = np.array([[-(dt )/(UDC*Q_bat)],
                    [0.0]])
        
        C_list[k] = np.eye(2)

        D_list[k] = np.array([[0],[1]])
    
    return A_list, Bu_list, Bv_list, C_list, D_list

def SimulateEMS2(Preq_horizon_raw, dt=1.0, n_horizon=3,
                W_H2=1, W_SoC=1, L_SoC=1, W_FC=1.0):
    
    # --- Downsample the 0.1s vehicle trajectory data to 1.0s intervals ---
    Preq_ems_input = []
    for step_1s in range(len(Preq_horizon_raw)):
        if step_1s % 4 == 0:
            forecast_vector_30points = Preq_horizon_raw[step_1s]
            forecast_3points_1s = forecast_vector_30points[[3, 7, 11]]
            Preq_ems_input.append(forecast_3points_1s)
            
    Preq_ems_input = np.array(Preq_ems_input) 
    sim_steps = len(Preq_ems_input)

    # --- Power Sources & Vehicle Constants (From Article Table 1) ---
    Q_bat_h = 90.0  # Battery Capacity (Ah)
    Q_bat_s = Q_bat_h * 3600.0 # Battery Capacity (As)
    E_bat = 47.3      # Battery Nominal Energy Capacity (kW h)
    rho_H2 = 120 * 1e6 # H2 Chemical Energy Density
    eta_dcdc = 0.9      # Unidirectional DC/DC Converter Efficiency
         
    x_current = np.array([0.6, 0.0])
    y_current = np.array([0.0, 0.0])
    u_last = 0
    v_last = 0
    
    t_history, SOC_history, Pfc_history = [0.0], [x_current[0]], [x_current[1]]
    cur_history = [0.0]
    Preq_history, P_batL_history, P_batU_history = [0.0], [0.0], [0.0]
    dPfc_history, P_bat_history, U_DC_history = [0.0], [0.0], [0.0]

    # ==============================================================================
    # --- FIXED CVXPY COMPLIANT SETUP (ELEMENT-SPECIFIC ALGEBRA) ---
    # ==============================================================================
    X_s = cp.Variable((2, n_horizon + 1))
    Y_s = cp.Variable((2, n_horizon + 1))
    U_s = cp.Variable((1, n_horizon))

    x_init = cp.Parameter(2)
    u_prev = cp.Parameter()

    # Numerical scalar parameters
    Preq_param = cp.Parameter(n_horizon)  
    etaFC_param = cp.Parameter() 
    Ibat_param = cp.Parameter() 

    PbatL_param = cp.Parameter()
    PbatU_param = cp.Parameter()

    Preq_ = cp.Parameter((1, n_horizon))
    A_stacked = cp.Parameter((n_horizon * 2, 2))
    
    Bu_stacked = cp.Parameter((n_horizon * 2, 1))
    Bv_stacked = cp.Parameter((n_horizon * 2, 1))
    C_stacked = cp.Parameter((n_horizon * 2, 2))
    D_stacked = cp.Parameter((n_horizon * 2, 1))

    cost = 0
    constraints = [X_s[:, 0] == x_init]

    for k in range(n_horizon):
        A_k = A_stacked[k*2 :(k+1)*2,:]
        Bu_k = Bu_stacked[k*2:(k+1)*2,:]
        Bv_k = Bv_stacked[k*2:(k+1)*2,:]
        C_k = C_stacked[k*2 :(k+1)*2,:]
        D_k = D_stacked[k*2 :(k+1)*2,:]

        constraints += [X_s[:,k+1] == A_k @ X_s[:,k] + Bu_k @ U_s[:,k] + Bv_k @ Preq_[:,k]]

        #constraints += [X_s[:,k+1] == X_s[:,k] - (Ibat_param * dt / Q_bat_h*3600)]

        constraints += [Y_s[:,k+1] == C_k @ X_s[:,k] + D_k @ U_s[:,k]]

        constraints += [X_s[1,k+1] == X_s[1,k] + U_s[:,k]]



        # Operational limits constraints
        constraints += [
            X_s[0,k+1] >=  0.3,      
            X_s[0,k+1] <=  0.9, 
            X_s[1,k+1] >=  0.0,       # Min FC Power (kW)
            X_s[1,k+1] <= 60.0,  
            U_s[0,k]   >= -1.0,       # Rate limit bounds (kW/s)
            U_s[0,k]   <=  1.0,
            Ibat_param >= -200,   # Min battery current (A)
            Ibat_param <=  300,
        ]

        # Costs
        C_H2  = W_H2 * X_s[1,k+1] * 1000 * dt / (etaFC_param * rho_H2)
        C_SOC = W_SoC * (X_s[0,k] - X_s[0,k+1]) * E_bat
        #L_SOC = L_SoC * cp.square(X_soc[k+1] - X_soc[k]) * 47.3
        cost += C_H2           
        cost += C_SOC           
         

    prob = cp.Problem(cp.Minimize(cost), constraints)

    # ==============================================================================
    # --- EXECUTION SIMULATION LOOP ---
    # ==============================================================================
    #print("--- Diagnostics Start ---")

    for step in range(sim_steps):
        horizon_req = Preq_ems_input[step]
        if len(horizon_req) < n_horizon:
            horizon_req = np.pad(horizon_req, (0, n_horizon - len(horizon_req)), 'edge')

        Preq_horizon = horizon_req / 1000.0

        Rbat, Uocv = GetBatterieParams(x_current[0])
        Pbat = Preq_horizon[0] - x_current[1] * eta_dcdc

        Ibat = GetI_bat(x_current[0], Pbat)

        if step ==0:
            print(Ibat,Preq_horizon[0])
            
        UDC = Uocv - Ibat * Rbat

        A_list, Bu_list, Bv_list, C_list, D_list = get_vehicle_matrices(n_horizon, UDC, dt=dt)

       
        Preq_.value = Preq_horizon.reshape(1,n_horizon)

        A_stacked.value  =  A_list.reshape(n_horizon * 2, 2)
        Bu_stacked.value = Bu_list.reshape(n_horizon * 2, 1)
        Bv_stacked.value = Bv_list.reshape(n_horizon * 2, 1)
        C_stacked.value  =  C_list.reshape(n_horizon * 2, 2)
        D_stacked.value  =  D_list.reshape(n_horizon * 2, 1)

        # FIX 1: UPDATE ALL PARAMETER VALUES PRIOR TO INVOKING THE SOLVER
        x_init.value = x_current
        u_prev.value = u_last
        
        Preq_param.value = Preq_horizon.flatten()
        etaFC_param.value = GetETAfc(x_current[1])
        Ibat_param.value = float(Ibat)


        # FIX 1 (CONTINUED): Solve now that parameters accurately describe the current step
        try:
            prob.solve(
                solver=cp.MOSEK, 
                verbose=False, 
                warm_start=True, 
            )
            u_control = float(U_s[:,0].value)
            if u_control is None: 
                raise ValueError
        except Exception:
            u_control = u_last  

        u_last = u_control
        u_in = np.array([[u_last]])
        v_in = np.array([[Preq_horizon[0]]])

        x_next = A_list[0] @ x_current.reshape(-1,1) + Bu_list[0] @ u_in + Bv_list[0] @ v_in
        x_next = x_next.flatten()
        print(x_next)
        y_current = C_list[0] @ x_current + D_list[0] @ u_in
        y_current = y_current.flatten()
        
        # --- Nonlinear Plant Physics Update ---
        Soc_next    = x_next[0]
        Pfc_prev    = float(x_current[1])
        Soc_current = float(y_current[0])
        Pfc_current = float(y_current[1])
        Pbat_current = float(Preq_horizon[0] - Pfc_current * eta_dcdc)
        Ibat_current, Rbat, UDC = GetBatterieParams2(Soc_current, Pbat_current)

        P_batL = (UDC * -200.0) / 1000.0
        P_batU = (UDC * 300.0) / 1000.0
    
        #print(x_current[0],y_current[0])

        x_current = np.array([Soc_next, Pfc_current])

        t_history.append((step + 1) * dt)
        Preq_history.append(Preq_horizon[0])
        cur_history.append(Ibat_current)
        U_DC_history.append(UDC)
        P_bat_history.append(Pbat_current)
        P_batL_history.append(P_batL)
        P_batU_history.append(P_batU)
        Pfc_history.append(x_current[1])
        dPfc_history.append(u_control)
        SOC_history.append(x_current[0])
        
        if step % 10 == 0:
            print(f"Step {step} | Preq: {Preq_horizon[0]:.2f} kW | "
                  #f"Pfc: {pfc_next_sol:.2f} kW | "
                  f"dPfc: {u_control:.2f} kW | "
                  f"Pbat: {Pbat_current:.2f} kW |"
                  f"Status: {prob.status}")
            
    print("--- Diagnostics End ---\n")
    
    vecs = [t_history, Preq_history, cur_history, U_DC_history, P_bat_history, P_batL_history, P_batU_history, Pfc_history, dPfc_history, SOC_history]
    column_names = ['Time', 'P_req', 'I_bat', 'U_DC', 'P_bat', 'P_batL', 'P_batU', 'P_fc', 'dP_fc', 'SOC']
    df = pd.DataFrame()
    for col, vec in zip(column_names, vecs):
        df[col] = np.array(vec)

    return df

Preq_horizon_data = data_sim[11]
ems = SimulateEMS2(Preq_horizon_data[1:], dt=1.0, n_horizon=3,W_H2=1e-5, W_SoC=1111, L_SoC=1, W_FC=896)

# %%
time, P_req, I_bat, U_DC, P_bat, P_batL, P_batU, P_fc, dP_fc, SOC = [ems[col].values for col in ems.columns]
ys = [P_bat,
      #P_batL,
      #P_batU,
      P_req,
      P_fc
      ]
xs = [time]*len(ys)
PlotSeriesPLY1(xs,ys, names=['P_bat (kW)', 'P_req (kW)', 'P_fc (kW)'], title=None, h=300)

# %%
time, P_req, I_bat, U_DC, P_bat, P_batL, P_batU, P_fc, dP_fc, SOC = [ems[col].values for col in ems.columns]
ys = [I_bat]
xs = [time]*len(ys)
PlotSeriesPLY1(xs,ys, names=['I_bat'], title=None, h=300)


