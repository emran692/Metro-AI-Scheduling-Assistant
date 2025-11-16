# optimizer.py
from ortools.sat.python import cp_model
import pandas as pd

def generate_schedule(daily_data, risk_scores, manual_overrides={}):
    opt_model = cp_model.CpModel()
    num_trains = len(daily_data)
    trains = range(num_trains)
    is_service = [opt_model.NewBoolVar(f'is_service_{i}') for i in trains]
    is_standby = [opt_model.NewBoolVar(f'is_standby_{i}') for i in trains]
    is_maintenance = [opt_model.NewBoolVar(f'is_maintenance_{i}') for i in trains]
    for i in trains:
        opt_model.AddExactlyOne([is_service[i], is_standby[i], is_maintenance[i]])
        train_id = daily_data.loc[i, 'Train_ID']
        if train_id in manual_overrides:
            override_status = manual_overrides[train_id]
            if override_status == 'Force Maintenance': opt_model.Add(is_maintenance[i] == 1)
            elif override_status == 'Force Service': opt_model.Add(is_service[i] == 1)
            continue 
        if daily_data.loc[i, 'JobCard_Priority'] == 'High' or daily_data.loc[i, 'Min_Cert_Days_to_Expiry'] < 0:
            opt_model.Add(is_maintenance[i] == 1)
    objective_terms = []
    for i in trains:
        branding_hours = daily_data.loc[i, 'Branding_Contract_Hours']
        if branding_hours > 0: objective_terms.append(is_service[i] * int(branding_hours))
        mileage_penalty = int(daily_data.loc[i, 'Bogie_KM_Since_Service'] / 1000)
        objective_terms.append(-is_service[i] * mileage_penalty)
        risk_penalty = int(risk_scores.iloc[i] * 100)
        objective_terms.append(-is_service[i] * risk_penalty)
    opt_model.Maximize(sum(objective_terms))
    solver = cp_model.CpSolver()
    status = solver.Solve(opt_model)
    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        results = []
        for i in trains:
            train_id = daily_data.loc[i, 'Train_ID']
            decision, reason = ("Unknown", "")
            if solver.Value(is_service[i]): decision = "✅ SERVICE"
            elif solver.Value(is_standby[i]): decision = "🅿️ STANDBY"
            elif solver.Value(is_maintenance[i]):
                decision = "🛠️ MAINTENANCE"
                if train_id in manual_overrides and manual_overrides[train_id] == 'Force Maintenance': reason = "Manual Override"
                elif daily_data.loc[i, 'JobCard_Priority'] == 'High': reason = f"High-priority job: {daily_data.loc[i, 'JobCard_Type']}"
                elif daily_data.loc[i, 'Min_Cert_Days_to_Expiry'] < 0: reason = "Certificate expired."
            results.append({'Train_ID': train_id, 'Assigned_Status': decision, 'Reason': reason})
        return pd.DataFrame(results)
    return None
