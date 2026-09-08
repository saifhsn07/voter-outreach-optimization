import numpy as np
from scipy.optimize import linprog, minimize
import matplotlib.pyplot as plt

channels = ["Door-to-Door", "Phone Banking", "Public HTC", "Campus HTC"]

vol_hours_per_shift = np.array([3, 2, 2, 2], dtype=float)
contacts_per_shift = np.array([80, 100, 100, 120], dtype=float)
signups_per_shift = np.array([5, 3, 10, 25], dtype=float)

signups_per_hour = signups_per_shift / vol_hours_per_shift     
conversion_rate = signups_per_shift / contacts_per_shift         

max_hours_per_week = np.array([90, 32, 20, 20], dtype=float)     
TOTAL_BUDGET = 80.0                                                  

print("Channel efficiency summary")
for i, c in enumerate(channels):
    print(f"  {c:20s}  {signups_per_hour[i]:.2f} signups/hr   "
          f"{conversion_rate[i]*100:.1f}% conversion   cap={max_hours_per_week[i]:.0f} hrs/wk")
  
def optimize_linear(budget, caps=max_hours_per_week, rates=signups_per_hour):
    """Maximize sum(rate_i * h_i) s.t. sum(h_i) <= budget, 0 <= h_i <= cap_i"""
    c = -rates  
    A_ub = [np.ones(len(rates))]
    b_ub = [budget]
    bounds = [(0, cap) for cap in caps]
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
    hours = res.x
    total_signups = -res.fun
    return hours, total_signups

linear_hours, linear_total = optimize_linear(TOTAL_BUDGET)

SATURATION_FRACTION = 0.8
c_param = max_hours_per_week / (-np.log(1 - SATURATION_FRACTION)) 
a_param = signups_per_hour * c_param                                 

def diminishing_signups(h, a=a_param, c=c_param):
    return a * (1 - np.exp(-np.asarray(h) / c))

def optimize_diminishing(budget, caps=max_hours_per_week, a=a_param, c=c_param):
    n = len(caps)
    def neg_total(h):
        return -np.sum(diminishing_signups(h, a, c))
    constraints = [{"type": "ineq", "fun": lambda h: budget - np.sum(h)}]
    bounds = [(0, cap) for cap in caps]
    x0 = np.full(n, budget / n)
    res = minimize(neg_total, x0, bounds=bounds, constraints=constraints, method="SLSQP")
    return res.x, -res.fun

dr_hours, dr_total = optimize_diminishing(TOTAL_BUDGET)

equal_hours = np.minimum(TOTAL_BUDGET / len(channels), max_hours_per_week)
leftover = TOTAL_BUDGET - equal_hours.sum()
while leftover > 1e-6:
    room = max_hours_per_week - equal_hours
    open_idx = np.where(room > 1e-6)[0]
    if len(open_idx) == 0:
        break
    add = leftover / len(open_idx)
    for i in open_idx:
        give = min(add, room[i])
        equal_hours[i] += give
        leftover -= give

status_quo_linear_total = np.sum(equal_hours * signups_per_hour)
status_quo_dr_total = np.sum(diminishing_signups(equal_hours))

print(f"\nLinear model      -> optimal total signups: {linear_total:.1f}")
print(f"Diminishing model -> optimal total signups: {dr_total:.1f}")
print(f"Status quo (equal split) -> linear: {status_quo_linear_total:.1f}, "
      f"diminishing: {status_quo_dr_total:.1f}")

budgets = np.linspace(10, 180, 25)
linear_curve = []
dr_curve = []
for b in budgets:
    _, lt = optimize_linear(b)
    _, dt = optimize_diminishing(b)
    linear_curve.append(lt)
    dr_curve.append(dt)

plt.rcParams.update({"figure.dpi": 150, "font.size": 10})
colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

fig, ax = plt.subplots(figsize=(7, 4.5))
x = np.arange(len(channels))
width = 0.35
ax.bar(x - width/2, linear_hours, width, label="Linear model", color="#4C72B0")
ax.bar(x + width/2, dr_hours, width, label="Diminishing returns model", color="#DD8452")
ax.set_xticks(x)
ax.set_xticklabels(channels, rotation=15)
ax.set_ylabel("Allocated Volunteer-Hours / Week")
ax.set_title("Optimal Hour Allocation: Linear vs. Diminishing-Returns Model")
ax.legend()
fig.tight_layout()
fig.savefig("charts/allocation_comparison.png")
plt.close(fig)

fig, ax = plt.subplots(figsize=(6, 4.5))
labels = ["Status Quo\n(equal split)", "Optimized\n(model-driven)"]
values = [status_quo_dr_total, dr_total]
bars = ax.bar(labels, values, color=["#C44E52", "#55A868"])
for bar, v in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width()/2, v + 2, f"{v:.0f}", ha="center", fontweight="bold")
ax.set_ylabel("Total Weekly Signups")
ax.set_title("Status Quo vs. Optimized Allocation (Diminishing Returns Model)")
fig.tight_layout()
fig.savefig("charts/status_quo_vs_optimized.png")
plt.close(fig)

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(budgets, linear_curve, label="Linear model", color="#4C72B0", linewidth=2)
ax.plot(budgets, dr_curve, label="Diminishing returns model", color="#DD8452", linewidth=2)
ax.axvline(TOTAL_BUDGET, color="gray", linestyle="--", linewidth=1, label="Current budget (80 hrs)")
ax.set_xlabel("Total Volunteer-Hour Budget / Week")
ax.set_ylabel("Optimal Total Signups")
ax.set_title("Sensitivity: Signups vs. Total Volunteer-Hour Budget")
ax.legend()
fig.tight_layout()
fig.savefig("charts/sensitivity_analysis.png")
plt.close(fig)

fig, ax = plt.subplots(figsize=(7, 4.5))
h_range = np.linspace(0, 100, 200)
for i, ch in enumerate(channels):
    ax.plot(h_range, diminishing_signups(h_range, a_param[i], c_param[i]),
            label=ch, color=colors[i], linewidth=2)
    ax.axvline(max_hours_per_week[i], color=colors[i], linestyle=":", alpha=0.5)
ax.set_xlabel("Volunteer-Hours / Week")
ax.set_ylabel("Expected Signups")
ax.set_title("Diminishing Returns Curves by Channel\n(dotted lines = feasibility cap)")
ax.legend()
fig.tight_layout()
fig.savefig("charts/diminishing_returns_curves.png")
plt.close(fig)

fig, ax1 = plt.subplots(figsize=(7, 4.5))
ax1.bar(x, signups_per_hour, color=colors)
ax1.set_xticks(x)
ax1.set_xticklabels(channels, rotation=15)
ax1.set_ylabel("Signups per Volunteer-Hour")
ax1.set_title("Channel Efficiency: Signups per Volunteer-Hour")
fig.tight_layout()
fig.savefig("charts/signups_per_hour.png")
plt.close(fig)

print("\nAll charts saved to charts/")

print("\n--- Optimal allocation (diminishing returns model) ---")
for i, c in enumerate(channels):
    print(f"  {c:20s}  {dr_hours[i]:6.1f} hrs  ->  {diminishing_signups(dr_hours[i], a_param[i], c_param[i]):6.1f} signups")
print(f"  {'TOTAL':20s}  {dr_hours.sum():6.1f} hrs  ->  {dr_total:6.1f} signups")
