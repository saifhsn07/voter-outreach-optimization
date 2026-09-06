# Volunteer Outreach Resource Allocation Model

**A resource allocation / optimization model for maximizing new member signups
across grassroots outreach channels, given limited volunteer-hours.**

## Background

During my time as Membership Coordinator and Battleground 159 Apprentice at
the Georgia Youth Justice Coalition (GYJC), I helped run a statewide student
outreach program with a limited pool of volunteers spread across several
outreach channels: campus canvassing, public high-traffic canvassing (e.g.
the BeltLine), targeted door-to-door canvassing, and phone banking.

Hour-by-hour, it was never obvious which channel was actually the best use
of a volunteer's time — decisions were often made by habit or convenience
rather than by looking at return per hour. This project formalizes that
decision as a constrained optimization problem: **given a fixed pool of
volunteer-hours, how should they be allocated across channels to maximize
new member signups?**

> **Note on data:** GYJC does not have clean historical per-channel data
> available, so the figures used here are simulated/illustrative — built to
> reflect plausible, realistic relative differences between channels based
> on firsthand field experience, not actual organizational records. The
> value of the project is the **modeling approach**, which is directly
> reusable once real data is available.

## Problem Setup

| Channel | Signups / Volunteer-Hour | Conversion Rate | Max Feasible Hours/Week |
|---|---|---|---|
| Door-to-Door | 1.67 | 6.2% | 90 |
| Phone Banking | 1.50 | 3.0% | 32 |
| Public High-Traffic Canvassing | 5.00 | 10.0% | 20 |
| Campus High-Traffic Canvassing | 12.50 | 20.8% | 20 |

Total weekly volunteer-hour budget: **80 hours**

## Two Models

**1. Linear model** — assumes constant signups-per-hour for each channel, up
to its feasibility cap. Solved as a linear program (`scipy.optimize.linprog`).

**2. Diminishing-returns model** — more realistic: each channel saturates as
more hours are poured into it (a channel's contact pool isn't infinite). Modeled as
`S(h) = a * (1 - e^(-h/c))`, calibrated so the initial slope matches the
observed linear rate and each channel reaches ~80% saturation at its
feasibility cap. Solved with nonlinear optimization (`scipy.optimize.minimize`, SLSQP).

## Key Results

### Optimal allocation (diminishing-returns model, 80-hour budget)

| Channel | Allocated Hours | Signups |
|---|---|---|
| Campus HTC | 20.0 | 124.3 |
| Public HTC | 20.0 | 49.7 |
| Door-to-Door | 31.1 | 39.7 |
| Phone Banking | 8.9 | 10.8 |
| **Total** | **80.0** | **224.5** |

![Allocation comparison](charts/allocation_comparison.png)

### Status quo vs. optimized

Compared to a naive equal split of hours across all four channels, the
optimized allocation increases weekly signups from **~221 to ~225** under
the diminishing-returns model — a modest gain at this budget level, because
the equal split happens to be close to the cap-constrained optimum. The gap
widens substantially at smaller budgets, where prioritization matters most
(see sensitivity analysis below).

![Status quo vs optimized](charts/status_quo_vs_optimized.png)

### Sensitivity analysis

![Sensitivity analysis](charts/sensitivity_analysis.png)

This is the most important chart in the project. Two takeaways:

- **The linear model overstates what's achievable.** It assumes no
  saturation, so it keeps climbing well past 100 volunteer-hours/week.
- **The diminishing-returns model shows a real ceiling** — signups plateau
  around 100 hours/week, meaning that beyond a certain point, *more
  volunteers stop translating into proportionally more signups.* That's a
  meaningful organizational insight: past a certain scale, the bottleneck
  isn't volunteer supply, it's the size of the addressable audience per
  channel. Growth requires **new channels**, not just more hours in
  existing ones.

### Per-channel saturation curves

![Diminishing returns curves](charts/diminishing_returns_curves.png)

Campus HTC saturates fastest (highest initial efficiency, smallest audience
pool), while door-to-door saturates slowest — meaning it has the most
untapped long-run capacity even though it's not the most efficient channel
per hour today.

## Interpretation / Recommendation

1. **Campus and Public HTC should be run at capacity first** — they deliver
   the highest signups per hour and should never be under-resourced.
2. **Phone banking is not worthless, but it is a last resort** — it only
   becomes worth using once the higher-efficiency channels have saturated,
   not before.
3. **Door-to-door is the main lever for growth** beyond the current ceiling,
   since it has the most remaining headroom before saturating.
4. Historically, if resources were spread by convenience/habit rather than
   this kind of prioritization, the organization was very likely leaving
   signups on the table at lower budget levels — the sensitivity chart shows
   the cost of under-prioritizing the top channels is highest when
   volunteer-hours are scarce.

## Assumptions & Limitations

- All input data is simulated/illustrative, not actual GYJC records.
- Diminishing-returns saturation points are approximated from feasibility
  caps, not measured directly.
- The model assumes channels are independent (no interaction effects, e.g.
  campus canvassing driving later phone-bank conversions).
- Volunteer skill/training differences within a channel are not modeled.

## Tech Stack

- Python (NumPy, SciPy, Matplotlib)
- Linear programming (`scipy.optimize.linprog`, HiGHS solver)
- Nonlinear constrained optimization (`scipy.optimize.minimize`, SLSQP)

## Files

- `model.py` — full model, optimization, and chart generation
- `charts/` — all generated figures
- `README.md` — this write-up
