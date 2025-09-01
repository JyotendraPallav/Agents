# app.py — Gradio portal for AI Product ROI
# Run: uv run app.py  (or)  python app.py

from __future__ import annotations
import json
from dataclasses import dataclass, asdict
from typing import Dict, Any, Tuple
import pandas as pd
import gradio as gr

# ---------- Core ROI logic ----------

WEEKS_PER_YEAR = 52

@dataclass
class Assumptions:
    product_name: str = "IKIA / ARIA / ADA / DT"
    currency: str = "USD"

    # Value drivers
    users: int = 200
    hours_per_user_per_week: float = 2.0
    cost_per_hour: float = 80.0
    process_annual_cost: float = 500000.0
    efficiency_pct: float = 20.0           # %
    fte_cost: float = 120000.0
    fte_avoided: float = 3
    error_cost: float = 50000.0
    errors_avoided: int = 10
    baseline_sales: float = 200_000_000.0
    sales_lift_pct: float = 1.0            # %
    earlier_rev_months: int = 3
    earlier_rev_monthly: float = 200000.0

    # Build inputs
    build_ftes: float = 4.0
    build_cost_per_fte_month: float = 12000.0
    build_months: int = 6
    build_licenses: float = 25_000.0
    build_data_prep: float = 20_000.0
    build_integration: float = 30_000.0
    build_training: float = 10_000.0
    build_other: float = 15_000.0

    # Operate inputs
    cloud_monthly: float = 8000.0
    llm_cost_per_1k: float = 2.0
    api_calls_thousands_per_year: float = 1000.0
    support_ftes: float = 1.5
    enhance_pct_of_build: float = 20.0     # %
    operate_training: float = 10_000.0
    operate_other: float = 20_000.0

def calc_value(a: Assumptions) -> Tuple[pd.DataFrame, float]:
    rows = []
    # Time savings
    time_val = a.users * a.hours_per_user_per_week * WEEKS_PER_YEAR * a.cost_per_hour
    rows.append(["Time Savings / Productivity",
                 "Users × Hours/week × 52 × $/hour", time_val])

    # Process reduction
    proc_val = (a.efficiency_pct / 100.0) * a.process_annual_cost
    rows.append(["Process Cost Reduction",
                 "% efficiency × Process Annual Cost", proc_val])

    # FTE avoidance
    fte_val = a.fte_avoided * a.fte_cost
    rows.append(["Headcount Avoidance", "FTEs avoided × FTE cost", fte_val])

    # Error reduction
    err_val = a.errors_avoided * a.error_cost
    rows.append(["Error Reduction / Compliance", "Errors avoided × $/error", err_val])

    # Revenue uplift
    rev_val = (a.sales_lift_pct / 100.0) * a.baseline_sales
    rows.append(["Revenue Uplift / Growth", "% sales lift × Baseline sales", rev_val])

    # Earlier capture
    early_val = a.earlier_rev_months * a.earlier_rev_monthly
    rows.append(["Earlier Revenue Capture", "Months accelerated × Monthly revenue", early_val])

    df = pd.DataFrame(rows, columns=["Category", "Basis", "Annual_Value"])
    return df, float(df["Annual_Value"].sum())

def calc_build(a: Assumptions) -> Tuple[pd.DataFrame, float]:
    effort = a.build_ftes * a.build_cost_per_fte_month * a.build_months
    rows = [
        ["Team Effort (FTE-months) – auto", f"{a.build_ftes} × {a.build_cost_per_fte_month:,.0f} × {a.build_months}", effort],
        ["Technology Licenses (one-time)", "Input", a.build_licenses],
        ["Data Preparation (one-time)", "Input", a.build_data_prep],
        ["Integration / Infra Setup (one-time)", "Input", a.build_integration],
        ["Training / Change Mgmt (one-time)", "Input", a.build_training],
        ["Other (Consulting / Vendor)", "Input", a.build_other],
    ]
    df = pd.DataFrame(rows, columns=["Build_Component", "Basis", "One_Time_Cost"])
    return df, float(df["One_Time_Cost"].sum())

def calc_operate(a: Assumptions, build_total: float) -> Tuple[pd.DataFrame, float]:
    cloud = a.cloud_monthly * 12
    llm = a.api_calls_thousands_per_year * (a.llm_cost_per_1k * 1000.0)
    support = a.support_ftes * a.fte_cost
    enhance = (a.enhance_pct_of_build / 100.0) * build_total
    rows = [
        ["Cloud / Infra (annual) – auto", "Cloud monthly × 12", cloud],
        ["LLM/API (annual) – auto", "1K calls × $/1K × estimated '000 calls", llm],
        ["Support & Maintenance (FTEs) – auto", "Support FTEs × FTE Cost", support],
        ["Enhancements (annual) – auto", "% of Build × Build Total", enhance],
        ["Training / Onboarding (annual)", "Input", a.operate_training],
        ["Other (monitoring, compliance, etc.)", "Input", a.operate_other],
    ]
    df = pd.DataFrame(rows, columns=["Operate_Component", "Basis", "Annual_Cost"])
    return df, float(df["Annual_Cost"].sum())

def summarize(a: Assumptions) -> Tuple[pd.DataFrame, Dict[str, float]]:
    value_df, total_value = calc_value(a)
    build_df, build_total = calc_build(a)
    operate_df, operate_total = calc_operate(a, build_total)

    total_cost_year1 = build_total + operate_total
    total_cost_year2 = operate_total
    net_year1 = total_value - total_cost_year1
    net_year2 = total_value - total_cost_year2
    roi_year1 = (net_year1 / total_cost_year1) if total_cost_year1 else 0.0
    roi_year2 = (net_year2 / total_cost_year2) if total_cost_year2 else 0.0
    payback_months = (build_total / ((total_value - operate_total) / 12)) if (total_value - operate_total) > 0 else None

    summary = {
        "Total Value / Year": total_value,
        "Total Cost (Year 1)": total_cost_year1,
        "Total Cost (Year 2+)": total_cost_year2,
        "Net Impact (Year 1)": net_year1,
        "Net Impact (Year 2+)": net_year2,
        "ROI % (Year 1)": roi_year1 * 100,
        "ROI % (Year 2+)": roi_year2 * 100,
        "Payback (Months)": payback_months if payback_months is not None else float("nan"),
    }

    # Assemble a flat table for download
    download_rows = []
    for _, r in value_df.iterrows():
        download_rows.append(["Value", r["Category"], r["Basis"], r["Annual_Value"]])
    for _, r in build_df.iterrows():
        download_rows.append(["Build", r["Build_Component"], r["Basis"], r["One_Time_Cost"]])
    for _, r in operate_df.iterrows():
        download_rows.append(["Operate", r["Operate_Component"], r["Basis"], r["Annual_Cost"]])
    flat = pd.DataFrame(download_rows, columns=["Section", "Item", "Basis", "Amount"])

    return flat, summary

# ---------- Glossary / Methodology ----------

def glossary_df() -> pd.DataFrame:
    rows = [
        ["Users", "Count of end users benefiting from the tool."],
        ["Hours Saved per User per Week", "Average time saved per user weekly via automation/assistants."],
        ["Fully Loaded Cost per Hour", "Hourly cost incl. salary, benefits, overhead."],
        ["Process Annual Cost (baseline)", "Total annual cost of the current (pre-AI) process."],
        ["Efficiency Improvement (%)", "Percent reduction in baseline process cost (e.g., 20%)."],
        ["FTE Cost per Year", "Fully loaded annual cost of an FTE."],
        ["FTEs Avoided (count)", "Number of roles avoided or capacity redeployed."],
        ["Error Cost per Incident", "Average financial impact per error/compliance incident."],
        ["Errors Avoided (count)", "Expected count of avoided incidents per year."],
        ["Baseline Annual Sales", "Annual revenue base relevant to the tool’s uplift."],
        ["Sales Lift (%)", "Incremental % sales growth enabled by the tool."],
        ["Earlier Revenue Capture (months)", "How many months earlier revenue can be realized."],
        ["Monthly Revenue Impact (Earlier Capture)", "Monthly revenue tied to the accelerated timeline."],
        ["Build FTEs", "Number of FTEs for build phase."],
        ["Cost per FTE per Month", "Monthly cost per build FTE."],
        ["Build Duration (months)", "Length of build phase in months."],
        ["Tech Licenses (one-time)", "One-time build licenses (LLM evals, tools, etc.)."],
        ["Data Preparation (one-time)", "Cleansing/annotation/mapping for initial launch."],
        ["Integration / Infra (one-time)", "Connectors, pipelines, platform setup."],
        ["Training / Change Mgmt (one-time)", "Initial enablement sessions and materials."],
        ["Other (one-time)", "Any other build CAPEX."],
        ["Cloud Monthly Cost", "Hosting/compute/storage monthly operating cost."],
        ["LLM/API Cost per 1K Calls", "Unit price per 1,000 LLM/API calls."],
        ["Estimated API Calls per Year (thousands)", "Annual volume (in thousands) of calls."],
        ["Support & Maintenance FTEs", "Ongoing team to run/monitor/support the tool."],
        ["Enhancements Budget (% of Build per year)", "Run-rate improvements as % of build total."],
        ["Training / Onboarding (annual)", "Ongoing enablement & onboarding."],
        ["Other Operate Costs (annual)", "Monitoring, compliance, observability, etc."],
    ]
    return pd.DataFrame(rows, columns=["Variable", "What it Means"])

def methodology_md() -> str:
    return f"""
### Methodology & Formulas

**Constant:** Weeks per year = **{WEEKS_PER_YEAR}**

#### Value (Annual)
- **Time Savings / Productivity** = Users × Hours/Week × {WEEKS_PER_YEAR} × Cost/Hour  
- **Process Cost Reduction** = Efficiency% × Process Annual Cost  
- **Headcount Avoidance** = FTEs Avoided × FTE Cost  
- **Error Reduction / Compliance** = Errors Avoided × Error Cost  
- **Revenue Uplift / Growth** = Sales Lift% × Baseline Sales  
- **Earlier Revenue Capture** = Earlier Months × Monthly Revenue Impact

**Total Value / Year** = Sum of all above

#### Costs
**Build (one-time)**  
- **Team Effort (auto)** = Build FTEs × Cost/FTE/Month × Build Months  
- Plus any one-time line items (Licenses, Data Prep, Integration, Training, Other)

**Operate (annual)**  
- **Cloud / Infra** = Cloud Monthly × 12  
- **LLM/API** = Calls ('000) × (Cost per 1K × 1000)  
- **Support & Maintenance** = Support FTEs × FTE Cost  
- **Enhancements** = Enhance% × Build Total  
- Plus Training (annual), Other (annual)

#### Roll-ups
- **Total Cost (Year 1)** = Build + Operate  
- **Total Cost (Year 2+)** = Operate  
- **Net Impact (Year 1)** = Value – (Build + Operate)  
- **Net Impact (Year 2+)** = Value – Operate  
- **ROI % (Year 1)** = (Value – (Build + Operate)) / (Build + Operate) × 100  
- **ROI % (Year 2+)** = (Value – Operate) / Operate × 100  
- **Payback (Months)** = Build / ((Value – Operate) / 12)

#### Guidance & Tips
- Use **conservative** estimates (pilot data, audit trails).  
- Separate **automation savings** vs **commercial uplift** when socializing results.  
- For sales lift, validate with **A/B** or **matched cohort** where possible.  
- For FTE avoidance, specify whether it’s **avoidance**, **redeployment**, or **backfill reduction**.  
"""



# ---------- Gradio UI ----------

def format_money(x: float, cur: str) -> str:
    try:
        return f"{cur} {x:,.0f}"
    except Exception:
        return str(x)

def run_calc(inputs: Dict[str, Any]) -> Tuple[str, pd.DataFrame, str]:
    a = Assumptions(**inputs)
    flat, s = summarize(a)
    cur = a.currency
    # Format Amount with currency + commas for display
    flat["Amount"] = flat["Amount"].apply(lambda x: f"{cur} {x:,.0f}")
    summary_md = f"""### ROI Summary — {a.product_name}

- **Total Value / Year:** {format_money(s['Total Value / Year'], cur)}
- **Total Cost (Year 1):** {format_money(s['Total Cost (Year 1)'], cur)}
- **Total Cost (Year 2+):** {format_money(s['Total Cost (Year 2+)'], cur)}
- **Net Impact (Year 1):** **{format_money(s['Net Impact (Year 1)'], cur)}**
- **Net Impact (Year 2+):** **{format_money(s['Net Impact (Year 2+)'], cur)}**
- **ROI % (Year 1):** {s['ROI % (Year 1)']:.1f}%
- **ROI % (Year 2+):** {s['ROI % (Year 2+)']:.1f}%
- **Payback (Months):** {'' if pd.isna(s["Payback (Months)"]) else f'{s["Payback (Months)"]:.1f}'}
"""
    # CSV for download
    csv_path = "roi_line_items.csv"
    flat.to_csv(csv_path, index=False)
    return summary_md, flat, csv_path

def load_scenario(file_obj) -> Dict[str, Any]:
    if file_obj is None:
        return gr.update()
    data = json.load(open(file_obj.name, "r"))
    return data

def save_scenario(inputs: Dict[str, Any]) -> str:
    fname = f"scenario_{inputs.get('product_name','product').replace(' ','_')}.json"
    with open(fname, "w") as f:
        json.dump(inputs, f, indent=2)
    return fname

with gr.Blocks(title="AI Product ROI Calculator") as demo:
    gr.Markdown("# AI Product ROI Calculator\nAdjust assumptions and view live ROI for Year-1 and Year-2+.")

    with gr.Tabs():
        with gr.TabItem("Calculator"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### General")
                    product_name = gr.Textbox(label="Product Name", value="IKIA / ARIA / ADA / DT")
                    currency = gr.Dropdown(["USD", "EUR", "INR"], value="USD", label="Currency")

                    gr.Markdown("### Value Drivers")
                    users = gr.Number(label="Number of Users", value=200, precision=0)
                    hours = gr.Slider(0, 10, value=2.0, step=0.1, label="Hours Saved per User per Week")
                    rate = gr.Number(label="Fully Loaded Cost per Hour", value=80.0)
                    process_cost = gr.Number(label="Process Annual Cost (baseline)", value=500000.0)
                    efficiency = gr.Slider(0, 100, value=20.0, step=0.5, label="Efficiency Improvement (%)")
                    fte_cost = gr.Number(label="FTE Cost per Year", value=120000.0)
                    fte_avoided = gr.Number(label="FTEs Avoided (count)", value=3, precision=0)
                    err_cost = gr.Number(label="Error Cost per Incident", value=50000.0)
                    errs_avoided = gr.Number(label="Errors Avoided (count)", value=10, precision=0)
                    baseline = gr.Number(label="Baseline Annual Sales", value=200_000_000.0)
                    sales_lift = gr.Slider(0, 10, value=1.0, step=0.1, label="Sales Lift (%)")
                    early_months = gr.Slider(0, 12, value=3, step=1, label="Earlier Revenue Capture (months)")
                    early_monthly = gr.Number(label="Monthly Revenue Impact (Earlier Capture)", value=200000.0)

                    gr.Markdown("### Build (One-time)")
                    build_ftes = gr.Number(label="Build FTEs (count)", value=4.0)
                    build_rate = gr.Number(label="Cost per FTE per Month", value=12000.0)
                    build_months = gr.Number(label="Build Duration (months)", value=6, precision=0)
                    build_licenses = gr.Number(label="Technology Licenses (one-time)", value=25000.0)
                    build_data = gr.Number(label="Data Preparation (one-time)", value=20000.0)
                    build_integ = gr.Number(label="Integration / Infra Setup (one-time)", value=30000.0)
                    build_train = gr.Number(label="Training / Change Mgmt (one-time)", value=10000.0)
                    build_other = gr.Number(label="Other (Consulting / Vendor)", value=15000.0)

                    gr.Markdown("### Operate (Annual)")
                    cloud_monthly = gr.Number(label="Cloud Monthly Cost", value=8000.0)
                    llm_per_1k = gr.Number(label="LLM/API Cost per 1K Calls", value=2.0)
                    api_calls_k = gr.Number(label="Estimated API Calls per Year (thousands)", value=1000.0)
                    support_ftes = gr.Number(label="Support & Maintenance FTEs", value=1.5)
                    enhance_pct = gr.Slider(0, 100, value=20.0, step=1, label="Enhancements Budget (% of Build per year)")
                    op_train = gr.Number(label="Training / Onboarding (annual)", value=10000.0)
                    op_other = gr.Number(label="Other Operate Costs (annual)", value=20000.0)

                    # Scenario save/load
                    gr.Markdown("### Scenarios")
                    scenario_file = gr.File(label="Load Scenario (JSON)", file_count="single", type="filepath")
                    load_btn = gr.Button("Load Scenario")
                    save_btn = gr.Button("Save Current Scenario")

                with gr.Column(scale=1):
                    summary = gr.Markdown("### ROI Summary will appear here")
                    table = gr.Dataframe(headers=["Section", "Item", "Basis", "Amount"], label="Line Items (for download)", interactive=False, wrap=True)
                    download = gr.File(label="Download CSV")

                    # Sensitivity quick sliders (mirror a couple of key drivers)
                    gr.Markdown("### Quick Sensitivity")
                    sens_hours = gr.Slider(0, 10, value=2.0, step=0.1, label="Hours Saved / User / Week")
                    sens_sales = gr.Slider(0, 10, value=1.0, step=0.1, label="Sales Lift (%)")

                    calc_btn = gr.Button("Recalculate ROI", variant="primary")

            # Wire up interactions for Calculator tab
            inputs = {
                "product_name": product_name, "currency": currency,
                "users": users, "hours_per_user_per_week": hours, "cost_per_hour": rate,
                "process_annual_cost": process_cost, "efficiency_pct": efficiency,
                "fte_cost": fte_cost, "fte_avoided": fte_avoided,
                "error_cost": err_cost, "errors_avoided": errs_avoided,
                "baseline_sales": baseline, "sales_lift_pct": sales_lift,
                "earlier_rev_months": early_months, "earlier_rev_monthly": early_monthly,
                "build_ftes": build_ftes, "build_cost_per_fte_month": build_rate,
                "build_months": build_months, "build_licenses": build_licenses,
                "build_data_prep": build_data, "build_integration": build_integ,
                "build_training": build_train, "build_other": build_other,
                "cloud_monthly": cloud_monthly, "llm_cost_per_1k": llm_per_1k,
                "api_calls_thousands_per_year": api_calls_k, "support_ftes": support_ftes,
                "enhance_pct_of_build": enhance_pct, "operate_training": op_train, "operate_other": op_other
            }

            def collect_inputs_and_run(*vals):
                keys = list(inputs.keys())
                data = {k: v for k, v in zip(keys, vals)}
                return run_calc(data)

            calc_btn.click(
                collect_inputs_and_run,
                inputs=list(inputs.values()),
                outputs=[summary, table, download],
            )

            # Quick sensitivity reuses existing fields but overrides key drivers
            def apply_sensitivity(h, s, *vals):
                keys = list(inputs.keys())
                data = {k: v for k, v in zip(keys, vals)}
                data["hours_per_user_per_week"] = h
                data["sales_lift_pct"] = s
                return run_calc(data)

            sens_hours.release(
                apply_sensitivity,
                inputs=[sens_hours, sens_sales, *list(inputs.values())],
                outputs=[summary, table, download],
            )
            sens_sales.release(
                apply_sensitivity,
                inputs=[sens_hours, sens_sales, *list(inputs.values())],
                outputs=[summary, table, download],
            )

            # Load/Save scenarios
            def _load(file_obj):
                data = load_scenario(file_obj)
                if not data: return [gr.update()]*len(inputs)
                return [gr.update(value=data.get(k)) for k in inputs.keys()]

            load_btn.click(
                _load,
                inputs=[scenario_file],
                outputs=list(inputs.values()),
            )

            def _save(*vals):
                keys = list(inputs.keys())
                data = {k: v for k, v in zip(keys, vals)}
                return save_scenario(data)

            save_btn.click(
                _save,
                inputs=list(inputs.values()),
                outputs=[download],
            )

        with gr.TabItem("Glossary & Methodology"):
            gr.Markdown("### Variable Glossary")
            gr.Dataframe(value=glossary_df(), interactive=False, wrap=True)

            with gr.Accordion("Methodology & Formulas", open=False):
                gr.Markdown(methodology_md())

            with gr.Accordion("How to Defend Your Assumptions", open=False):
                gr.Markdown(
                    "- Use pilot logs and time-motion studies for **hours saved**.\n"
                    "- Validate **sales lift** with A/B or matched cohorts.\n"
                    "- For **FTE avoidance**, specify avoidance vs redeployment.\n"
                    "- Track **LLM/API** usage in telemetry to true-up run costs.\n"
                    "- Keep a versioned **assumptions pack** per scenario."
                )

demo.launch()
