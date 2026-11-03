from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.services.pipeline import run_scenario

app = FastAPI(
    title="ECONOSIM-AI API",
    version="1.0.0",
    description="Counterfactual economic policy simulation and AI impact analysis.",
)


class ScenarioRequest(BaseModel):
    policy_name: str = Field(
        pattern="^(income_tax|interest_rate|government_spending|subsidy)$"
    )
    baseline_value: float
    counterfactual_value: float
    duration_years: int = Field(default=5, ge=1, le=20)


@app.get("/")
def root():
    return {"project": "ECONOSIM-AI", "status": "running"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/simulate")
def simulate(request: ScenarioRequest):
    try:
        result = run_scenario(
            request.policy_name,
            request.baseline_value,
            request.counterfactual_value,
            request.duration_years,
        )
        return {
            "policy": result["policy"],
            "baseline_history": result["baseline_history"],
            "counterfactual_history": result["counterfactual_history"],
            "comparison": result["comparison"].to_dict(orient="records"),
            "predicted_baseline_gdp": result["predicted_baseline_gdp"],
            "predicted_counterfactual_gdp": result["predicted_counterfactual_gdp"],
            "explanation": result["explanation"],
            "feature_importance": result["feature_importance"].to_dict(orient="records"),
            "model": result["model"],
            "metrics": result["metrics"],
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
