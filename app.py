from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from Models.cultural_fit import calculate_cultural_fit
from Models.monetary_motivation import calculate_monetary_score
from Models.recommendation_engine import generate_final_score
from Models.stability_score import calculate_stability
from Models.long_term_prediction import predict_long_term
from Models.ml_long_term_predictor import predict_long_term_ml

app = FastAPI(
    title="Hiring Recommendation API",
    description="Scores candidates on stability, long-term retention, monetary motivation, and cultural fit.",
    version="1.0.0"
)

# Allow the local frontend (or any origin, for local/dev use) to call this API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class CandidateInput(BaseModel):
    job_durations: List[float] = Field(
        ..., description="Durations (months) of each previous job", examples=[[12, 18, 24]]
    )
    companies: int = Field(..., description="Number of companies worked at", examples=[3])
    experience: float = Field(..., description="Total years of experience", examples=[4.5])
    current_salary: float = Field(..., description="Current salary", examples=[8])
    expected_salary: float = Field(..., description="Expected salary", examples=[12])
    candidate_skills: List[str] = Field(
        ..., description="Skills the candidate has", examples=[["Python", "FastAPI", "PostgreSQL"]]
    )
    company_skills: List[str] = Field(
        ..., description="Skills the company requires", examples=[["Python", "FastAPI", "Git", "Docker"]]
    )


class ScoreBreakdown(BaseModel):
    stability: float
    long_term_probability: float
    long_term_recommendation: str
    monetary: float
    culture: float
    final_score: float


class ScoreResponse(BaseModel):
    rule_based: ScoreBreakdown
    ml_based: ScoreBreakdown
    inputs_used: dict


@app.get("/")
def root():
    return {"status": "ok", "message": "Hiring Recommendation API is running"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/score", response_model=ScoreResponse)
def score_candidate(candidate: CandidateInput):
    if candidate.current_salary <= 0:
        raise HTTPException(status_code=400, detail="current_salary must be greater than 0")
    if not candidate.job_durations:
        raise HTTPException(status_code=400, detail="job_durations must contain at least one value")
    if not candidate.company_skills:
        raise HTTPException(status_code=400, detail="company_skills must contain at least one value")

    avg_tenure = sum(candidate.job_durations) / len(candidate.job_durations)
    salary_jump = ((candidate.expected_salary - candidate.current_salary) / candidate.current_salary) * 100

    # Shared components (identical for both approaches)
    stability = calculate_stability(candidate.job_durations)
    monetary = calculate_monetary_score(candidate.current_salary, candidate.expected_salary)
    culture = calculate_cultural_fit(candidate.candidate_skills, candidate.company_skills)

    # Rule-based long-term prediction + final score
    rule_long_term = predict_long_term(avg_tenure)
    rule_final = generate_final_score(
        stability, rule_long_term["long_term_probability"], monetary, culture
    )

    # ML-based long-term prediction + final score
    ml_long_term = predict_long_term_ml(
        avg_tenure=avg_tenure,
        companies=candidate.companies,
        salary_jump=salary_jump,
        experience=candidate.experience
    )
    ml_final = generate_final_score(
        stability, ml_long_term["long_term_probability"], monetary, culture
    )

    return ScoreResponse(
        rule_based=ScoreBreakdown(
            stability=stability,
            long_term_probability=rule_long_term["long_term_probability"],
            long_term_recommendation=rule_long_term["recommendation"],
            monetary=monetary,
            culture=culture,
            final_score=rule_final
        ),
        ml_based=ScoreBreakdown(
            stability=stability,
            long_term_probability=ml_long_term["long_term_probability"],
            long_term_recommendation=ml_long_term["recommendation"],
            monetary=monetary,
            culture=culture,
            final_score=ml_final
        ),
        inputs_used={
            "avg_tenure_months": round(avg_tenure, 2),
            "salary_jump_percent": round(salary_jump, 2)
        }
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
