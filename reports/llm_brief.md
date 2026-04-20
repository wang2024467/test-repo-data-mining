# LLM Brief: Heart Disease Risk Modeling

You are a healthcare-aware ML reviewer.
Given the results below, produce:
1) model comparison summary,
2) deployment recommendation,
3) risk/ethics caveats,
4) next 3 experiments to improve recall.

## Metrics
- Model: random_forest
  - ROC-AUC: 0.9216
  - PR-AUC: 0.9301
- Model: logistic_regression
  - ROC-AUC: 0.9045
  - PR-AUC: 0.9106

## Constraints
- This is a screening support prototype, not diagnosis.
- Minimize false negatives where possible.
- Keep recommendations explainable for clinicians.