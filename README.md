# Preserving Privacy in XAI

Can we make machine learning models explainable without compromising patient privacy? This project investigates how Differential Privacy (DP) and Federated Learning (FL) affect model performance, explainability, and resilience to data extraction attacks — using real clinical data from the MIMIC dataset.

📄 **Thesis:** [Master thesis - Benedikt Hornig.pdf](Master%20thesis%20-%20Benedikt%20Hornig.pdf) (Maastricht University, 2024)  
🏛️ **Conducted at:** Fraunhofer IOSB / Karlsruhe Institute of Technology

## Research Question

Explainability techniques (XAI) can expose model internals, potentially leaking private training data. Privacy-preserving techniques like DP and FL protect data but may degrade model quality and explainability. This thesis asks: **Is there a fundamental trade-off between privacy and explainability, or can we have both?**

## What This Project Does

1. **Trains logistic regression and neural network models** on the MIMIC clinical dataset with varying configurations of DP privacy budgets (ε) and FL client numbers.
2. **Measures model performance** (accuracy, AUC) under each privacy configuration.
3. **Evaluates explainability** using feature attribution methods to assess whether privacy techniques distort explanations.
4. **Tests privacy resilience** by running Membership Inference Attacks (MIA) against each model configuration.

## Key Findings

- **DP's impact is model-dependent**: neural networks are more affected than logistic regression.
- **FL generally maintains performance** but shows diminishing effects beyond a certain client threshold.
- **Explainability remains largely intact** under both DP and FL; combining the two can actually mitigate some of DP's negative effects.
- **FL reduces MIA attack precision**, enhancing privacy, while DP alone does not significantly alter MIA success rates.
- **No clear universal trade-off** between privacy and explainability — the relationship is highly model-dependent.

## Repository Structure

```
├── experiments/          # Experiment scripts and configurations
├── requirements.txt      # Python dependencies
└── Master thesis - Benedikt Hornig.pdf  # Full thesis
```

## Citation

```
@mastersthesis{hornig2024privacy,
  title={Preserving Privacy in XAI: An Exploration of the Interplay between Differential Privacy and Federated Learning},
  author={Hornig, Benedikt},
  school={Maastricht University},
  year={2024}
}
```
