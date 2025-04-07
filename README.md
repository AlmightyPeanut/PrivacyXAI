# Preserving Privacy in XAI: An Exploration of the Interplay between Differential Privacy and Federated Learning

The digitization of health data offers significant opportunities for enhancing medical diagnostics
and research, but it also raises concerns about data privacy and security, particularly with the
use of machine learning models. This thesis addresses the critical challenge of balancing model
explainability with data privacy by investigating the effects of Differential Privacy (DP) and
Federated Learning (FL) on machine learning models.
The study employs the MIMIC dataset to train logistic regression and neural network mod-
els, assessing various configurations of DP privacy budgets and FL client numbers. It evaluates
the impact of these privacy-preserving techniques on model performance, explainability, and
resilience to Membership Inference Attacks (MIA). The findings reveal that DP’s impact on per-
formance is model-dependent, with neural networks being more affected than logistic models.
FL generally maintains performance, but has minimal effects beyond a certain client threshold.
Explainability metrics remain largely unaffected by DP and FL, though combining these tech-
niques can mitigate some of DP’s negative effects. Privacy evaluations show that while DP does
not significantly alter MIA success rates, FL reduces attack precision, thus enhancing privacy.
This research suggests that there is no clear trade-off between privacy and explainability,
as the effects are highly model-dependent. These insights still provide actionable guidance for
deploying privacy-preserving machine learning models, ensuring both interpretability and robust
data protection.