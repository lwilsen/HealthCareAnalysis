# HealthCareAnalysis

This is a forked repo of Renu Ghering's HealthCareAnalysis that was created with the intent of testing data science methods on synthetic Medicare and Medicaid data.

I'm going to be exploring this data based on her teaching's as a part of an independent study for my Master's in Biostatistics.

### Abstract

Predicting hospital length of stay (LOS) and disease severity remains a critical challenge in healthcare management. Accurate predictions can assist hospital administrators in optimizing resource allocation, identifying high-risk patients, and improving overall care efficiency. However, LOS prediction has mainly been studied on an ad-hoc basis, making it hard to generalize the findings to other hospitals [(Stone et al., 2022)](https://pmc.ncbi.nlm.nih.gov/articles/PMC9931263/#sec008). Replicating and extending these methodologies on different datasets is essential to assess their generalizability and effectiveness.

This study aims to develop predictive models for LOS (using regression methods) and categorical risk outcomes, including the risk of prolonged hospitalization and severe patient outcomes. Using a synthetic Center for Medicaid and Medicare Services dataset, we will apply rigorous data preprocessing techniques such as cleaning, transformation, and variable selection, potentially incorporating principal component analysis (PCA).

Our modeling approach follows a structured methodology inspired by previous research (Muhlestein et al., 2018, Jain et al, 2024 ) on LOS prediction, incorporating machine learning ensembles, classification and regression, and gradient boosting methods . We will partition data into training and test sets, employ five-fold cross-validation, and explore a broad range of models using Bayesian optimization, grid search, and random search for hyperparameter tuning. The best-performing models will be ensembled using elastic net regression to improve predictive accuracy.

To ensure interpretability in a medical setting, we will analyze variable importance using permutation importance and partial dependence plots. Given the complexity of machine learning models, interpretability is crucial for gaining trust from medical practitioners. Finally, in an ideal validation scenario, we aim to evaluate the model on real Centers for Medicare & Medicaid Services (CMS) data to enhance credibility and real-world applicability. This study contributes to improving hospital efficiency, reducing unnecessary costs, and enhancing patient care through data-driven decision-making.
