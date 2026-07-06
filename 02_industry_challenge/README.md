# Industry Challenge — Material Property Prediction

A public summary of our final team-based industry challenge from the appliedAI ML & MLOps track.

The challenge was carried out with the industry partner [moldflow.eu](https://moldflow.eu) and focused on predicting missing polymer material properties for injection molding simulation workflows.

> **Note:** This folder intentionally contains no source code, notebooks, data, or model artifacts. The full implementation is kept in a private repository because the dataset and challenge materials are tied to an industry partner and are subject to data confidentiality and partner/IP constraints.

---

## Problem

Accurate material data is essential for reliable injection molding simulations. In practice, however, material datasets are often incomplete. Measuring missing material parameters requires specialized laboratory testing, engineering effort, and additional cost.

The objective of the challenge was to build a reproducible machine learning workflow that can estimate missing material properties from the available information in existing material datasets.

The target variables were ten anonymized material-property parameters:

* `f004`
* `f009`
* `f021`
* `f031`
* `f038`
* `f044`
* `f056`
* `f068`
* `f081`
* `f085`

The feature names and semantic meaning of the parameters were intentionally anonymized, so the project had to be approached as a purely data-driven machine learning problem.

---

## Dataset

The dataset consisted of approximately 600 material-level CSV files. Each file represented one material and contained a mix of:

* material-specific metadata,
* data-quality indicators,
* scalar material parameters,
* target material-property parameters,
* sequence-based curve parameters.

Two columns, `f017` and `f090`, contained sequence-based measurement data. Instead of single scalar values, these fields represented curves as sequences of `(x, y)` value pairs.

This made the task more than a standard tabular regression problem: we had to decide how to represent curve information in a way that could be used by classical machine learning models.

---

## Data preparation

We first built a data ingestion and consolidation workflow that read all material CSV files and combined them into a single modeling table.

The preparation steps included:

* parsing raw material files,
* separating target columns from input features,
* converting scalar columns from object-like raw representations into usable numeric features,
* validating missing values and duplicate rows,
* applying mandatory quality filters,
* investigating the optional `f012` high-quality filter,
* keeping curve columns separate for dedicated feature engineering.

A key decision was to **not blindly apply the optional `f012` filter** as the default modeling setup. Although it represented the highest available data-quality level, it also reduced the number of usable samples substantially. In our experiments, the performance gain was limited compared with the amount of data lost, so we treated it as an experimental filter rather than the main baseline.

---

## Feature engineering

We developed feature sets in stages so that every modeling improvement could be measured clearly.

### 1. Scalar features

The first baseline used only scalar material parameters. This gave us a clean starting point and helped identify which targets were already predictable without sequence information.

We compared several scalar feature-selection strategies:

* all available scalar features,
* top-k features by absolute target correlation,
* top-k features by tree-based feature importance,
* target-specific selected feature sets.

This showed that compact, target-specific feature sets often performed as well as or better than using all scalar features.

### 2. Curve features from `f017` and `f090`

The curve columns were then converted into engineered descriptors. Instead of feeding raw sequence data directly into the models, we extracted tabular features that describe curve behavior.

The curve feature engineering included ideas such as:

* curve length,
* x/y summary statistics,
* start and end values,
* min/max values,
* ranges,
* slopes,
* area-like descriptors,
* shape-oriented summary features.

We then compared whether adding curve descriptors improved model quality for each target.

### 3. Combined feature sets

The final experiments combined scalar features with selected curve descriptors from `f017` and `f090`.

We compared setups such as:

* scalar-only,
* scalar + `f017`,
* scalar + `f090`,
* scalar + `f017` + `f090`.

This made it possible to quantify where sequence-based data added predictive value and where it did not.

---

## Modeling approach

The task was treated as a multi-target material-property prediction problem, but model selection was performed target by target. This allowed us to choose the best-performing configuration for each target instead of forcing one global setup across all targets.

We evaluated several model families, including:

* dummy median baselines,
* linear baselines,
* Random Forest,
* ExtraTrees,
* HistGradientBoosting,
* XGBoost,
* LightGBM,
* CatBoost.

Tree-based models and gradient boosting models were the strongest candidates overall. CatBoost and ExtraTrees were especially useful during the final target-specific optimization phase.

---

## Evaluation

We used repeated cross-validation to make the results more stable than a single train/test split.

The main metrics were:

* **MAE** — mean absolute error,
* **RMSE** — root mean squared error,
* **R²** — explained variance,
* **nMAE p95–p05 %** — MAE normalized by the central target range.

The normalized MAE metric was especially helpful because the targets had very different scales. Instead of comparing raw MAE values across targets, we could express the error relative to the central 90% range of each target.

The evaluation process focused on:

* comparing model families,
* comparing feature sets,
* checking whether curve features improved results,
* checking whether the optional quality filter helped,
* selecting the final configuration per target.

---

## Final solution

The final solution was a reproducible machine learning workflow with the following components:

* data ingestion pipeline,
* data cleaning and filtering pipeline,
* scalar feature preparation,
* curve feature extraction for `f017` and `f090`,
* feature-set generation,
* repeated cross-validation evaluation,
* target-specific model comparison,
* final model-selection tables,
* diagnostic plots,
* feature-importance analysis,
* documented findings and recommendations.

The final modeling strategy was target-specific: each target received the best-performing combination of scalar features, curve descriptors, model family, and weighting strategy based on cross-validation results.

---

## Key findings

### 1. Some targets were highly predictable

Certain targets showed very strong predictive performance from the available features. This suggests that the anonymized material parameters contain meaningful structure and that some missing properties can be estimated reliably from existing material data.

### 2. Target-specific modeling mattered

A single global setup was not optimal for every target. Different targets benefited from different feature sets, curve descriptors, and model choices.

The strongest final approach was therefore not one universal model, but a target-specific selection process.

### 3. Curve features added value, but not equally for every target

The sequence-based columns `f017` and `f090` contained useful information, but their value depended on the target.

For some targets, curve descriptors improved performance meaningfully. For others, scalar features already captured most of the predictive signal.

### 4. The optional quality filter was a trade-off

The optional `f012` quality filter improved data consistency but reduced the number of training samples. In our experiments, this trade-off did not clearly justify using the filter as the default setup.

This was an important practical finding: more restrictive quality filtering is not automatically better when the dataset is already small.

### 5. Compact feature sets were often competitive

Using all scalar features was not always the best choice. Target-specific feature sets based on correlation or tree importance often produced comparable or better results with fewer inputs.

This made the models easier to inspect and helped reduce unnecessary feature noise.

---

## What we implemented

Although the implementation is private, the project included the following technical work:

* raw CSV ingestion,
* dataset consolidation,
* duplicate and missing-value analysis,
* data-quality filtering,
* scalar feature preparation,
* sequence parsing for curve columns,
* curve descriptor generation,
* feature-set comparison framework,
* repeated cross-validation evaluation,
* model-family comparison,
* target-specific model optimization,
* weighted training experiments for extreme target values,
* diagnostic visualizations,
* permutation and model-based feature importance,
* final result tables and notebook documentation.

---

## MLOps and engineering practices

The challenge was not only a modeling exercise. We also applied MLOps and ML engineering practices from the previous weeks of the track:

* structured project layout,
* reusable pipeline components,
* automated evaluation loops,
* reproducible experiment configuration,
* consistent metrics,
* documented modeling decisions,
* version-aware development workflow,
* clear separation between exploration and final methodology.

The goal was to move from exploratory notebooks toward a workflow that could be executed and reviewed systematically.

---

## Limitations

The project had several realistic constraints:

* small dataset size after filtering,
* anonymized features with no direct domain interpretation,
* target values with very different numerical scales,
* sequence data requiring custom feature extraction,
* limited time for deeper hyperparameter optimization,
* private industrial data that cannot be published publicly.

Because of these constraints, the final results should be understood as a strong proof of concept and an engineering workflow, not as a fully production-validated material-property prediction product.

---

## Future improvements

Possible next steps include:

* deeper hyperparameter optimization,
* uncertainty estimation for predictions,
* SHAP-based explainability,
* improved curve representations,
* interpolation of curves onto a shared grid,
* dimensionality reduction for sequence features,
* model ensembling,
* more robust target-specific pipelines,
* production deployment design,
* continuous retraining strategy when more material data becomes available.

---

## Summary

This challenge brought together the full ML lifecycle: data ingestion, cleaning, feature engineering, model evaluation, feature importance analysis, and final communication of results.

The main takeaway is that machine learning can surface useful predictive structure in anonymized industrial material data, especially when scalar features, curve descriptors, and target-specific model selection are combined in a reproducible workflow.
