# Healthcare Operations Intelligence

## Enhanced Healthcare Data Analytics Project

**Student:** Nitin Kohli
**Program context:** IBM SkillsBuild Data Analytics with AI Internship 2026
**Project type:** Synthetic healthcare data analytics + operational intelligence + exploratory ML

> **Healthcare disclaimer:** This project uses synthetic data generated for education and analytics practice. It is not a clinical tool and must not be used for diagnosis, treatment, triage, or medical decision-making.

---

## 1. What this project does

The baseline project analyzed a 150,500-row synthetic healthcare CSV, removed 500 duplicate records, performed EDA, answered key healthcare questions, and evaluated one Decision Tree model. This enhanced version keeps the same source dataset and the important original findings, but turns the work into a reusable analytical product.

### Enhanced capabilities

1. **Auditable data-quality pipeline**

   * Duplicate removal
   * Category normalization
   * Missing-value treatment
   * Impossible oxygen-saturation correction
   * Extreme systolic-BP handling
   * Row-level quality flags

2. **Interactive decision dashboard**

   * Executive KPI view
   * Department workload intelligence
   * Symptom and severity explorer
   * Cohort analysis
   * Data-quality monitoring
   * Evidence-based question/answer panel

3. **Transparent operational index**

   * `Operations Load Index` combines department volume, severe-case share, chronic-condition share, and average symptom duration.
   * The index is explicitly labeled as a custom analytical score, **not a medical or clinical score**.

4. **Stronger ML audit**

   * Majority baseline
   * Logistic Regression
   * Decision Tree
   * Random Forest
   * Accuracy, balanced accuracy, macro-F1, precision and recall
   * Synthetic-feature leakage audit using Cramer's V and symptom lookup accuracy

5. **Reproducible project**

   * Python source modules
   * Notebook
   * Automated artifact generation
   * Tests
   * Cleaned and enriched data outputs
   * DOCX report
   * GitHub-ready README

---

## 2. Why it is materially better than the baseline

| Area                | Baseline           | Enhanced project                                 |
| ------------------- | ------------------ | ------------------------------------------------ |
| Cleaning            | Notebook-only      | Reusable cleaning module + audit flags           |
| EDA                 | Static notebook    | Notebook + interactive dashboard                 |
| Insights            | Fixed findings     | Filterable department/cohort insights            |
| Data quality        | Counts only        | Quality scorecard + row-level issue flags        |
| Department analysis | Volume + averages  | Multi-factor Operations Load Index               |
| ML                  | One Decision Tree  | 4-model benchmark + baselines                    |
| Synthetic leakage   | Manual explanation | Quantified association audit                     |
| Reproducibility     | Notebook           | Source modules + build script + tests            |
| Deliverables        | Notebook/report    | Dashboard + data + notebook + report + artifacts |

---

## 3. Source dataset

Raw file: `data/healthcare_raw_data_150500.csv`

* Raw rows: **150,500**
* Cleaned rows: **150,000**
* Exact duplicates removed: **500**
* Columns: **14**
* Dataset type: synthetic/educational

Original dataset fields:

`patient_id`, `age`, `gender`, `symptoms`, `temperature`, `duration_days`, `bp_systolic`, `bp_diastolic`, `heart_rate`, `oxygen_saturation`, `pain_level`, `symptom_severity`, `chronic_condition`, `department`

---

## 4. Cleaning rules

The pipeline preserves the raw file and creates a separate cleaned dataset.

| Data-quality issue             | Treatment                          |
| ------------------------------ | ---------------------------------- |
| 500 exact duplicates           | Remove duplicate rows              |
| Mixed/variant category casing  | Map values to canonical labels     |
| 6,246 oxygen values > 100      | Cap at 100.0 and record correction |
| 158 extreme systolic BP values | Set to missing, then median-impute |
| Missing numeric values         | Column median                      |
| Missing categorical values     | Column mode                        |
| Auditability                   | Store row-level quality flags      |

The final cleaned dataset contains no missing values.

---

## 5. Enhanced analytical questions

The dashboard and notebook go beyond the original 10 questions. They cover:

* Which departments carry the largest patient volume?
* How concentrated is total workload across departments?
* How does severe-case share differ by department?
* How does chronic-condition share vary by age group?
* Which symptoms are most common?
* Which symptoms are strongly associated with departments in this synthetic dataset?
* How do pain and symptom duration vary across cohorts?
* Where are the biggest data-quality interventions?
* Does an ML model add useful signal beyond a majority baseline?
* What changes when the high-leakage `symptoms` feature is excluded?

---

## 6. Key source findings retained

From the cleaned dataset:

* Average age: **49.05 years**
* Most common symptom: **Chest Pain — 6,629**
* Busiest department: **Dermatology — 24,447**
* Most common severity: **Mild — 45.3%**
* Chronic-condition prevalence: **37.28%**
* Highest average heart rate department: **Cardiology — 87.59 bpm**
* Highest average pain department: **Orthopedics — 6.90/10**
* Lowest average oxygen saturation department: **Pulmonology — 96.25%**
* Largest age group: **61–80 — 47,402**

These findings describe this synthetic dataset only.

---

## 7. ML design

### Target

`department` (9 classes)

### Excluded feature

`symptoms` is excluded from the final predictive benchmark because it is extremely associated with department in this synthetic data-generation process.

The project quantifies this relationship instead of hiding it:

* Symptom-to-department lookup accuracy: **95.63%**
* Cramer's V: **0.969**

This is a dataset-structure finding, not evidence about real healthcare.

### Final benchmark features

* age
* temperature
* duration_days
* bp_systolic
* bp_diastolic
* heart_rate
* oxygen_saturation
* pain_level
* gender
* chronic_condition
* symptom_severity

---
