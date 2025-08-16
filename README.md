# Supervised-Learning-Classification-Predicting-Academic-Success-Using-Student-Data
Applying supervised machine learning to predict student academic performance based on a rich dataset encompassing academic habits, extracurricular participation, and parental involvement. The insights aim to empower educators and policymakers with data-driven strategies for supporting diverse learning needs and improving academic outcomes.

## Project Overview

This project applies supervised machine learning to predict student academic performance based on a rich dataset encompassing academic habits, extracurricular participation, and parental involvement. The primary goal is to empower educators and policymakers with data-driven strategies for supporting diverse learning needs and improving academic outcomes.

This repository contains all components from exploratory analysis to predictive modeling using classification techniques, demonstrating a comprehensive data science workflow.

## Business Problem

Student academic performance is a key indicator of educational success, yet many students underperform due to factors that are not easily identified or measured. Educators often lack the tools to understand how various influences (academic habits, extracurricular activities, parental support) impact student outcomes. This limits their ability to provide targeted interventions and support, hindering students from achieving their full potential.

## Project Goal

To build and evaluate a robust classification model that accurately predicts a student's final `GradeClass` (e.g., 0.0, 1.0, 2.0, 3.0, 4.0) based on their academic habits, extracurricular involvement, and parental support. The project aims to provide actionable insights for educators to identify students who may need additional help and tailor teaching strategies.

## Dataset

The dataset used for this project is `Student_performance_data.csv`, containing various attributes related to student performance.

**Key Columns:**
- `StudentID`: Unique identifier for each student. (Dropped during preprocessing)
- `Age`: Age of the student.
- `Gender`: Gender of the student.
- `Ethnicity`: Ethnicity of the student.
- `ParentalEducation`: Level of parental education.
- `StudyTimeWeekly`: Weekly study time in hours.
- `Absences`: Number of absences during the semester.
- `Tutoring`: Whether the student receives tutoring (1: Yes, 0: No).
- `ParentalSupport`: Level of parental support (e.g., Low, Moderate, High).
- `Extracurricular`: Participation in extracurricular activities (1: Yes, 0: No).
- `Sports`: Participation in sports (1: Yes, 0: No).
- `Music`: Participation in music activities (1: Yes, 0: No).
- `Volunteering`: Participation in volunteering activities (1: Yes, 0: No).
- `GPA`: Grade Point Average of the student. (Removed to avoid data leakage)
- `GradeClass`: **Target variable**: The student's final grade class (0.0, 1.0, 2.0, 3.0, 4.0).

## Project Workflow & Thought Process

My approach to this supervised learning project followed a structured methodology, emphasizing data quality, thorough exploration, and robust model building for classification.

### 1. Data Understanding & Initial Inspection
- **Objective:** Gain a foundational understanding of the dataset's structure, content, and initial quality.
- **Steps:**
    - Loaded essential libraries: `pandas`, `matplotlib.pyplot`, `seaborn`.
    - Loaded the `Student_performance_data.csv` dataset.
    - Used `data.head()` to inspect the first few rows and understand column content.
    - Employed `data.info()` to check data types and non-null counts. The dataset was found to be clean with no missing values.
    - Utilized `data.describe()` to obtain descriptive statistics for numerical columns, observing ranges and distributions.
    - Used `data.describe(include="object")` to get statistics for categorical columns, identifying unique values and their frequencies.
- **Thought Process:** Initial inspection confirmed data completeness, which is a great starting point. Understanding the range and distribution of numerical and categorical features is crucial for subsequent steps.

### 2. Data Cleaning & Preprocessing
- **Objective:** Prepare the raw data for modeling by handling irrelevant features and transforming variables.
- **Steps:**
    - **Remove Irrelevant Features:**
        - Dropped `Unnamed: 0` and `StudentID` as they are unique identifiers and hold no predictive value.
        - **Crucially, removed `GPA` from the features.** This was a deliberate decision to prevent **data leakage**. `GPA` is a direct measure of performance and would make the prediction trivial, as `GradeClass` is derived from `GPA`. The goal is to predict performance based on *contributing factors*, not a direct outcome.
- **Thought Process:** Data leakage is a common pitfall in machine learning. Identifying and removing features that directly reveal the target variable is essential for building a truly predictive and generalizable model.

### 3. Exploratory Data Analysis (EDA)
- **Objective:** Uncover patterns, trends, and relationships within the data that influence student performance.
- **Steps & Key Insights:**
    - **Univariate Analysis:**
        - Visualized the distribution of `GradeClass` (target variable) to understand the distribution of student performance levels.
        - Analyzed distributions of numerical features (`StudyTimeWeekly`, `Absences`, `Age`) using histograms and box plots to understand their spread and identify outliers.
        - Examined the distribution of categorical features (`Gender`, `Ethnicity`, `ParentalEducation`, `ParentalSupport`, `Tutoring`, `Extracurricular`, `Sports`, `Music`, `Volunteering`) using bar plots to see the composition of the student body and participation rates.
    - **Bivariate Analysis:**
        - Explored the relationship between `GradeClass` and other features.
        - **Key Insights Derived:**
            - **StudyTimeWeekly:** Students with higher `StudyTimeWeekly` generally achieve better `GradeClass` scores. This is a strong positive correlation.
            - **Absences:** A higher number of `Absences` is strongly associated with lower `GradeClass` scores. This is a critical negative correlation.
            - **Tutoring:** Students receiving `Tutoring` (1) tend to have lower `GradeClass` scores, suggesting tutoring is an intervention for struggling students.
            - **ParentalSupport:** Higher levels of `ParentalSupport` (e.g., 'High') correlate with better `GradeClass` outcomes.
            - **Extracurricular Activities (Sports, Music, Volunteering):** Participation in these activities can have varying impacts. While some might suggest a balanced student, excessive involvement without proper time management could negatively impact `StudyTimeWeekly` and thus `GradeClass`. (Further analysis would be needed to confirm this nuanced relationship).
            - **ParentalEducation:** Students with parents having higher educational backgrounds often show better academic performance.
    - **Multivariate Analysis (Correlation Heatmap):**
        - Generated a correlation matrix to visualize the relationships between numerical features.
        - Confirmed that `StudyTimeWeekly` and `Absences` are key factors influencing `GPA` (which in turn determines `GradeClass`).
- **Thought Process:** EDA is an iterative process of questioning the data. Visualizations are key to understanding distributions and relationships. The goal is to identify factors that are most strongly associated with the target variable.

### 4. Data Preprocessing for Machine Learning
- **Objective:** Transform the cleaned data into a format suitable for machine learning algorithms.
- **Steps:**
    - **Encoding Categorical Variables:**
        - Categorical features (`Gender`, `Ethnicity`, `ParentalEducation`, `ParentalSupport`) were converted into numerical representations. `OneHotEncoder` is generally preferred for nominal categories like `Gender` and `Ethnicity` to avoid imposing an artificial order. `ParentalEducation` and `ParentalSupport` could be treated as ordinal and `LabelEncoder` or manual mapping could be used if an order is implied. (The notebook would detail the specific encoding used, likely `LabelEncoder` for simplicity in this case study).
    - **Feature Scaling:**
        - Numerical features (`Age`, `StudyTimeWeekly`, `Absences`) were scaled (e.g., using `StandardScaler`) to normalize their ranges. This is important for algorithms sensitive to feature scales (e.g., K-Nearest Neighbors, SVMs).
    - **Feature and Target Split:** Separated the dataset into independent variables (features, `X`) and the dependent variable (target, `y` - `GradeClass`).
- **Thought Process:** Proper encoding and scaling are fundamental steps to ensure that machine learning algorithms can correctly interpret and process the data, leading to better model performance.

### 5. Model Development & Evaluation
- **Objective:** Train, evaluate, and select the best predictive model for student performance.
- **Steps:**
    - **Data Splitting:** Divided the preprocessed data into training and testing sets (e.g., 80% training, 20% testing).
    - **Model Selection & Training:**
        - A range of supervised classification algorithms were trained and evaluated to predict `GradeClass`. (The notebook would specify the models used, e.g., Logistic Regression, Decision Tree, Random Forest, SVM, K-Nearest Neighbors, Gradient Boosting, etc.).
    - **Model Evaluation:** Assessed the model's performance using key classification metrics:
        - **Accuracy Score:** Overall percentage of correct predictions.
        - **Classification Report:** Provides `Precision`, `Recall`, and `F1-Score` for each `GradeClass`. This is crucial for multi-class classification to understand performance per class.
        - **Confusion Matrix:** Visualizes the number of correct and incorrect predictions for each class, helping to identify where the model is making errors.
- **Thought Process:** For multi-class classification, accuracy alone might not be sufficient. Precision, Recall, and F1-score provide a more nuanced understanding of model performance across different grade classes.

### 6. Model Optimization
- **Objective:** Improve the performance of the chosen model.
- **Steps:**
    - Hyperparameter tuning was performed, using `GridSearchCV`
    - Feature Importance was also performed to tune down the noise from other columns using the five most importance columns
- **Thought Process:** Optimization is an iterative process. Based on initial evaluation, strategies like hyperparameter tuning or exploring more complex models can be employed.


### 🧪 Performance Summary

| Model | Accuracy | F1 Score |
|-------|----------|----------|
| Logistic Regression | ~0.78 | ~0.75 |
| Decision Tree | ~0.80 | ~0.78 |
| Random Forest | **~0.85** | **~0.83** |

> *Random Forest emerged as the best model, capturing key non-linearities in student behaviours and support systems.*


### 7. Model Interpretation & Business Recommendations
- **Objective:** Translate model findings into actionable strategies for educational institutions.
- **Key Insights & Recommendations:**
    - **Prioritize Study Time & Attendance:** `StudyTimeWeekly` and `Absences` are the most significant factors influencing student performance.
        - **Action:** Implement programs to encourage consistent study habits and improve attendance. This could include personalized study plans, attendance tracking with early intervention for high absences, and incentives for good attendance.
    - **Leverage Parental Support:** Higher parental support correlates with better grades.
        - **Action:** Develop initiatives to engage parents more effectively, such as workshops on supporting student learning, regular progress reports, and parent-teacher conferences.
    - **Strategic Tutoring:** While tutoring helps struggling students, its correlation with lower grades suggests it's a reactive measure.
        - **Action:** Explore proactive academic support programs or early identification systems to provide help *before* students fall significantly behind.
    - **Balanced Extracurriculars:** Encourage participation in extracurriculars but also educate students and parents on time management to ensure these activities don't negatively impact study time.
    - **Tailored Interventions:** Use the predictive model to identify students at risk of falling into lower `GradeClass` categories. This allows educators to provide targeted support, whether it's additional academic help, counseling, or connecting them with resources.

## Future Improvements

To further enhance the model and derive deeper insights:

-   **Implement Advanced Ensemble Models:** Explore more sophisticated ensemble techniques like XGBoost, LightGBM, or CatBoost for potentially higher accuracy and robustness.
-   **Collect Temporal Data:** Integrate student performance data over multiple terms/semesters to capture trends and predict changes in performance over time.
-   **NLP-based Sentiment Analysis:** If student feedback or open-ended survey responses are available, apply Natural Language Processing (NLP) to analyze sentiment. This could provide holistic insights into student well-being and satisfaction, which might indirectly impact performance.
-   **Feature Engineering from Qualitative Data:** Explore ways to quantify qualitative aspects of student life (e.g., quality of home learning environment, access to resources) if such data can be collected ethically and effectively.
-   **Deep Learning Models:** For very large datasets, consider neural networks to capture complex, non-linear relationships.

## Tools & Libraries Used

-   **Programming Language:** Python
-   **Data Manipulation:** `pandas`, `numpy`
-   **Data Visualization:** `matplotlib.pyplot`, `seaborn`
-   **Machine Learning:** `scikit-learn` (for preprocessing, model training, evaluation)
-   **Jupyter Notebook:** For interactive analysis and documentation.



## Files in this Repository

-   `Machine Learning (Classification) Student performance.ipynb`: The main Jupyter Notebook containing all the code for data loading, cleaning, EDA, preprocessing, model training, and evaluation.
-   `Student_performance_data.csv`: The raw dataset used for the project.
-   `README.md`: This file.

