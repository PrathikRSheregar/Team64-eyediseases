"""
Traditional ML model factory. Every model is wrapped behind the same
fit / predict / predict_proba interface so training/evaluation code doesn't
need to special-case any of them.
"""
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression


def build_traditional_model(name: str, class_weight="balanced"):
    """class_weight='balanced' by default to handle class imbalance, per spec."""
    if name == "Decision Tree":
        return DecisionTreeClassifier(
            max_depth=12, min_samples_leaf=5, class_weight=class_weight, random_state=42
        )
    if name == "Random Forest":
        return RandomForestClassifier(
            n_estimators=300, max_depth=18, min_samples_leaf=3,
            class_weight=class_weight, random_state=42, n_jobs=-1
        )
    if name == "SVM":
        # probability=True enables predict_proba (needed for ROC-AUC), at the
        # cost of extra training time via internal cross-validation.
        return SVC(
            C=5.0, kernel="rbf", gamma="scale", probability=True,
            class_weight=class_weight, random_state=42
        )
    if name == "KNN":
        return KNeighborsClassifier(n_neighbors=9, weights="distance", n_jobs=-1)
    if name == "Logistic Regression":
        return LogisticRegression(
            max_iter=2000, C=1.0, class_weight=class_weight,
            multi_class="multinomial", random_state=42, n_jobs=-1
        )
    raise ValueError(f"Unknown traditional model: {name}")
