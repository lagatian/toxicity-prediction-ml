import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_validate
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, roc_curve, precision_recall_curve)
from sklearn.inspection import permutation_importance
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────
#  1. LOAD DATA
# ─────────────────────────────────────────
df = pd.read_csv('/mnt/user-data/uploads/data-1.csv')
print("=" * 65)
print("  TOXICITY PREDICTION PIPELINE")
print("=" * 65)
print(f"\nDataset shape : {df.shape}")
print(f"Features      : {df.shape[1] - 1}")
print(f"Samples       : {df.shape[0]}\n")

# ─────────────────────────────────────────
#  2. EXPLORATORY DATA ANALYSIS (EDA)
# ─────────────────────────────────────────
print("─" * 65)
print("  SECTION 1 – EXPLORATORY DATA ANALYSIS")
print("─" * 65)

# Class distribution
print("\n[Class Distribution]")
class_counts = df['Class'].value_counts()
for cls, cnt in class_counts.items():
    print(f"  {cls:12s}: {cnt:4d}  ({cnt/len(df)*100:.1f}%)")

# Missing values
print(f"\n[Missing Values] : {df.isnull().sum().sum()} total")

# Basic statistics
X_raw = df.drop('Class', axis=1)
print(f"\n[Feature Statistics]")
desc = X_raw.describe()
print(f"  Mean range  : [{desc.loc['mean'].min():.4f}, {desc.loc['mean'].max():.4f}]")
print(f"  Std range   : [{desc.loc['std'].min():.4f}, {desc.loc['std'].max():.4f}]")
print(f"  Min value   : {desc.loc['min'].min():.4f}")
print(f"  Max value   : {desc.loc['max'].max():.4f}")

# Constant / near-zero variance features
var = X_raw.var()
n_constant = (var == 0).sum()
n_low_var  = (var < 0.01).sum()
print(f"\n[Variance Analysis]")
print(f"  Constant features (var=0)  : {n_constant}")
print(f"  Near-zero variance (<0.01) : {n_low_var}")

# Correlation overview
corr_matrix = X_raw.corr().abs()
upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
high_corr = (upper_tri > 0.95).sum().sum()
print(f"\n[High Correlation (>0.95)] : {high_corr} feature pairs")

# ─────────────────────────────────────────
#  3. EDA VISUALISATIONS
# ─────────────────────────────────────────
sns.set_style("whitegrid")
palette = {"Toxic": "#E74C3C", "NonToxic": "#2ECC71"}

fig = plt.figure(figsize=(20, 22))
fig.suptitle("Toxicity Dataset — Exploratory Data Analysis", fontsize=18, fontweight='bold', y=0.98)
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.4)

# ── Plot 1: Class distribution
ax1 = fig.add_subplot(gs[0, 0])
bars = ax1.bar(class_counts.index, class_counts.values,
               color=[palette[c] for c in class_counts.index], edgecolor='white', width=0.5)
ax1.set_title("Class Distribution", fontweight='bold')
ax1.set_ylabel("Count")
for bar in bars:
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
             f'{int(bar.get_height())}', ha='center', va='bottom', fontweight='bold')

# ── Plot 2: Feature variance distribution (log scale)
ax2 = fig.add_subplot(gs[0, 1])
var_vals = var[var > 0]
ax2.hist(np.log10(var_vals + 1e-10), bins=40, color='steelblue', edgecolor='white', alpha=0.8)
ax2.set_title("Feature Variance Distribution (log₁₀)", fontweight='bold')
ax2.set_xlabel("log₁₀(Variance)")
ax2.set_ylabel("Count")

# ── Plot 3: Class imbalance pie
ax3 = fig.add_subplot(gs[0, 2])
ax3.pie(class_counts.values, labels=class_counts.index, autopct='%1.1f%%',
        colors=[palette[c] for c in class_counts.index],
        startangle=90, wedgeprops=dict(edgecolor='white', linewidth=2))
ax3.set_title("Class Proportion", fontweight='bold')

# ── Plot 4: Top features – ANOVA F-score
le = LabelEncoder()
y = le.fit_transform(df['Class'])  # Toxic=1, NonToxic=0
selector_eda = SelectKBest(f_classif, k=20)
selector_eda.fit(X_raw, y)
scores = pd.Series(selector_eda.scores_, index=X_raw.columns)
top20 = scores.nlargest(20)

ax4 = fig.add_subplot(gs[1, :])
colors_bar = plt.cm.RdYlGn_r(np.linspace(0.1, 0.9, 20))
bars = ax4.barh(top20.index[::-1], top20.values[::-1], color=colors_bar, edgecolor='white')
ax4.set_title("Top 20 Features by ANOVA F-Score", fontweight='bold')
ax4.set_xlabel("F-Score")
ax4.axvline(x=0, color='black', linewidth=0.5)

# ── Plot 5: Feature correlation heatmap (top 15)
ax5 = fig.add_subplot(gs[2, :2])
top15_feats = top20.index[:15].tolist()
corr_top = X_raw[top15_feats].corr()
mask = np.triu(np.ones_like(corr_top, dtype=bool))
sns.heatmap(corr_top, mask=mask, cmap='RdBu_r', center=0, vmin=-1, vmax=1,
            annot=False, linewidths=0.3, ax=ax5, cbar_kws={"shrink": 0.8})
ax5.set_title("Correlation — Top 15 Features", fontweight='bold')
ax5.tick_params(axis='x', rotation=45, labelsize=7)
ax5.tick_params(axis='y', labelsize=7)

# ── Plot 6: Feature distributions (Toxic vs NonToxic) for top 3
ax6 = fig.add_subplot(gs[2, 2])
top3 = top20.index[:3].tolist()
data_plot = df[top3 + ['Class']].copy()
data_melt = data_plot.melt(id_vars='Class', var_name='Feature', value_name='Value')
# Normalise per feature for visibility
for feat in top3:
    mn = data_melt.loc[data_melt['Feature'] == feat, 'Value'].mean()
    sd = data_melt.loc[data_melt['Feature'] == feat, 'Value'].std() + 1e-9
    data_melt.loc[data_melt['Feature'] == feat, 'Value'] = (
        data_melt.loc[data_melt['Feature'] == feat, 'Value'] - mn) / sd

sns.boxplot(data=data_melt, x='Feature', y='Value', hue='Class',
            palette=palette, ax=ax6, width=0.5, fliersize=3)
ax6.set_title("Top 3 Features\n(Standardised)", fontweight='bold')
ax6.set_xlabel("")
ax6.tick_params(axis='x', rotation=20, labelsize=7)
ax6.legend(loc='upper right', fontsize=8)

plt.savefig('/home/claude/eda_report.png', dpi=140, bbox_inches='tight')
plt.close()
print("\n[EDA plots saved]")

# ─────────────────────────────────────────
#  4. PREPROCESSING
# ─────────────────────────────────────────
print("\n" + "─" * 65)
print("  SECTION 2 – PREPROCESSING")
print("─" * 65)

X = X_raw.copy()

# Step 1: Remove constant features
vt = VarianceThreshold(threshold=0)
X_vt = vt.fit_transform(X)
removed_const = X.shape[1] - X_vt.shape[1]
X = pd.DataFrame(X_vt, columns=X.columns[vt.get_support()])
print(f"\nStep 1 – Removed {removed_const} constant features → {X.shape[1]} remaining")

# Step 2: Remove near-zero variance (< 0.01)
vt2 = VarianceThreshold(threshold=0.01)
X_vt2 = vt2.fit_transform(X)
removed_low = X.shape[1] - X_vt2.shape[1]
X = pd.DataFrame(X_vt2, columns=X.columns[vt2.get_support()])
print(f"Step 2 – Removed {removed_low} near-zero variance features → {X.shape[1]} remaining")

# Step 3: Remove highly correlated features (> 0.95)
corr = X.corr().abs()
upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
to_drop_corr = [col for col in upper.columns if any(upper[col] > 0.95)]
X.drop(columns=to_drop_corr, inplace=True)
print(f"Step 3 – Removed {len(to_drop_corr)} highly correlated features (r>0.95) → {X.shape[1]} remaining")

# Step 4: Standard Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
print(f"Step 4 – StandardScaler applied")

print(f"\nFinal preprocessed shape: {X_scaled.shape}")

# ─────────────────────────────────────────
#  5. FEATURE SELECTION
# ─────────────────────────────────────────
print("\n" + "─" * 65)
print("  SECTION 3 – FEATURE SELECTION")
print("─" * 65)

K_FEATURES = 50  # Select top-K features
selector = SelectKBest(f_classif, k=K_FEATURES)
X_selected = selector.fit_transform(X_scaled, y)
selected_features = X_scaled.columns[selector.get_support()].tolist()

f_scores = pd.Series(selector.scores_[selector.get_support()],
                     index=selected_features).sort_values(ascending=False)

print(f"\nSelected {K_FEATURES} features via ANOVA F-test")
print("\nTop 10 selected features:")
for i, (feat, score) in enumerate(f_scores.head(10).items(), 1):
    print(f"  {i:2d}. {feat:<35s}  F={score:.2f}")

# ─────────────────────────────────────────
#  6. MODEL TRAINING + CROSS-VALIDATION
# ─────────────────────────────────────────
print("\n" + "─" * 65)
print("  SECTION 4 – MODEL TRAINING WITH CROSS-VALIDATION")
print("─" * 65)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

models = {
    "Random Forest"         : RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced'),
    "Gradient Boosting"     : GradientBoostingClassifier(n_estimators=150, random_state=42),
    "Logistic Regression"   : LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
    "SVM (RBF)"             : SVC(kernel='rbf', probability=True, class_weight='balanced', random_state=42),
}

results = {}
print()
for name, model in models.items():
    cv_res = cross_validate(model, X_selected, y, cv=cv,
                            scoring=['accuracy', 'f1', 'roc_auc', 'precision', 'recall'],
                            return_train_score=False)
    results[name] = {
        'accuracy' : cv_res['test_accuracy'],
        'f1'       : cv_res['test_f1'],
        'roc_auc'  : cv_res['test_roc_auc'],
        'precision': cv_res['test_precision'],
        'recall'   : cv_res['test_recall'],
    }
    print(f"[{name}]")
    print(f"  Accuracy  : {cv_res['test_accuracy'].mean():.4f} ± {cv_res['test_accuracy'].std():.4f}")
    print(f"  F1-Score  : {cv_res['test_f1'].mean():.4f} ± {cv_res['test_f1'].std():.4f}")
    print(f"  ROC-AUC   : {cv_res['test_roc_auc'].mean():.4f} ± {cv_res['test_roc_auc'].std():.4f}")
    print(f"  Precision : {cv_res['test_precision'].mean():.4f} ± {cv_res['test_precision'].std():.4f}")
    print(f"  Recall    : {cv_res['test_recall'].mean():.4f} ± {cv_res['test_recall'].std():.4f}")
    print()

# Best model
best_model_name = max(results, key=lambda n: results[n]['roc_auc'].mean())
print(f"★  Best model: {best_model_name}  (ROC-AUC = {results[best_model_name]['roc_auc'].mean():.4f})")

# ─────────────────────────────────────────
#  7. RESULTS VISUALISATION
# ─────────────────────────────────────────
fig2, axes = plt.subplots(2, 2, figsize=(16, 13))
fig2.suptitle("Model Evaluation — 5-Fold Stratified Cross-Validation", fontsize=16, fontweight='bold')

model_names = list(results.keys())
metrics = ['accuracy', 'f1', 'roc_auc']
metric_labels = ['Accuracy', 'F1-Score', 'ROC-AUC']
colors_m = ['#3498DB', '#E74C3C', '#2ECC71', '#9B59B6']

# ── Plot A: Metric comparison (grouped bar)
ax = axes[0, 0]
x = np.arange(len(model_names))
w = 0.25
for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
    means = [results[n][metric].mean() for n in model_names]
    stds  = [results[n][metric].std()  for n in model_names]
    ax.bar(x + i*w, means, w, yerr=stds, label=label, capsize=4,
           color=colors_m[i], alpha=0.85, edgecolor='white')
ax.set_xticks(x + w)
ax.set_xticklabels([n.replace(' ', '\n') for n in model_names], fontsize=9)
ax.set_ylim(0.5, 1.05)
ax.set_ylabel("Score")
ax.set_title("Model Comparison", fontweight='bold')
ax.legend(fontsize=9)
ax.axhline(0.9, color='gray', linestyle='--', linewidth=0.8)

# ── Plot B: ROC curves (best model, each fold)
ax = axes[0, 1]
best_model = models[best_model_name]
tprs, aucs_roc = [], []
mean_fpr = np.linspace(0, 1, 100)
for fold, (tr, te) in enumerate(cv.split(X_selected, y)):
    best_model.fit(X_selected[tr], y[tr])
    probs = best_model.predict_proba(X_selected[te])[:, 1]
    fpr, tpr, _ = roc_curve(y[te], probs)
    tprs.append(np.interp(mean_fpr, fpr, tpr))
    aucs_roc.append(roc_auc_score(y[te], probs))
    ax.plot(fpr, tpr, alpha=0.3, linewidth=1, color='#3498DB')

mean_tpr = np.mean(tprs, axis=0)
ax.plot(mean_fpr, mean_tpr, color='#E74C3C', linewidth=2.5,
        label=f'Mean ROC (AUC={np.mean(aucs_roc):.3f})')
ax.plot([0,1],[0,1],'k--', linewidth=1)
ax.fill_between(mean_fpr,
                np.mean(tprs,0) - np.std(tprs,0),
                np.mean(tprs,0) + np.std(tprs,0),
                alpha=0.15, color='#3498DB', label='±1 std')
ax.set_title(f"ROC Curves — {best_model_name}", fontweight='bold')
ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
ax.legend(fontsize=9)

# ── Plot C: Feature Importance (best model = RF/GB → built-in; else permutation)
ax = axes[1, 0]
best_model.fit(X_selected, y)
if hasattr(best_model, 'feature_importances_'):
    imp = pd.Series(best_model.feature_importances_, index=selected_features).nlargest(15)
else:
    perm = permutation_importance(best_model, X_selected, y, n_repeats=10, random_state=42)
    imp = pd.Series(perm.importances_mean, index=selected_features).nlargest(15)

c_imp = plt.cm.viridis(np.linspace(0.2, 0.9, len(imp)))
ax.barh(imp.index[::-1], imp.values[::-1], color=c_imp, edgecolor='white')
ax.set_title(f"Top 15 Feature Importances\n({best_model_name})", fontweight='bold')
ax.set_xlabel("Importance")

# ── Plot D: CV Score Distribution (boxplot)
ax = axes[1, 1]
plot_data = {n: results[n]['roc_auc'] for n in model_names}
bp = ax.boxplot(plot_data.values(), labels=[n.replace(' ', '\n') for n in plot_data.keys()],
                patch_artist=True, medianprops=dict(color='white', linewidth=2))
for patch, color in zip(bp['boxes'], colors_m):
    patch.set_facecolor(color)
    patch.set_alpha(0.75)
ax.set_ylabel("ROC-AUC")
ax.set_title("ROC-AUC Distribution per Model\n(5-Fold CV)", fontweight='bold')
ax.axhline(0.9, color='gray', linestyle='--', linewidth=0.8)
ax.tick_params(axis='x', labelsize=9)

plt.tight_layout()
plt.savefig('/home/claude/model_results.png', dpi=140, bbox_inches='tight')
plt.close()
print("\n[Model result plots saved]")

# ─────────────────────────────────────────
#  8. SUMMARY
# ─────────────────────────────────────────
print("\n" + "=" * 65)
print("  PIPELINE SUMMARY")
print("=" * 65)
print(f"  Original features     : {df.shape[1] - 1}")
print(f"  After variance filter : {X.shape[1] + len(to_drop_corr)}")
print(f"  After corr. filter    : {X.shape[1]}")
print(f"  After feature select  : {K_FEATURES}")
print(f"  Cross-validation      : StratifiedKFold (k=5)")
print(f"\n  {'Model':<25}  {'Accuracy':>9}  {'F1':>9}  {'ROC-AUC':>9}")
print("  " + "-"*55)
for name in model_names:
    acc = results[name]['accuracy'].mean()
    f1  = results[name]['f1'].mean()
    auc = results[name]['roc_auc'].mean()
    star = " ★" if name == best_model_name else "  "
    print(f"  {name:<25}  {acc:9.4f}  {f1:9.4f}  {auc:9.4f}{star}")
print("=" * 65)
print(f"\nBest model → {best_model_name}")
print("Pipeline complete.")
