import numpy as np
from sklearn.metrics import roc_curve

def find_optimal_threshold_for_far(y_true, y_pred, target_far=0.0047):  # 0.47%
    """Encuentra el threshold que produce FAR cercano a target_far"""
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    far_values = fpr  # En evaluación binaria, FPR = FAR para clase positiva
    
    # Encontrar threshold que dé FAR más cercano a target
    idx = np.argmin(np.abs(far_values - target_far))
    return thresholds[idx], far_values[idx], tpr[idx]

# Usar en infer_func después de obtener pred_interp
optimal_thresh, achieved_far, tpr_at_thresh = find_optimal_threshold_for_far(
    list(gt), pred_interp, target_far=0.0047
)
print(f"Optimal threshold: {optimal_thresh:.4f}, FAR: {achieved_far:.4f}, TPR: {tpr_at_thresh:.4f}")