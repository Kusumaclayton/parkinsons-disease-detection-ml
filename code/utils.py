import numpy as np

def combine_predictions(voice_pred, spiral_pred):
    # Simple logic: majority / average
    final = (voice_pred + spiral_pred) / 2
    return 1 if final >= 0.5 else 0
