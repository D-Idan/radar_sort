# save_model_outputs.py
import pandas as pd
import numpy as np
from pathlib import Path
from utils.util import process_predictions_FFT


def extract_model_predictions(model_outputs, encoder, confidence_threshold=0.2):
    """Extract predictions from model outputs"""
    pred_obj = model_outputs['Detection'].detach().cpu().numpy().copy()[0]
    pred_obj = encoder.decode(pred_obj, 0.05)
    pred_obj = np.asarray(pred_obj)

    predictions = []
    if len(pred_obj) > 0:
        processed_pred = process_predictions_FFT(pred_obj, confidence_threshold=confidence_threshold)

        for i, detection in enumerate(processed_pred):
            predictions.append({
                'detection_id': i,
                'confidence': detection[0],
                'x1': detection[1],
                'y1': detection[2],
                'x2': detection[3],
                'y2': detection[4],
                'x3': detection[5],
                'y3': detection[6],
                'x4': detection[7],
                'y4': detection[8],
                'range_m': detection[9],
                'azimuth_deg': detection[10]
            })

    return pd.DataFrame(predictions)


def save_predictions_to_csv(model_outputs, encoder, sample_id, output_path):
    """Save model predictions for a sample to CSV"""
    df = extract_model_predictions(model_outputs, encoder)
    df['sample_id'] = sample_id

    # Reorder columns
    cols = ['sample_id', 'detection_id', 'confidence', 'range_m', 'azimuth_deg'] + \
           [f'{coord}{i}' for coord in ['x', 'y'] for i in range(1, 5)]
    df = df[cols]

    df.to_csv(output_path, index=False)
    return df


def batch_save_predictions(model_outputs_dict, encoder, output_dir):
    """Save predictions for multiple samples"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_predictions = []

    for sample_id, outputs in model_outputs_dict.items():
        df = extract_model_predictions(outputs, encoder)
        df['sample_id'] = sample_id
        all_predictions.append(df)

    # Combine all predictions
    combined_df = pd.concat(all_predictions, ignore_index=True)
    combined_df.to_csv(output_dir / 'all_predictions.csv', index=False)

    return combined_df