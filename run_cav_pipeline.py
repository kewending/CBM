import argparse
import pandas as pd
from pathlib import Path
import pickle

from cav import config
from cav.train import train_all_probes, train_probe
from cav.validate import run_validation_report

def main():
    parser = argparse.ArgumentParser(description="Run CAV Pipeline")
    parser.add_argument("--skip_training", action="store_true", help="Load saved probes, skip retraining")
    parser.add_argument("--concept", type=str, help="Train/validate only one concept")
    args = parser.parse_args()
    
    print("--- CAV Pipeline ---")
    
    # 1. Load data
    if not config.TRAIN_CSV.exists() or not config.TEST_CSV.exists():
        print(f"Error: {config.TRAIN_CSV} or {config.TEST_CSV} not found.")
        return
        
    train_df = pd.read_csv(config.TRAIN_CSV)
    test_df = pd.read_csv(config.TEST_CSV)
    df = pd.concat([train_df, test_df], ignore_index=True)
    
    # Filter valid transcripts
    df = df[df['par_transcript'].notna()]
    transcripts = df['par_transcript'].tolist()
    labels = df['label'].tolist()
    
    print(f"Loaded {len(transcripts)} transcripts.")
    
    # 2. Train or load probes
    probes = []
    if args.skip_training:
        print("Skipping training, loading existing probes...")
        for p_file in config.PROBE_OUT_DIR.glob("*.pkl"):
            if args.concept and p_file.stem != args.concept:
                continue
            with open(p_file, 'rb') as f:
                probes.append(pickle.load(f))
    else:
        if args.concept:
            probes = [train_probe(args.concept)]
        else:
            probes = train_all_probes()
            
    if not probes:
        print("No probes found or trained.")
        return
        
    # 3. Validation Report (which also builds bottleneck matrix)
    run_validation_report(probes, transcripts, labels)
    
    print("\n--- CAV Pipeline Complete ---")
    print(f"Concepts evaluated : {len(probes)}")
    reliable = sum(1 for p in probes if p['cv_accuracy'] >= config.MIN_CV_ACCURACY)
    print(f"Reliable probes    : {reliable}  (CV accuracy >= {config.MIN_CV_ACCURACY})")
    print(f"Flagged probes     : {len(probes) - reliable}")
    print(f"Bottleneck matrix  : saved to {config.BOTTLENECK_OUT}")
    print(f"Next step          : train CBM classifier on {config.BOTTLENECK_OUT}")

if __name__ == "__main__":
    main()
