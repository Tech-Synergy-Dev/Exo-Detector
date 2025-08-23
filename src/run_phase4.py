# src/run_phase4.py

import os
import argparse
import logging
import json
import pandas as pd
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_phase4(data_dir="data", top_n=20):
    """
    Loads Phase 3 results, ranks candidates by anomaly score, and saves the final list.
    """
    logger.info("--- Starting Phase 4: Final Candidate Ranking ---")
    data_dir = os.path.abspath(data_dir)
    results_dir = os.path.join(data_dir, "results")
    
    phase3_results_path = os.path.join(results_dir, "phase3_results.json")
    
    # 1. Load the real results from Phase 3
    if not os.path.exists(phase3_results_path):
        logger.error(f"CRITICAL ERROR: Could not find Phase 3 results at: {phase3_results_path}")
        logger.error("Please run Phase 3 successfully before running Phase 4.")
        return None
        
    with open(phase3_results_path, 'r') as f:
        phase3_data = json.load(f)
        
    logger.info(f"Loaded {len(phase3_data['anomaly_scores'])} anomaly scores from Phase 3.")
    
    # 2. Create a DataFrame of every window that was tested
    candidates_df = pd.DataFrame({
        'candidate_id': range(len(phase3_data['anomaly_scores'])),
        'anomaly_score': phase3_data['anomaly_scores'],
        'is_known_transit': phase3_data['true_labels']
    })
    
    # 3. Rank the candidates by their anomaly score to find the best ones
    ranked_candidates = candidates_df.sort_values(by='anomaly_score', ascending=False)
    
    # 4. Select the Top N candidates to present
    top_candidates = ranked_candidates.head(top_n)
    
    logger.info("--- Top 5 Exoplanet Candidates Found ---")
    logger.info("\n" + top_candidates.head(5).to_string())
    
    # 5. Save the final, correct ranked list
    final_results_path = os.path.join(results_dir, "final_ranked_candidates.json")
    top_candidates.to_json(final_results_path, orient='records', indent=2)
    
    summary = {
        "timestamp": datetime.now().isoformat(),
        "status": "Success",
        "total_candidates_ranked": len(ranked_candidates),
        "top_n_saved": top_n,
        "results_file": str(final_results_path)
    }
    
    logger.info("Phase 4 pipeline completed successfully.")
    logger.info(f"Saved final ranked list to: {final_results_path}")
    
    return summary

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Final Candidate Ranking (Phase 4)")
    parser.add_argument('--data-dir', type=str, default='data', help='Directory for data')
    parser.add_argument('--top-n', type=int, default=20, help='Number of top candidates to save')
    args = parser.parse_args()

    run_phase4(data_dir=args.data_dir, top_n=args.top_n)
