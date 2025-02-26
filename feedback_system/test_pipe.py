import sys
import os
# Ensure project root is in sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)  # Use insert(0, ...) to prioritize it

from utils.config import update_config
from alphapose.scripts.demo_inference import run_inference


def main():
    # Load config
    run_args = update_config("./feedback_system/pipe_test.yaml") # TODO Testing set up fix for full pipeline
    output_path, results_list = run_inference(run_args)
    print(output_path)
    print(len(results_list))






if __name__ == '__main__':
    main()