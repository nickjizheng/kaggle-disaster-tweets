from disaster_tweet_test_utils import run_cli


MODEL_PATH = None


if __name__ == "__main__":
    run_cli("aux_mc_cemn", default_model_path=MODEL_PATH)
