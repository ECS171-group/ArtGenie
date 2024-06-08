import argparse
import json
import pandas as pd


def load_json(file_path):
    """Load JSON data from file."""
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data


def rank_models(df):
    """Rank models based on mean test score."""
    df['mean_test_score'] = df['mean_test_score'] * \
        100  # Convert to percentage for easier interpretation
    df['rank'] = df['mean_test_score'].rank(ascending=False).astype(int)
    return df


def print_top_models(df, n):
    """Print the top-ranked models."""
    top_models = df.sort_values(by='rank').head(n)
    print(f"Top {n} Models:")
    print(top_models[['param_model_type', 'param_epochs',
          'param_learning_rate', 'param_batch_size', 'mean_test_score', 'rank']])


def analyze_fit_time(df):
    """Perform analysis on mean fit time."""
    mean_fit_time_range = df['mean_fit_time'].max() - df['mean_fit_time'].min()
    print(f"\nRange of Mean Fit Time: {mean_fit_time_range} seconds")


def analyze_score_time(df):
    """Perform analysis on mean score time."""
    mean_score_time_range = df['mean_score_time'].max(
    ) - df['mean_score_time'].min()
    print(f"Range of Mean Score Time: {mean_score_time_range} seconds")


def main(args):
    data = load_json(args.grid_search_result)
    df = pd.DataFrame(data)

    df = rank_models(df)
    print_top_models(df, n=None)
    analyze_fit_time(df)
    analyze_score_time(df)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Grid Search Results Analysis')
    parser.add_argument('--grid_search_result', type=str, help='Path to grid search results JSON file',
                        default="./saved_results/grid_search_results.json")
    args = parser.parse_args()
    main(args)
