#!/usr/bin/env python3
import os
import re
import json
import argparse
import pandas as pd
from datetime import datetime


def read_current_leaderboard(leaderboard_file):
    """Read the current leaderboard table from the markdown file."""
    if not os.path.exists(leaderboard_file):
        # Return empty DataFrame if leaderboard doesn't exist yet
        return pd.DataFrame(columns=[
            'Username', 'Model Size (MB)', 'Size Score', 'Latency (ms)',
            'Latency Score', 'Accuracy (%)', 'Accuracy Score', 'Total Score',
            'Submission Date'
        ])
    
    with open(leaderboard_file, 'r') as f:
        lines = f.readlines()
    
    # Find the leaderboard table
    table_start = -1
    table_end = -1
    for i, line in enumerate(lines):
        if '| Rank | Username' in line:
            table_start = i + 2  # Skip header row and separator row
        if table_start > 0 and line.strip() == '' and i > table_start:
            table_end = i
            break
    
    if table_start == -1:
        return pd.DataFrame(columns=[
            'Username', 'Model Size (MB)', 'Size Score', 'Latency (ms)',
            'Latency Score', 'Accuracy (%)', 'Accuracy Score', 'Total Score',
            'Submission Date'
        ])
    
    if table_end == -1:
        table_end = len(lines)
    
    # Parse table rows
    table_rows = []
    for i in range(table_start, table_end):
        if not lines[i].strip():
            continue
        
        # Skip the rank number and extract fields
        fields = [field.strip() for field in lines[i].strip().split('|')[2:-1]]
        
        if len(fields) >= 8:  # Ensure there are at least 8 fields
            row = {
                'Username': fields[0],
                'Model Size (MB)': float(fields[1]),
                'Size Score': float(fields[2]),
                'Latency (ms)': float(fields[3]),
                'Latency Score': float(fields[4]),
                'Accuracy (%)': float(fields[5]),
                'Accuracy Score': float(fields[6]),
                'Total Score': float(fields[7]),
                'Submission Date': fields[8] if len(fields) > 8 else 'N/A'
            }
            table_rows.append(row)
    
    return pd.DataFrame(table_rows)


def read_raw_results(leaderboard_file):
    """Read the raw results from the detailed archive in the markdown file."""
    if not os.path.exists(leaderboard_file):
        return pd.DataFrame(columns=[
            'Username', 'Model Size (MB)', 'Latency (ms)', 'Accuracy (%)',
            'Total Score', 'Submission Date', 'Notes'
        ])
    
    with open(leaderboard_file, 'r') as f:
        content = f.read()
    
    # Extract the raw results table from the <details> section
    raw_table_match = re.search(r'<details>.*?\n\n(.*?)\n\n</details>', content, re.DOTALL)
    if not raw_table_match:
        return pd.DataFrame(columns=[
            'Username', 'Model Size (MB)', 'Latency (ms)', 'Accuracy (%)',
            'Total Score', 'Submission Date', 'Notes'
        ])
    
    raw_table = raw_table_match.group(1)
    lines = raw_table.strip().split('\n')
    
    # Find the raw results table
    table_start = -1
    table_end = len(lines)
    for i, line in enumerate(lines):
        if '| Username |' in line:
            table_start = i + 2  # Skip header row and separator row
            break
    
    if table_start == -1:
        return pd.DataFrame(columns=[
            'Username', 'Model Size (MB)', 'Latency (ms)', 'Accuracy (%)',
            'Total Score', 'Submission Date', 'Notes'
        ])
    
    # Parse table rows
    table_rows = []
    for i in range(table_start, table_end):
        if not lines[i].strip():
            continue
        
        fields = [field.strip() for field in lines[i].strip().split('|')[1:-1]]
        
        if len(fields) >= 6:  # Ensure there are at least 6 fields
            row = {
                'Username': fields[0],
                'Model Size (MB)': float(fields[1]),
                'Latency (ms)': float(fields[2]),
                'Accuracy (%)': float(fields[3]),
                'Total Score': float(fields[4]),
                'Submission Date': fields[5],
                'Notes': fields[6] if len(fields) > 6 else ''
            }
            table_rows.append(row)
    
    return pd.DataFrame(table_rows)


def update_leaderboard(leaderboard_df, username, model_size, latency, accuracy,
                       size_score, latency_score, accuracy_score, total_score, submission_date):
    """Update the leaderboard with a new submission."""
    
    # Check if user already exists in the leaderboard
    user_idx = leaderboard_df[leaderboard_df['Username'] == username].index
    
    new_row = {
        'Username': username,
        'Model Size (MB)': float(model_size),
        'Size Score': float(size_score),
        'Latency (ms)': float(latency),
        'Latency Score': float(latency_score),
        'Accuracy (%)': float(accuracy),
        'Accuracy Score': float(accuracy_score),
        'Total Score': float(total_score),
        'Submission Date': submission_date
    }
    
    if len(user_idx) > 0:
        # User exists, update if new score is better
        existing_score = float(leaderboard_df.loc[user_idx[0], 'Total Score'])
        if float(total_score) > existing_score:
            leaderboard_df.loc[user_idx[0]] = new_row
    else:
        # User doesn't exist, add new row
        leaderboard_df = pd.concat([leaderboard_df, pd.DataFrame([new_row])], ignore_index=True)
    
    # Sort by Total Score (descending) and reset index
    leaderboard_df = leaderboard_df.sort_values('Total Score', ascending=False).reset_index(drop=True)
    
    return leaderboard_df


def update_raw_results(raw_results_df, username, model_size, latency, accuracy, total_score, submission_date):
    """Update the raw results archive with a new submission."""
    
    new_row = {
        'Username': username,
        'Model Size (MB)': float(model_size),
        'Latency (ms)': float(latency),
        'Accuracy (%)': float(accuracy),
        'Total Score': float(total_score),
        'Submission Date': submission_date,
        'Notes': ''
    }
    
    # Always add the new submission to raw results
    raw_results_df = pd.concat([raw_results_df, pd.DataFrame([new_row])], ignore_index=True)
    
    return raw_results_df


def generate_leaderboard_markdown(leaderboard_df, raw_results_df):
    """Generate the full markdown content for the leaderboard file."""
    
    # Create the header
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")
    markdown = f"# ResNet18 Model Compression Challenge Leaderboard\n\n"
    markdown += f"Last updated: {now}\n\n"
    
    # Create the main leaderboard table
    markdown += "| Rank | Username | Model Size (MB) | Size Score | Latency (ms) | Latency Score | Accuracy (%) | Accuracy Score | Total Score | Submission Date |\n"
    markdown += "|------|----------|----------------|------------|--------------|---------------|--------------|----------------|-------------|------------------|\n"
    
    for i, row in leaderboard_df.iterrows():
        markdown += f"| {i+1} | {row['Username']} | {row['Model Size (MB)']:.2f} | {row['Size Score']:.2f} | "
        markdown += f"{row['Latency (ms)']:.2f} | {row['Latency Score']:.2f} | {row['Accuracy (%)']:.2f} | "
        markdown += f"{row['Accuracy Score']:.2f} | {row['Total Score']:.2f} | {row['Submission Date']} |\n"
    
    # Add scoring details
    markdown += "\n## Scoring Details\n\n"
    markdown += "- **Size Score**: Normalized relative to baseline (smaller is better)\n"
    markdown += "- **Latency Score**: Normalized relative to baseline (faster is better)\n"
    markdown += "- **Accuracy Score**: Normalized relative to baseline (higher is better)\n"
    markdown += "- **Total Score**: Weighted sum using formula: (Size_Score × 0.3) + (Latency_Score × 0.3) + (Accuracy_Score × 0.4)\n"
    
    # Add raw results archive
    markdown += "\n## Raw Results Archive\n\n"
    markdown += "<details>\n"
    markdown += "<summary>View all submissions including previous attempts</summary>\n\n"
    
    markdown += "| Username | Model Size (MB) | Latency (ms) | Accuracy (%) | Total Score | Submission Date | Notes |\n"
    markdown += "|----------|----------------|--------------|--------------|-------------|-----------------|-------|\n"
    
    for _, row in raw_results_df.iterrows():
        markdown += f"| {row['Username']} | {row['Model Size (MB)']:.2f} | {row['Latency (ms)']:.2f} | "
        markdown += f"{row['Accuracy (%)']:.2f} | {row['Total Score']:.2f} | {row['Submission Date']} | {row['Notes']} |\n"
    
    markdown += "\n</details>"
    
    return markdown


def main():
    parser = argparse.ArgumentParser(description='Update leaderboard with new submission')
    parser.add_argument('--username', type=str, required=True, help='GitHub username of the participant')
    parser.add_argument('--model_size', type=float, required=True, help='Model size in MB')
    parser.add_argument('--latency', type=float, required=True, help='Model latency in ms')
    parser.add_argument('--accuracy', type=float, required=True, help='Model accuracy in %')
    parser.add_argument('--total_score', type=float, required=True, help='Total weighted score')
    parser.add_argument('--submission_date', type=str, required=True, help='Submission date')
    parser.add_argument('--leaderboard_file', type=str, required=True, help='Path to the leaderboard markdown file')
    
    args = parser.parse_args()
    
    # Calculate normalized scores
    size_score = 44.7 / args.model_size  # Baseline size / submission size
    latency_score = 30.0 / args.latency  # Baseline latency / submission latency
    accuracy_score = args.accuracy / 40.0  # Submission accuracy / baseline accuracy
    
    # Read current leaderboard and raw results
    leaderboard_df = read_current_leaderboard(args.leaderboard_file)
    raw_results_df = read_raw_results(args.leaderboard_file)
    
    # Update leaderboard and raw results
    leaderboard_df = update_leaderboard(
        leaderboard_df, args.username, args.model_size, args.latency, args.accuracy,
        size_score, latency_score, accuracy_score, args.total_score, args.submission_date
    )
    
    raw_results_df = update_raw_results(
        raw_results_df, args.username, args.model_size, args.latency, args.accuracy,
        args.total_score, args.submission_date
    )
    
    # Generate and save updated markdown
    markdown = generate_leaderboard_markdown(leaderboard_df, raw_results_df)
    
    with open(args.leaderboard_file, 'w') as f:
        f.write(markdown)
    
    print(f"Leaderboard updated at {args.leaderboard_file}")


if __name__ == '__main__':
    main() 