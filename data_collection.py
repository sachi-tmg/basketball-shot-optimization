import os
import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import random
import csv
from tqdm import tqdm

input_csv = "links/playoff_pbp_links_2015_2024.csv"
output_csv = "data/nba_playoff_shots_2015_2024.csv"
os.makedirs("data", exist_ok=True)
os.makedirs("links", exist_ok=True)

links_df = pd.read_csv(input_csv)
all_links = links_df.iloc[:, 0].dropna().tolist()

scraped_links = set()
if os.path.exists(output_csv):
    try:
        existing_df = pd.read_csv(output_csv)
        if 'source_url' in existing_df.columns:
            scraped_links = set(existing_df['source_url'].unique())
    except Exception as e:
        print(f"Existing output read error: {e}")

# CSV fieldnames
fieldnames = ['period', 'time_left', 'team', 'desc', 'score', 'source_url']
if not os.path.exists(output_csv):
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

headers = {
    "User-Agent": "Mozilla/5.0",
    "Referer": "https://www.basketball-reference.com/"
}

shot_phrases = ['makes', 'misses', 'layup', 'jumper', 'dunk', '3-pt', 'free throw']

for link in tqdm(all_links, desc="Scraping all PBP links"):
    if link in scraped_links:
        continue

    try:
        for attempt in range(3):
            try:
                resp = requests.get(link, headers=headers, timeout=10)
                if resp.status_code == 200:
                    break
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(2.5)
        else:
            print(f"Failed after 3 attempts: {link}")
            continue

        soup = BeautifulSoup(resp.text, 'html.parser')
        pbp_table = soup.find('table', {'id': 'pbp'})
        if not pbp_table:
            print(f"No PBP table for {link}")
            continue

        period = None
        game_shots = []
        last_score = "0-0"

        for row in pbp_table.find_all('tr'):
            th = row.find('th')
            if th and 'colspan' in th.attrs:
                th_text = th.get_text(strip=True).lower()
                if '1st' in th_text: period = 1
                elif '2nd' in th_text: period = 2
                elif '3rd' in th_text: period = 3
                elif '4th' in th_text: period = 4
                elif 'ot' in th_text:
                    ot_num = 1
                    import re
                    ot_match = re.search(r'(\d+)[a-z]{2} ot', th_text)
                    if ot_match:
                        ot_num = int(ot_match.group(1))
                    period = 4 + ot_num
                continue

            cols = row.find_all('td')
            if not cols or len(cols) < 6:
                continue

            # time left and score 
            time_left = cols[0].get_text(strip=True)
            score_cell = cols[3].get_text(strip=True)
            if score_cell and '-' in score_cell and all(x.isdigit() for x in score_cell.split('-')):
                last_score = score_cell

            # LEFT TEAM event (col 1)
            left_desc = cols[1].get_text(strip=True)
            if left_desc and any(word in left_desc.lower() for word in shot_phrases):
                shot_data = {
                    'period': period,
                    'time_left': time_left,
                    'team': 'left',
                    'desc': left_desc,
                    'score': last_score,
                    'source_url': link
                }
                game_shots.append(shot_data)

            # RIGHT TEAM event (col 5)
            right_desc = cols[5].get_text(strip=True)
            if right_desc and any(word in right_desc.lower() for word in shot_phrases):
                shot_data = {
                    'period': period,
                    'time_left': time_left,
                    'team': 'right',
                    'desc': right_desc,
                    'score': last_score,
                    'source_url': link
                }
                game_shots.append(shot_data)

        #
        if game_shots:
            with open(output_csv, 'a', encoding='utf-8', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writerows(game_shots)

        # scraping
        time.sleep(random.uniform(1, 3))

    except Exception as e:
        print(f"Error processing {link}: {e}")

if os.path.exists(output_csv):
    df_preview = pd.read_csv(output_csv)
    pd.set_option("display.max_columns", None)
    print(df_preview.head(10))

print("Scraping complete.")
