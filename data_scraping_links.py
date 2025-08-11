import os
import requests
from bs4 import BeautifulSoup
import pandas as pd
from tqdm import tqdm

output_dir = "links"
output_file = "playoff_pbp_links_2015_2024.csv"
os.makedirs(output_dir, exist_ok=True)  

def get_playoff_game_links(year):
    """Returns list of all Playoff PBP URLs for a given year"""
    base_url = f"https://www.basketball-reference.com/playoffs/NBA_{year}_games.html"
    res = requests.get(base_url)
    soup = BeautifulSoup(res.content, "html.parser")

    links = []
    for a in soup.select("td[data-stat='box_score_text'] a"):
        href = a.get("href")
        if href and "boxscores" in href:
            pbp_href = href.replace("/boxscores/", "/boxscores/pbp/")
            links.append("https://www.basketball-reference.com" + pbp_href)
    return links

all_pbp_links = []
for yr in tqdm(range(2015, 2025), desc="Scraping PBP links"):
    try:
        links = get_playoff_game_links(yr)
        all_pbp_links.extend(links)
    except Exception as e:
        print(f"Failed for {yr}: {e}")

df_links = pd.DataFrame({"pbp_url": all_pbp_links})
full_path = os.path.join(output_dir, output_file)
df_links.to_csv(full_path, index=False)
pd.set_option("display.max_colwidth", None)
print(df_links.head(10))
print(f"Saved {len(all_pbp_links)} links to {full_path}")

