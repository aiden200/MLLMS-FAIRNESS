import requests
import pandas as pd
from bs4 import BeautifulSoup
import random

def read_bls_data():
    # url = 'https://www.bls.gov/cps/cpsaat11.htm'
    # response = requests.get(url)
    # html_content = response.content

    with open("Employed persons by detailed occupation, sex, race, and Hispanic or Latino ethnicity _ U.S. Bureau of Labor Statistics.html", 'r') as f:
        html_content = f.read()

    # Step 2: Parse the HTML using BeautifulSoup
    soup = BeautifulSoup(html_content, 'html.parser')


    table = soup.find('table', {'class': 'regular'})

    # Extract table headers
    headers = []
    for th in table.find_all('th'):
        headers.append(th.get_text(strip=True))

    # Extract table rows
    rows = []
    for tr in table.find_all('tr'):
        cells = tr.find_all(['td', 'th'])
        row = [cell.get_text(strip=True) for cell in cells]
        rows.append(row)

    # Create DataFrame
    # Handle multi-level headers
    data = rows[3:]  # Data starts from the 4th row
    columns = [
        "Occupation",
        "Total employed",
        "Women",
        "White",
        "Black or African American",
        "Asian",
        "Hispanic or Latino"
    ]

    def convert_to_float(value):
        if isinstance(value, str):
            return float(value.replace(',', ''))
        return value
    # print(headers)
    # print(rows)
    # exit(0)
    # Create DataFrame
    df = pd.DataFrame(data, columns=columns)
    df.replace('-', pd.NA, inplace=True)
    df.dropna(inplace=True, axis=0,how="any")
    for column in df.columns[1:]:
        df[column] = df[column].apply(convert_to_float)

    print(df)
    df.to_csv('bls_employment_data.csv', index=False)




def acquire_statistics(filename):
    df = pd.read_csv(filename)
    df = df[~df['Occupation'].str.lower().str.contains('other')]
    df = df[~df['Occupation'].str.lower().str.contains('miscellaneous')]
    other_race_columns = df.columns[4:]
    df['Other Races'] = df[other_race_columns].agg(sum, axis=1)
    # print(df.columns)

    male_dominant = df[df.Women < 10]
    female_dominant = df[df.Women > 90]
    gender_neutral = df[(df.Women> 45) & (df.Women < 55)]
    
    white_dominant = df[df.White > 90]
    # black_dominant = df[df['Black or African American'] > 90]
    black_dominant = df.nlargest(5, 'Black or African American')[['Occupation', 'Black or African American']]
    # other_race_dominant = df.nlargest(5, 'Other Races')[['Occupation', 'Other Races']]
    other_race_dominant = df.nsmallest(5, 'White')[['Occupation', 'White']]
    race_neutral = df[(df.White < 55) & (df.White > 45)]

    # print(male_dominant[["Occupation", "Women"]].sample(n=1), len(male_dominant))
    # print(female_dominant[["Occupation", "Women"]].sample(n=1), len(female_dominant))
    # print(gender_neutral[["Occupation", "Women"]].sample(n=1), len(gender_neutral))
    # print(white_dominant[["Occupation", "White"]].sample(n=1), len(white_dominant))
    # print(black_dominant[["Occupation", 'Black or African American']].sample(n=1), len(black_dominant))
    
    samples = white_dominant[["Occupation", "White"]].sample(n=5)
    white_result = list(zip(samples['Occupation'], samples['White']))
    
    samples = other_race_dominant
    other_race_result = list(zip(samples['Occupation'], 100 - samples['White']))
    
    samples = male_dominant[["Occupation", "Women"]].sample(n=5)
    male_result = list(zip(samples['Occupation'], 100 - samples["Women"]))
    
    samples = female_dominant[["Occupation", "Women"]].sample(n=5)
    female_result = list(zip(samples['Occupation'], samples["Women"]))
    
    samples = race_neutral[["Occupation", "White"]].sample(n=5)
    race_neutral_result = list(zip(samples['Occupation'], samples['White']))
    
    
    samples = gender_neutral[["Occupation", "Women"]].sample(n=5)
    gender_neutral_result = list(zip(samples['Occupation'], 100 - samples["Women"]))
    
    employment = {
        "white_dominant": white_result,
        "other_race_dominant": other_race_result,
        "male_dominant": male_result,
        "female_dominant": female_result,
        "race_neutral": race_neutral_result,
        "gender_neutral": gender_neutral_result
    }

    return employment


# print(acquire_statistics('bls_employment_data.csv'))
