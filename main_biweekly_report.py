import time
import pandas as pd
import os
import requests
from selenium import webdriver
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup
import concurrent.futures
import plotly.graph_objects as go

import numpy as np
import plotly.graph_objs as go
from sklearn.linear_model import LinearRegression


def extract_text(url):
    driver = None
    try:
        chrome_options = webdriver.ChromeOptions()
        chrome_options.add_argument('--ignore-certificate-errors')
        chrome_options.add_argument('--incognito')
        chrome_options.add_argument("--disable-extensions")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("disable-infobars")
        chrome_options.add_argument("disable-cookies")
        chrome_options.add_argument("--headless")

        driver = webdriver.Chrome(r"C:\github_repos\thales_report\chromedriver.exe", chrome_options=chrome_options)

        driver.get(url)
        driver.implicitly_wait(2)

        element = driver.find_element(By.XPATH, '//img[@src="../imagen/botonimpreso.png"]')
        element.click()
        driver.implicitly_wait(2)

        html = driver.page_source
        soup = BeautifulSoup(html, "html.parser")
        text = soup.get_text()
        print(f"Successfully extracted text from {url}")
        
    except Exception as e:
        print(f"Error extracting text from {url}: {e}")
        text = None
    finally:
        if driver is not None:
            driver.close()
    return text


def convert_currency(amount, from_currency, to_currency):
    url = f"https://api.frankfurter.app/latest?amount={amount}&from={from_currency}&to={to_currency}"
    response = requests.get(url)
    data = response.json()
    return data['rates'][to_currency]

def parallel_extract_text(url):
    return (url, extract_text(url))

# Read Excel file of the CURRENT MONTH (BIWEEKLY)
skiprows = 3
nrows = 15

file_path_current_month = r"C:\github-repos\thales-report\thales_15th_march_2023_complete.xlsx" # PENDING TO CHANGE
df_current_month = pd.read_excel(file_path_current_month, skiprows=skiprows, nrows=nrows)
empty_row_idx = df_current_month.index[df_current_month.isnull().all(axis=1)][0]
df_current_month = df_current_month.loc[:empty_row_idx-1, :]
print(df_current_month.shape)
print(df_current_month.head())

# Read Excel file of the LAST MONTH (BIWEEKLY)
file_path_last_month = r"C:\github_repos\thales_report\project\output_files\3.- march\thales_coverage_march_processed_vf.xlsx" # UPDATED 12/04/2023
df_last_month = pd.read_excel(file_path_last_month, nrows=nrows, sheet_name='Sheet1')
print(df_last_month.shape)
print(df_last_month.head())

# ---- CONTENT EXTRACTION ----
start_time = time.time()
results = []
with concurrent.futures.ThreadPoolExecutor() as executor:
    futures = [executor.submit(parallel_extract_text, url) for url in df_current_month['URL_TESTIGO']]
    for future in concurrent.futures.as_completed(futures):
        results.append(future.result())

elapsed_time = (time.time() - start_time) / 60
print(f"Elapsed time: {elapsed_time:.2f} minutes")

df_current_month['CONTENT'] = [r[1] for r in results]
print(df_current_month.shape)
print(df_current_month.head())

# Apply currency conversion to a specific column
df_current_month['COSTO EURO'] = df_current_month.apply(lambda row: row['COSTO DOLAR'] * convert_currency(1, 'USD', 'EUR'), axis=1)
df_current_month.head()

###------------- KPIs COMPARISON ---------------------#
#last_month = pd.read_excel(r"C:\github-repos\thales-report\thales_february_15th_coverage_processed.xlsx")
#current_month = pd.read_excel(r"C:\github-repos\thales-report\thales_march_15th_coverage_processed.xlsx")

print(df_last_month.shape)
print('\n')
print(df_current_month.shape)


volume_1 = df_last_month.shape[0]
volume_2 = df_current_month.shape[0]

change = round((volume_2 - volume_1) / volume_1 * 100, 2)

print(f"Previous month volume: {volume_1} News")
print(f"Previous month volume: {volume_2} News")
print(f"The percentage change compared last report is: {change}%")

#---
prev_cost = df_last_month['COSTO EURO'].sum()
curr_cost = df_current_month['COSTO EURO'].sum()

percent_change = (curr_cost - prev_cost) / prev_cost * 100

print(f"Previous month cost: {prev_cost:.2f} Euros")
print(f"Current month cost: {curr_cost:.2f} Euros")
print(f"Percentage change compared last report is: {percent_change:.2f}%")


# Write KPIs to a text file
output_dir = r"C:\github_repos\thales_report\project\output_files\4.- april"
file_path = os.path.join(output_dir, 'KPIs Comparison Mar-Apr_Biweekly.txt')

with open(file_path, 'w') as file:
    file.write("VOLUME\n")
    file.write(f"March Volume: {volume_1} News\n")
    file.write(f"April volume: {volume_2} News\n")
    file.write(f"The Percentage Change Compared Last Report Is: {change}%\n\n")
    file.write("COST\n")
    file.write(f"March Total Cost: {prev_cost:.2f} Euros\n")
    file.write(f"April Total Cost: {curr_cost:.2f} Euros\n")
    file.write(f"Percentage Change Compared Last Report: {percent_change:.2f}%\n")

### ------------- PLOTS ---------------------#
"""
Media Plot
"""
media = df_current_month.groupby("MEDIO").size().reset_index(name="counts")
media = media.sort_values(by="counts", ascending=True)
fig = go.Figure(data=[go.Bar(x=media["MEDIO"], y=media["counts"], text=media["counts"], 
                            textposition='auto', marker=dict(color='darkblue'))])

fig.update_layout(title="Number of Records by Medio: Thales Report", xaxis_title="Medio", yaxis_title="Number of Records",
                  plot_bgcolor='white')

fig.show()

"""
Media Type Plot
"""
media_type = df_current_month.groupby("TIPO DE MEDIO").size().reset_index(name="counts")
media_type = media_type.sort_values(by="counts", ascending=True)
fig = go.Figure(data=[go.Bar(x=media_type["TIPO DE MEDIO"], y=media_type["counts"], text=media_type["counts"], 
                            textposition='auto', marker=dict(color='darkred'))])

fig.update_layout(title="Number of Records by Tipo de Medio: Thales Report", xaxis_title="Tipo de Medio", yaxis_title="Number of Records",
                  plot_bgcolor='white')

fig.show()

"""
Tier Plot
"""
import plotly.express as px

tier = df_current_month.groupby("TIER_TEMP").size().reset_index(name="counts")
tier = tier.sort_values(by="counts", ascending=True)

fig = px.bar(tier, x="TIER_TEMP", y="counts", text="counts", color="TIER_TEMP",
             color_discrete_sequence=px.colors.qualitative.Pastel,
             labels={"TIER_TEMP": "Tier", "counts": "Number of Records"})

fig.update_layout(title="Number of Records by Tier: Thales Report", plot_bgcolor='white')

fig.show()


"""
Sentiment Plot
"""
sentiment = df_current_month.groupby("VALORACION").size().reset_index(name="counts")
total = sentiment["counts"].sum()
sentiment["percentage"] = round(sentiment["counts"] / total * 100, 2)
sentiment = sentiment.sort_values(by="counts", ascending=True)

colors = {'NEG -': 'red', 'POS +': 'green', 'NEU =': 'grey', 'SIN': 'orange'}
sentiment['colors'] = sentiment['VALORACION'].apply(lambda x: colors[x])

fig = go.Figure(data=[go.Pie(labels=sentiment["VALORACION"], values=sentiment["counts"], 
                             textinfo='percent+label', marker_colors=sentiment['colors'])])

fig.update_layout(title="Number of Records by Sentimiento: Thales Report", plot_bgcolor='white')

fig.show()


"""
Costo Euro Plot
"""
current_costs = df_current_month.groupby('FECHA')['COSTO EURO'].sum().reset_index()
last_costs = df_last_month.groupby('FECHA')['COSTO EURO'].sum().reset_index()

current_trace = go.Scatter(x=current_costs['FECHA'], y=current_costs['COSTO EURO'], name='Current Month')
last_trace = go.Scatter(x=last_costs['FECHA'], y=last_costs['COSTO EURO'], name='Last Month')

layout = go.Layout(title='Cost Over Time', xaxis_title='Date', yaxis_title='Cost')

fig = go.Figure(data=[last_trace, current_trace], layout=layout)

fig.show()

# Calculate total costs by date for current and last month
current_costs = df_current_month.groupby('FECHA')['COSTO EURO'].sum().reset_index()
last_costs = df_last_month.groupby('FECHA')['COSTO EURO'].sum().reset_index()

current_trace = go.Scatter(x=current_costs['FECHA'], y=current_costs['COSTO EURO'], name='Current Month')
last_trace = go.Scatter(x=last_costs['FECHA'], y=last_costs['COSTO EURO'], name='Last Month')

# Calculate linear regression line for current and last month
current_lr = LinearRegression().fit(np.array(range(len(current_costs))).reshape(-1, 1), current_costs['COSTO EURO'])
last_lr = LinearRegression().fit(np.array(range(len(last_costs))).reshape(-1, 1), last_costs['COSTO EURO'])

# Calculate slope and intercept of linear regression line
current_slope, current_intercept = current_lr.coef_[0], current_lr.intercept_
last_slope, last_intercept = last_lr.coef_[0], last_lr.intercept_

# Calculate predicted values for current and last month
current_trendline = go.Scatter(x=current_costs['FECHA'], y=current_slope*np.array(range(len(current_costs))) + current_intercept, 
                               name='Current Trend', line=dict(color='blue', width=2, dash='dash'))
last_trendline = go.Scatter(x=last_costs['FECHA'], y=last_slope*np.array(range(len(last_costs))) + last_intercept, 
                            name='Last Trend', line=dict(color='green', width=2, dash='dash'))


layout = go.Layout(title='Cost Over Time', xaxis_title='Date', yaxis_title='Cost')
fig = go.Figure(data=[last_trace, current_trace, last_trendline, current_trendline], layout=layout)

fig.show()