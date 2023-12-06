import time
import pandas as pd
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

        
        driver = webdriver.Chrome(r"C:\github_repos\tiktok_app\chromedriver.exe", chrome_options=chrome_options)

        driver.get(url)
        driver.implicitly_wait(2)

        # click on "impreso" button
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

skiprows = 3
df = pd.read_excel(r"C:\github-repos\thales-report\thales_15th_march_2023_complete.xlsx", skiprows=skiprows) # change the path to the current file
empty_row_idx = df.index[df.isnull().all(axis=1)][0]
df = df.loc[:empty_row_idx-1, :]
print(df.shape)
df.head(3)


def parallel_extract_text(url):
    return (url, extract_text(url))

start_time = time.time()

results = []
with concurrent.futures.ThreadPoolExecutor() as executor:
    futures = [executor.submit(parallel_extract_text, url) for url in df['URL_TESTIGO']]
    for future in concurrent.futures.as_completed(futures):
        results.append(future.result())

elapsed_time = (time.time() - start_time) / 60
print(f"Elapsed time: {elapsed_time:.2f} minutes")

df['CONTENT'] = [r[1] for r in results]
print(df.shape)
df.head(3)


def tier_number(tier):
    if tier == 'AAA':
        return 1
    elif tier == 'AA':
        return 2
    elif tier == 'A':
        return 3
    else:
        return None

#df['TIER_TEMP'] = df['PONDERACION'].apply(tier_number)
#print(df.shape)
#df.head()

def usd_to_eur(amount):
    response = requests.get("https://api.exchangerate-api.com/v4/latest/USD")
    data = response.json()
    eur_rate = data["rates"]["EUR"]
    return amount * eur_rate

df["COSTO EURO"] = df["COSTO DOLAR"].apply(usd_to_eur)
print(df.shape)
df.head()


# Define function to convert currency
def convert_currency(amount, from_currency, to_currency):
    url = f"https://api.frankfurter.app/latest?amount={amount}&from={from_currency}&to={to_currency}"
    response = requests.get(url)
    data = response.json()
    return data['rates'][to_currency]



# Apply currency conversion to a specific column
df['COSTO EURO_2'] = df.apply(lambda row: row['COSTO DOLAR'] * convert_currency(1, 'USD', 'EUR'), axis=1)
df.head()

df.to_excel("thales_march_15th_coverage_processed.xlsx", index = False, encoding = 'utf-8', engine='xlsxwriter')




###------------- KPIs COMPARISON ---------------------#
previous_month = pd.read_excel(r"C:\github-repos\thales-report\thales_february_15th_coverage_processed.xlsx")
current_month = pd.read_excel(r"C:\github-repos\thales-report\thales_march_15th_coverage_processed.xlsx")

print(previous_month.shape)
print('\n')
print(current_month.shape)


#volume_1 = previous_month.shape[:1]
#print(volume_1)
#print('\n')
#volume_2 = current_month.shape[:1]
#print(volume_2)


#volume_1 = previous_month.shape[:1]
#print(volume_1)
#print('\n')
#volume_2 = current_month.shape[:1]
#print(volume_2)

#volume_1 = list(volume_1)
#volume_2 = list(volume_2)

#percent_change = ((volume_2[0] - volume_1[0]) / volume_1[0]) * 100

#percent_change = tuple([percent_change])

#print('Percentage Change:', percent_change)


volume_1 = previous_month.shape[0]
volume_2 = current_month.shape[0]

change = round((volume_2 - volume_1) / volume_1 * 100, 2)

print(f"Previous month volume: {volume_1} News")
print(f"Previous month volume: {volume_2} News")
print(f"The percentage change compared last report is: {change}%")

#---
prev_cost = previous_month['COSTO EURO'].sum()
curr_cost = current_month['COSTO EURO'].sum()

percent_change = (curr_cost - prev_cost) / prev_cost * 100

print(f"Previous month cost: {prev_cost:.2f} Euros")
print(f"Current month cost: {curr_cost:.2f} Euros")
print(f"Percentage change compared last report is: {percent_change:.2f}%")


with open('KPIs Comparison Feb-Mar-15th.txt', 'w') as file:
    file.write("VOLUME\n")
    file.write(f"January Volume: {volume_1} News\n")
    file.write(f"February volume: {volume_2} News\n")
    file.write(f"The Percentage Change Compared Last Report Is: {change}%\n\n")
    file.write("COST\n")
    file.write(f"January Total Cost: {prev_cost:.2f} Euros\n")
    file.write(f"February Total Cost: {curr_cost:.2f} Euros\n")
    file.write(f"Percentage Change Compared Last Report: {percent_change:.2f}%\n")



"""
Media Plot
"""
media = current_month.groupby("MEDIO").size().reset_index(name="counts")
media = media.sort_values(by="counts", ascending=True)
fig = go.Figure(data=[go.Bar(x=media["MEDIO"], y=media["counts"], text=media["counts"], 
                            textposition='auto', marker=dict(color='darkblue'))])

fig.update_layout(title="Number of Records by Medio: Thales Report", xaxis_title="Medio", yaxis_title="Number of Records",
                  plot_bgcolor='white')

fig.show()

"""
Media Type Plot
"""
media_type = current_month.groupby("TIPO DE MEDIO").size().reset_index(name="counts")
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

tier = current_month.groupby("TIER_TEMP").size().reset_index(name="counts")
tier = tier.sort_values(by="counts", ascending=True)

fig = px.bar(tier, x="TIER_TEMP", y="counts", text="counts", color="TIER_TEMP",
             color_discrete_sequence=px.colors.qualitative.Pastel,
             labels={"TIER_TEMP": "Tier", "counts": "Number of Records"})

fig.update_layout(title="Number of Records by Tier: Thales Report", plot_bgcolor='white')

fig.show()


"""
Sentiment Plot
"""
sentiment = current_month.groupby("VALORACION").size().reset_index(name="counts")
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
current_costs = current_month.groupby('FECHA')['COSTO EURO'].sum().reset_index()
last_costs = previous_month.groupby('FECHA')['COSTO EURO'].sum().reset_index()

current_trace = go.Scatter(x=current_costs['FECHA'], y=current_costs['COSTO EURO'], name='Current Month')
last_trace = go.Scatter(x=last_costs['FECHA'], y=last_costs['COSTO EURO'], name='Last Month')

layout = go.Layout(title='Cost Over Time', xaxis_title='Date', yaxis_title='Cost')

fig = go.Figure(data=[last_trace, current_trace], layout=layout)

fig.show()






# Calculate total costs by date for current and last month
current_costs = current_month.groupby('FECHA')['COSTO EURO'].sum().reset_index()
last_costs = previous_month.groupby('FECHA')['COSTO EURO'].sum().reset_index()

# Create traces for current and last month
current_trace = go.Scatter(x=current_costs['FECHA'], y=current_costs['COSTO EURO'], name='Current Month')
last_trace = go.Scatter(x=last_costs['FECHA'], y=last_costs['COSTO EURO'], name='Last Month')

# Calculate linear regression line for current and last month
current_lr = LinearRegression().fit(np.array(range(len(current_costs))).reshape(-1, 1), current_costs['COSTO EURO'])
last_lr = LinearRegression().fit(np.array(range(len(last_costs))).reshape(-1, 1), last_costs['COSTO EURO'])

# Calculate slope and intercept of linear regression line
current_slope, current_intercept = current_lr.coef_[0], current_lr.intercept_
last_slope, last_intercept = last_lr.coef_[0], last_lr.intercept_

# Add trend lines to traces
current_trendline = go.Scatter(x=current_costs['FECHA'], y=current_slope*np.array(range(len(current_costs))) + current_intercept, 
                               name='Current Trend', line=dict(color='blue', width=2, dash='dash'))
last_trendline = go.Scatter(x=last_costs['FECHA'], y=last_slope*np.array(range(len(last_costs))) + last_intercept, 
                            name='Last Trend', line=dict(color='green', width=2, dash='dash'))

# Add traces and layout to figure
layout = go.Layout(title='Cost Over Time', xaxis_title='Date', yaxis_title='Cost')
fig = go.Figure(data=[last_trace, current_trace, last_trendline, current_trendline], layout=layout)

fig.show()