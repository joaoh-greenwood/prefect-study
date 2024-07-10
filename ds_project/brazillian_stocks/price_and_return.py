# Import Prefect
from prefect import task, flow

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import pandas_datareader.data as web
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from statsmodels.graphics.tsaplots import plot_acf

# Override yfinance
yf.pdr_override()

# Define tasks
@task
def download_data():
    return web.get_data_yahoo('PETR4.SA', period='1y')["Adj Close"]

@task
def calculate_returns(petr):
    return petr.pct_change()

@task
def prepare_data(petr, ret):
    dados = pd.DataFrame()
    dados['Close'] = petr
    dados['retornos'] = ret
    for i in range(1, 6):
        dados[f'Lag{i}'] = dados['retornos'].pct_change(i)
    dados = dados.dropna()
    dados = dados[~dados.isin([np.nan, np.inf, -np.inf]).any(1)]
    return dados

@task
def plot_data(petr, ret, dados):
    plt.figure(figsize=(20, 10))
    petr.plot()
    plt.xlabel('Time - Days')
    plt.ylabel('Price')
    plt.title('Petrobras - PETR4')
    plt.show()
    
    plt.figure(figsize=(20, 10))
    ret.plot()
    plt.xlabel('Time - Days')
    plt.ylabel('Price')
    plt.title('Petrobras - PETR4 Returns')
    plt.show()

    dados.iloc[:, 2:].plot(figsize=(20, 10))
    plt.show()

@task
def train_model(dados):
    Y = dados['retornos']
    X = dados.iloc[:, 2:]
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    model = sm.OLS(y_train, x_train).fit()
    predictions = model.predict(x_test)
    return model.summary()

@task
def plot_acf_data(dados):
    plot_acf(dados['Close'])
    plt.show()
    plot_acf(dados['retornos'])
    plt.show()

# Define flow
@flow(name="Financial Data Analysis")
def process_analysis():
    petr = download_data()
    ret = calculate_returns(petr)
    dados = prepare_data(petr, ret)
    plot_data(petr, ret, dados)
    model_summary = train_model(dados)
    plot_acf_data(dados)

# Execute the flow
if __name__ == "__main__":
    process_analysis()
