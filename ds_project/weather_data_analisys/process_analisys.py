from prefect import task, flow, serve, get_run_logger
from prefect.engine import State
from prefect.client.schemas.actions import StateType
import requests
import asyncio

API_KEY = 'c46b3f66f840089f520cbbf5f9b5b36b'
CITIES = ['São Paulo', 'Curitiba', 'Recife']

@task
def get_weather_data(city):
    url = f'http://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}'
    response = requests.get(url)
    data = response.json()
    if response.status_code != 200:
        raise ValueError(f"Error fetching data for {city}: {data.get('message', 'Unknown error')}")
    return data

@task
def transform_data(weather_data):
    transformed_data = {
        'city': weather_data['name'],
        'temperature': weather_data['main']['temp'],
        'humidity': weather_data['main']['humidity']
    }
    return transformed_data

@task
def submit_data(data):
    logger = get_run_logger()
    logger.info(f'Data submitted: {data}')

@task
def process_city_data(city):
    weather_data = get_weather_data(city)
    transformed_data = transform_data(weather_data)
    submit_data(transformed_data)
    return {
        'city': city
    }

@flow
async def weather_data_pipeline(cities):
    futures = [process_city_data.submit(city) for city in cities]
    results = [future.wait() for future in futures]
    return results

@flow
async def process_analisys():
    results = await weather_data_pipeline(CITIES)
    
    for state in results:
        if state:
            logger = get_run_logger()
            logger.info(f'State: {state}')

if __name__ == '__main__':
    deployment = process_analisys.to_deployment(name="weather-analisys-subflow")
    deployment.apply()
    
    serve(deployment)

# if __name__ == '__main__':
#     asyncio.run(process_analisys())