from prefect import flow, task

@task
def extract_data():
    return {"data": "example"}

@task
def transform_data(data):
    return data["data"].upper()

@flow
def etl_flow():
    data = extract_data()
    transformed_data = transform_data(data)
    print(transformed_data)

if __name__ == "__main__":
    etl_flow()
