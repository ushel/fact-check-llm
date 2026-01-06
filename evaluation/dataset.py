from langsmith import Client

QUESTIONS = [
    "what's the weather in sf",
    "whats the weather in san fran",
    "whats the weather in tangier"
]

ANSWERS = [
    "It's 60 degrees and foggy.",
    "It's 60 degrees and foggy.",
    "It's 90 degrees and sunny.",
]

def create_dataset(name: str = "weather agent"):
    client = Client()

    # 🔐 STEP 1: list datasets (SDK-safe)
    datasets = client.list_datasets()
    for ds in datasets:
        if ds.name == name:
            return ds

    # 🔐 STEP 2: create only if not found
    dataset = client.create_dataset(name)

    for q, a in zip(QUESTIONS, ANSWERS):
        client.create_example(
            inputs={"question": q},
            outputs={"answer": a},
            dataset_id=dataset.id
        )

    return dataset
