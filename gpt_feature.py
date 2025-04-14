import os
from openai import OpenAI, AsyncOpenAI
import pandas as pd
from tqdm.asyncio import tqdm_asyncio
from tqdm import tqdm
from time import sleep


# Set your OpenAI API key (alternatively, set the environment variable OPENAI_API_KEY)
client = AsyncOpenAI(
    api_key="<your_openai_api_key>",
)

# Prompt template for generating additional tweet context.
PROMPT_TEMPLATE = (
    "Given the following tweet information:\n"
    "Original Tweet: {text}\n"
    "Time of Tweet: {time_of_tweet}\n"
    "Age of User: {age}\n"
    "Country: {country}\n\n"
    "Please analyze the tweet and provide additional context that might be useful for a tweet sentiment analysis task. "
    "Include your best guess as to what the user might be talking about, any relevant events or topics, "
    "and any underlying subtexts that could be inferred from the tweet. "
    "Return your answer as a clear, concise plain text paragraph suitable as a input to a BERT model. "
    "Do no include the original tweet text in your answer. "
    "Include inferred sentiment at the beginning of the response. Options are: positive, negative, neutral. "
)

def load_data(filepath: str) -> pd.DataFrame:
    """Load a CSV file and return the DataFrame."""
    return pd.read_csv(filepath)

async def get_tweet_features(text: str, time_of_tweet: str, age: str, country: str) -> str:
    """Query the OpenAI API to get additional tweet context based on tweet information."""
    prompt = PROMPT_TEMPLATE.format(
        text=text,
        time_of_tweet=time_of_tweet,
        age=age,
        country=country
    )

    response = await client.responses.create(
        model="gpt-4o",
        input=prompt
    )
    additional_features = response.output_text.strip()
    return additional_features
    

async def process_dataframe(df: pd.DataFrame, cache_path) -> pd.DataFrame:
    """
    Process the DataFrame by concurrently querying additional features for each row
    and adding a new column 'additional_context'.
    """
    # Check if the cache file exists and if not empty then load it

    # If it exists, read the cached data
    # and load it into a DataFrame
    # Check if the file is empty
    if os.path.getsize(cache_path) == 0:
        cache = pd.DataFrame()
    else:
        # Read the cached data into a DataFrame
        # and skip the header row
        # if it exists
        # Read the cached data into a DataFrame 
        cache = pd.read_csv(cache_path, header=None)

    results = []
    # extend the results with the cached data
    if not cache.empty:
        results.extend(cache[0].tolist())

    batch_size = 20  # Number of concurrent API calls

    tasks = []
    # Create a task for each API call
    for idx, row in tqdm(df.iterrows()):
        if idx < len(results):
            # Skip already processed rows
            continue
        tweet_text = row['text']
        time_of_tweet = row['Time of Tweet']
        age = row['Age of User']
        country = row['Country']
        tasks.append(get_tweet_features(tweet_text, time_of_tweet, age, country))

        if len(tasks) >= batch_size:
            # Await the batch of tasks
            temp = await tqdm_asyncio.gather(*tasks)
            temp = [x.replace('\n', ' ') for x in temp]
            results.extend(temp)
            tasks = []

            # dump results to a new csv file as we go
            with open(cache_path, 'a') as f:
                pd.DataFrame(temp).to_csv(f, header=False, index=False)

            print("finished batch, sleeping for 10 seconds:", idx)
            sleep(10)  # Optional: Add a small delay to avoid hitting the API rate limit

    # Await any remaining tasks
    if tasks:
        results.extend(await tqdm_asyncio.gather(*tasks))
    
    df['additional_context'] = results
    return df

def save_dataframe(df: pd.DataFrame, filepath: str):
    """Save the DataFrame to a CSV file without the index."""
    df.to_csv(filepath, index=False)

async def main():
    # Load the training and testing data.
    train_filepath = "archive/train.csv"
    test_filepath = "archive/test.csv"
    
    print("Loading training data...")
    train_df = load_data(train_filepath)
    print("Loading testing data...")
    test_df = load_data(test_filepath)
    
    # Process each DataFrame to add additional context.
    print("Processing training data...")
    train_df = await process_dataframe(train_df, "archive/intermediate_results.csv")
    print("Processing testing data...")
    test_df = await process_dataframe(test_df, "archive/intermediate_results_test.csv")
    
    # Save the augmented DataFrames to new files.
    train_output = "archive/train_with_features.csv"
    test_output = "archive/test_with_features.csv"
    
    print(f"Saving processed training data to {train_output} ...")
    save_dataframe(train_df, train_output)
    print(f"Saving processed testing data to {test_output} ...")
    save_dataframe(test_df, test_output)
    print("Processing complete.")

if __name__ == "__main__":
    i = 0
    while i < 100:
        try:
            # Run the main function
            import asyncio
            asyncio.run(main())
            break
        except Exception as e:
            print(f"Error: {e}. Retrying...")
            sleep(5)
    i += 1
    