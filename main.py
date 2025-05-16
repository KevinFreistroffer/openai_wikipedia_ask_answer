import pandas as pd
import logging
from scipy import spatial
from dotenv import load_dotenv
from openai import OpenAI
import os
import ast
import tiktoken

load_dotenv()

OPEN_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPEN_API_KEY:
    raise ValueError("Missing OpenAI key")

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

EMBEDDING_MODEL = "text-embedding-3-small"
GPT_MODELS = ["gpt-4o", "gpt-4o-mini"]
client = OpenAI(api_key=OPEN_API_KEY)
embedding_path = "data/winter_olympics_2022.csv"
df = pd.read_csv(embedding_path)
# Convert the string representation of embeddings to actual lists
df["embedding"] = df["embedding"].apply(ast.literal_eval)


def strings_ranked_by_relatedness(
    query: str,
    df: pd.DataFrame,
    relatedness_fn=lambda x, y: 1 - spatial.distance.cosine(x, y),
    top_n: int = 100,
) -> tuple[list[str], list[float]]:
    """Returns a list of strings and relatednesses, sorted from most related to least."""
    query_embedding_response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=query,
    )
    query_embedding = query_embedding_response.data[0].embedding
    strings_and_relatednesses = [
        (row["text"], relatedness_fn(query_embedding, row["embedding"]))
        for i, row in df.iterrows()
    ]
    strings_and_relatednesses.sort(key=lambda x: x[1], reverse=True)
    strings, relatednesses = zip(*strings_and_relatednesses)
    return strings[:top_n], relatednesses[:top_n]


def num_tokens(text: str, model: str = GPT_MODELS[0]) -> int:
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))


def query_message(query: str, df: pd.DataFrame, model: str, token_budget: int) -> str:
    """Return a message for GPT, with relevant source texts pulled from a dataframe."""
    strings, relatednesses = strings_ranked_by_relatedness(query, df)
    introduction = 'Use the below articles on the 2022 Winter Olympics to answer the subsequent question. If the answer cannot be found in the articles, write "I could not find an answer."'
    question = f"\n\nQuestion: {query}"
    message = introduction
    for string in strings:
        next_article = f'\n\nWikipedia article section:\n"""\n{string}\n"""'
        if num_tokens(message + next_article + question, model=model) > token_budget:
            break
        else:
            message += next_article
    return message + question


def ask(
    query: str,
    df: pd.DataFrame = df,
    model: str = GPT_MODELS[0],
    token_budget: int = 4096 - 500,
    print_message: bool = False,
) -> str:
    """Answers a query using GPT and a dataframe of relevant texts and embeddings."""
    message = query_message(query, df, model=model, token_budget=token_budget)
    if print_message:
        print(message)
    messages = [
        {
            "role": "system",
            "content": "You answer questions about the 2022 Winter Olympics.",
        },
        {"role": "user", "content": message},
    ]
    response = client.chat.completions.create(
        model=model, messages=messages, temperature=0
    )
    response_message = response.choices[0].message.content
    return response_message


def main():
    logger.info(f"df: {df}")

    answer = ask(
        "Did any controversies occur at 2022 Winter Olympics?", print_message=True
    )
    logger.info(f"answer: {answer}")
    # examples
    # strings, relatednesses = strings_ranked_by_relatedness("curling gold medal", df, top_n=5)
    # for string, relatedness in zip(strings, relatednesses):
    #     print(f"{relatedness=:.3f}")
    #     print(string)

    # ask(query="What is the gold medal in curling?", print_message=True)


if __name__ == "__main__":
    main()
