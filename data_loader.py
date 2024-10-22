import requests
from dotenv import load_dotenv
import os

load_dotenv()


def retrieve_data():
    """
    Used to retrieve data from dev.to/evanhameed99 to build the vectorstore.
    """
    url = "https://dev.to/api/articles/me"
    headers = {"api-key": os.getenv("DEV_TO_API_KEY")}  # TODO: Put API-KEY as env.

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        res = response.json()
        return res
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")


def extract_content_and_tags(data):

    tags_set = set()
    content = []

    for post in data:
        content.append(post["body_markdown"])
        tags_set.update(post["tag_list"])

    tags_list = list(tags_set)

    return content, tags_list
