import requests

API_KEY = "AIzaSyBoCfO2wRcNlva1z5rXXhYH6_oxjc8X18k"


def fact_check_api(query: str) -> dict:
    url = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
    params = {"query": query, "key": API_KEY}
    response = requests.get(url, params=params)
    return response.json()


def get_verdict(data: dict) -> str:
    try:
        claims = data.get("claims", [])
        if not claims:
            return "No fact-check found"
        review = claims[0]['claimReview'][0]
        source = review['publisher']['name']
        rating = review['textualRating']
        return f"{source}: {rating}"
    except Exception:
        return "Error in API response"


def check_with_api(statement: str) -> str:
    data = fact_check_api(statement)
    return get_verdict(data)
