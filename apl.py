import datetime
import re
import requests
from newsapi import NewsApiClient

# Replace the API keys here directly or use environment variables
NEWS = 'e6d882611c8b46c0abe512bacd551144'  # Replace with your NewsAPI key
API_KEY = '37b8d347a274e6b7599dfdb037c12d3b'
TMDB = 'your_tmdb_api_key'  # Replace with your TMDB API key

news = NewsApiClient(api_key=NEWS)


def get_ip(_return=False):
    try:
        response = requests.get(f'http://ip-api.com/json/').json()
        if _return:
            return response
        else:
            return f'Your IP address is {response["query"]}'
    except KeyboardInterrupt:
        return None
    except requests.exceptions.RequestException:
        return None

def get_joke():
    try:
        joke = requests.get('https://v2.jokeapi.dev/joke/Any?format=txt').text
        return joke
    except KeyboardInterrupt:
        return None
    except requests.exceptions.RequestException:
        return None
import requests
import re

def get_new():
    print("ok")
    categories = ['business', 'entertainment', 'general', 'health', 'science', 'sports', 'technology']
    all_important_news = []
    print("ok")
    for category in categories:
        try:
            important_news = []
            print(f"\nFetching top headlines for category: {category}")
            top_headlines = news.get_top_headlines(country="us", category=category)
            
            # Loop through articles and filter important headlines
            for i in range(min(5, len(top_headlines['articles']))):
                title = top_headlines['articles'][i]['title']
                clean_title = re.sub(r'[|-] [A-Za-z0-9 |:.]*', '', title).replace("’", "'")
                if "breaking" in clean_title.lower() or "urgent" in clean_title.lower():
                    important_news.append(clean_title)
            
            # If important news found, add it to all_important_news
            if important_news:
                print(f"\nImportant {category.capitalize()} News Headlines:")
                for headline in important_news[:2]:  # Speak only the first 2 important headlines from each category
                    print(f"- {headline}")
                    all_important_news.append(headline)  # Accumulate important headlines
        except KeyboardInterrupt:
            print("Process interrupted. Exiting...")
            return None
        except requests.exceptions.RequestException as e:
            print(f"Error fetching news for category {category}: {e}")
            continue  # Continue with the next category if one fails

    # Return all the important news headlines collected
    if all_important_news:
        return all_important_news
    else:
        print("No important news found.")
        return None


def get_weather(city=''):
    if not city:
        city = 'Nandyal'  # You can replace this with your city or logic to detect location
    
    # Make API call to OpenWeather
    response = requests.get(f'https://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric').json()
    print(f"API Response: {response}")
    
    if response.get('cod') != 200:
        return "Sorry, I couldn't fetch the weather information."

    # Format weather response
    weather = f'It\'s {response["main"]["temp"]}° Celsius and {response["weather"][0]["main"]}. ' \
               f'It feels like {response["main"]["feels_like"]}° Celsius.\n' \
               f'Wind speed: {round(response["wind"]["speed"] * 3.6, 2)} km/h.\n' \
               f'Visibility: {int(response["visibility"] / 1000)} km.'
    return weather

def get_popular_movies():
    try:
        response = requests.get(f"https://api.themoviedb.org/3/movie/popular?api_key={TMDB}&region=IN&sort_by=popularity.desc&"f"primary_release_year={datetime.date.today().year}").json()
    except requests.exceptions.RequestException:
        return None
    try:
        print()
        for movie in response["results"]:
            title = movie['title']
            print(title)
    except KeyError:
        return None

def get_popular_tvseries():
    try:
        response = requests.get(f"https://api.themoviedb.org/3/tv/popular?api_key={TMDB}&region=IN&sort_by=popularity.desc&"
                                f"primary_release_year={datetime.date.today().year}").json()
    except requests.exceptions.RequestException:
        return None
    try:
        print()
        for show in response["results"]:
            title = show['name']
            print(title)
    except KeyError:
        return None
