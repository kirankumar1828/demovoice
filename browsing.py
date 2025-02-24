import webbrowser
import re
import wikipedia
import websites


def googleSearch(query):
	if 'image' in query:
		query += "&tbm=isch"
	#query = re.sub(r'\b(images|image|search|show|google|tell me about|for)\b', '', query)

	query = query.replace('images', '')
	query = query.replace('image', '')
	query = query.replace('search', '')
	query = query.replace('show', '')
	query = query.replace('google', '')
	query = query.replace('tell me about', '')
	query = query.replace('for', '')
	webbrowser.open("https://www.google.com/search?q=" + query)
	return "Here you go..."

def youtube(query):
	#query = re.sub(r'\b(on youtube|play|youtube)\b', '', query)

	query = query.replace('play', ' ')
	query = query.replace('on youtube', ' ')
	query = query.replace('youtube', ' ')

	print("Searching for videos...")
	print("Finished searching!")
	video_url = f"https://www.youtube.com/results?search_query={query.replace(' ', '+')}"
	webbrowser.open(video_url)
	return "Enjoy..."

def open_specified_website(query):
	match = re.search(r"(open|go to|visit)?\s*([\w.-]+)", query, re.IGNORECASE)
	if match:
		website = match.group(2).lower()  # Extract the website name
		if website in websites.websites_dict:  # Access the imported dictionary
			url = websites.websites_dict[website]
			webbrowser.open(url)
			print(f"Opening {website}...")
			return True
		else:
			print(f"Sorry, I don't have {website} in my database.")
			return False
	else:
		print("Invalid query format.")
		return False

def tell_me_about(query):
	try:
		topic = query.replace("tell me about ", "") #re.search(r'([A-Za-z]* [A-Za-z]* [A-Za-z]*)$', query)[1]
		result = wikipedia.summary(topic, sentences=3)
		result = re.sub(r'\[.*]', '', result)
		return result
	except (wikipedia.WikipediaException, Exception) as e:
		return None

def get_map(query):
	webbrowser.open(f'https://www.google.com/maps/search/{query}')