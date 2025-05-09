import requests
from bs4 import BeautifulSoup
import csv
import time
from urllib.parse import urljoin, urlparse
import re

# Initialize the list of visited URLs and the CSV file
visited = set()
max_pages = 10
pages_crawled = 0

# CRAWLER WAS ASSISTED BY GPT

# Function to escape double quotes in content
def escape_quotes(text):
    """Escapes double quotes in text by replacing them with two double quotes."""
    return text.replace('"', '""')

# Function to limit content to 30,000 characters to avoid Excel's character limit
def limit_content_length(content, max_length=30000):
    """Limit the content to a specified length to avoid Excel's character limit."""
    if len(content) > max_length:
        return content[:max_length]
    return content

# Create a CSV file to store the content
with open('crawler.csv', 'w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file, quoting=csv.QUOTE_MINIMAL)  # Ensure quoting when necessary
    writer.writerow(['url', 'title', 'content'])  # Header for the CSV file

def is_valid_url(url):
    """Check if the URL is valid, not previously visited, and matches the Wikipedia pattern."""
    parsed_url = urlparse(url)
    # Check if the URL starts with /wiki/ and doesn't reference a fragment (e.g., cite_note)
    if "#cite_note" in url:
        return False
    return (parsed_url.scheme in ['http', 'https'] and 
            url not in visited and
            re.match(r'^/wiki/[^:]+$', parsed_url.path))  # Match Wikipedia titles and avoid fragment links

def clean_text(text):
    """Clean the text content by removing extra spaces, newlines, and commas."""
    return text.replace('\n', ' ').replace(',', ' ').strip()

def get_page_content(url):
    """Fetch the content of a page."""
    global pages_crawled
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            # Extract title (from <h1> tag)
            title = soup.find('h1').get_text(strip=True) if soup.find('h1') else 'No Title'
            # Extract content from paragraphs only
            paragraphs = soup.find_all('p')
            content = ' '.join([clean_text(para.get_text(separator=' ', strip=True)) for para in paragraphs])  # Clean and join paragraph text
            # Extract links in paragraphs only
            links = [a['href'] for a in soup.find_all('a', href=True) if a.find_parent('p')]  # Links within paragraphs
            return title, content, links
    except requests.exceptions.RequestException as e:
        print(f"Error fetching {url}: {e}")
    return None, None, None

def crawl(start_url):
    """Crawl the pages starting from a given URL."""
    global pages_crawled
    to_crawl = [start_url]

    while to_crawl and pages_crawled < max_pages:
        url = to_crawl.pop(0)
        
        # Skip already visited URLs
        if url in visited:
            continue
        
        visited.add(url)
        
        print(f"Crawling {url}... ({pages_crawled + 1}/{max_pages})")
        
        title, content, links = get_page_content(url)
        
        if title and content:
            # Limit the content length to avoid exceeding Excel's limit
            content = limit_content_length(content)
            
            # Write the page content to the CSV file (URL, Title, Content)
            with open('crawler.csv', 'a', newline='', encoding='utf-8') as file:
                writer = csv.writer(file, quoting=csv.QUOTE_MINIMAL)  # Quote cells if necessary
                # Write URL, title, and content (cleaned and escaped) in one row
                writer.writerow([url, title, escape_quotes(content)])
            
            pages_crawled += 1
            
            # Add links to the crawl queue if they are valid
            for link in links:
                full_url = urljoin(url, link)
                if is_valid_url(full_url) and full_url not in visited:
                    to_crawl.append(full_url)
        
        # Sleep to avoid hitting the server too hard
        time.sleep(1)

# Start crawling from a given URL
start_url = 'https://en.wikipedia.org/wiki/Chicago_bears'
crawl(start_url)

print(f"Crawling completed. {pages_crawled} pages crawled.")
