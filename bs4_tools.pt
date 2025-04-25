#Pl include this file in tools directory which shud be created inside the directory where u r running q_learning_tools.py agent
from bs4 import BeautifulSoup

def parse_html(html_content, parser="html.parser"):
    """Parses HTML content and creates a BeautifulSoup object."""
    return BeautifulSoup(html_content, parser)

def find_element(soup, tag, attributes=None):
    """Finds the first HTML element matching the given tag or attributes."""
    return soup.find(tag, attributes)

def find_all_elements(soup, tag, attributes=None):
    """Finds all HTML elements matching the given tag or attributes."""
    return soup.find_all(tag, attributes)

def get_text(element):
    """Extracts text content from an HTML element."""
    return element.get_text() if element else None
