from abc import ABC, abstractmethod

from bs4 import BeautifulSoup
import requests


class BaseScraper(ABC):
    @abstractmethod
    def scrape(self, url: str) -> dict: ...

    @abstractmethod
    def fetch_text_from_page(self, url: str) -> str: ...

class BS4Scraper(BaseScraper):
    soup: BeautifulSoup

    def scrape(self, url: str) -> BeautifulSoup:
        """
        Scrape the website using BeautifulSoup4.
        """
        response = requests.get(url)
        if response.status_code != 200:
            print(f"Failed to retrieve content from {url}")
            return {}

        soup = BeautifulSoup(response.text, "html.parser")
        return soup

    def fetch_text_from_page(self, url: str) -> str:
        """
        Fetch the text from the page.
        """
        soup = self.scrape(url)
        paragraphs = soup.find_all(["p", "span", "div"])
        text = "\n".join(p.get_text(separator=" ", strip=True) for p in paragraphs)
        return text.strip()
