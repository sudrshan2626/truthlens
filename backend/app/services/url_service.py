import newspaper
from app.utils.logger import get_logger

logger = get_logger(__name__)


class URLService:
    """Extracts clean article text from a news URL using newspaper3k."""

    def extract_article(self, url: str) -> dict:
        """
        Downloads and parses article from URL.

        Returns:
            {
                "title":   article title,
                "text":    full article body,
                "authors": list of author names,
                "success": True/False
            }
        """
        try:
            logger.info(f"Fetching article from: {url}")

            article = newspaper.Article(url)
            article.download()
            article.parse()

            if not article.text or len(article.text.strip()) < 50:
                return {
                    "success": False,
                    "error": "Could not extract article text. The page may require login or block scrapers."
                }

            logger.info(f"Extracted {len(article.text)} chars from {url}")

            return {
                "success": True,
                "title":   article.title or "",
                "text":    article.text,
                "authors": article.authors or [],
            }

        except Exception as e:
            logger.error(f"URL extraction failed: {e}")
            return {
                "success": False,
                "error": f"Failed to fetch article: {str(e)}"
            }


_url_service_instance = None

def get_url_service() -> URLService:
    global _url_service_instance
    if _url_service_instance is None:
        _url_service_instance = URLService()
    return _url_service_instance