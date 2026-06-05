from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import urlsplit


@dataclass(frozen=True, slots=True)
class TrustedPublisherCatalog:
    name: str
    publisher: str
    source_type: str
    region: str
    base_domain: str
    listing_urls: tuple[str, ...]
    google_review_site: str | None = None


TRUSTED_PUBLISHER_CATALOGS: tuple[TrustedPublisherCatalog, ...] = (
    TrustedPublisherCatalog(
        name="rumor_scanner",
        publisher="Rumor Scanner",
        source_type="fact_check",
        region="bangladesh",
        base_domain="rumorscanner.com",
        listing_urls=(
            "https://rumorscanner.com/",
            "https://rumorscanner.com/en",
            "https://rumorscanner.com/archives",
        ),
        google_review_site="rumorscanner.com",
    ),
    TrustedPublisherCatalog(
        name="fact_watch",
        publisher="FactWatch",
        source_type="fact_check",
        region="bangladesh",
        base_domain="fact-watch.org",
        listing_urls=("https://www.fact-watch.org/",),
        google_review_site="fact-watch.org",
    ),
    TrustedPublisherCatalog(
        name="prothom_alo",
        publisher="Prothom Alo",
        source_type="article",
        region="bangladesh",
        base_domain="prothomalo.com",
        listing_urls=("https://www.prothomalo.com/",),
    ),
    TrustedPublisherCatalog(
        name="daily_star",
        publisher="The Daily Star",
        source_type="article",
        region="bangladesh",
        base_domain="thedailystar.net",
        listing_urls=("https://www.thedailystar.net/news/bangladesh",),
    ),
    TrustedPublisherCatalog(
        name="dhaka_tribune",
        publisher="Dhaka Tribune",
        source_type="article",
        region="bangladesh",
        base_domain="dhakatribune.com",
        listing_urls=("https://www.dhakatribune.com/",),
    ),
    TrustedPublisherCatalog(
        name="bdnews24",
        publisher="bdnews24.com",
        source_type="article",
        region="bangladesh",
        base_domain="bdnews24.com",
        listing_urls=("https://bdnews24.com/",),
    ),
    TrustedPublisherCatalog(
        name="reuters_fact_check",
        publisher="Reuters Fact Check",
        source_type="fact_check",
        region="international",
        base_domain="reuters.com",
        listing_urls=("https://www.reuters.com/fact-check/",),
        google_review_site="reuters.com",
    ),
    TrustedPublisherCatalog(
        name="associated_press_fact_check",
        publisher="Associated Press",
        source_type="fact_check",
        region="international",
        base_domain="apnews.com",
        listing_urls=("https://apnews.com/hub/ap-fact-check",),
        google_review_site="apnews.com",
    ),
    TrustedPublisherCatalog(
        name="afp_fact_check",
        publisher="AFP Fact Check",
        source_type="fact_check",
        region="international",
        base_domain="factcheck.afp.com",
        listing_urls=("https://factcheck.afp.com/",),
        google_review_site="factcheck.afp.com",
    ),
    TrustedPublisherCatalog(
        name="factcheck_org",
        publisher="FactCheck.org",
        source_type="fact_check",
        region="international",
        base_domain="factcheck.org",
        listing_urls=("https://www.factcheck.org/",),
        google_review_site="factcheck.org",
    ),
    TrustedPublisherCatalog(
        name="snopes",
        publisher="Snopes",
        source_type="fact_check",
        region="international",
        base_domain="snopes.com",
        listing_urls=("https://www.snopes.com/fact-check/",),
        google_review_site="snopes.com",
    ),
    TrustedPublisherCatalog(
        name="full_fact",
        publisher="Full Fact",
        source_type="fact_check",
        region="international",
        base_domain="fullfact.org",
        listing_urls=("https://fullfact.org/latest/",),
        google_review_site="fullfact.org",
    ),
    TrustedPublisherCatalog(
        name="bbc_news",
        publisher="BBC News",
        source_type="article",
        region="international",
        base_domain="bbc.com",
        listing_urls=("https://www.bbc.com/news",),
    ),
)


def iter_trusted_catalogs() -> tuple[TrustedPublisherCatalog, ...]:
    return TRUSTED_PUBLISHER_CATALOGS


def trusted_google_review_sites() -> list[str]:
    sites: list[str] = []
    for catalog in TRUSTED_PUBLISHER_CATALOGS:
        site = catalog.google_review_site
        if site and site not in sites:
            sites.append(site)
    return sites


def hostname_matches_trusted_domain(hostname: str | None, domain: str) -> bool:
    if not hostname:
        return False
    normalized_host = hostname.lower().strip(".")
    normalized_domain = domain.lower().strip(".")
    return normalized_host == normalized_domain or normalized_host.endswith(f".{normalized_domain}")


def get_catalog_for_url(url: str) -> TrustedPublisherCatalog | None:
    hostname = urlsplit(url).hostname
    for catalog in TRUSTED_PUBLISHER_CATALOGS:
        if hostname_matches_trusted_domain(hostname, catalog.base_domain):
            return catalog
    return None


def is_trusted_url(url: str) -> bool:
    return get_catalog_for_url(url) is not None
