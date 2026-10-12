import ipaddress
import socket
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup
from pydantic import BaseModel

from usage import ResearchPhase, reserve_page_read


MAX_REDIRECTS = 5
MAX_RESPONSE_BYTES = 2_000_000


class WebPage(BaseModel):
    url: str
    content: str | None = None
    error: str | None = None


def _is_public_ip(value: str) -> bool:
    try:
        return ipaddress.ip_address(value).is_global
    except ValueError:
        return False


def _is_public_web_url(url: str) -> bool:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        return False
    if parsed.username is not None or parsed.password is not None:
        return False

    hostname = parsed.hostname.rstrip(".").lower()
    if hostname == "localhost" or hostname.endswith(".localhost"):
        return False
    if _is_public_ip(hostname):
        return True
    try:
        ipaddress.ip_address(hostname)
        return False
    except ValueError:
        pass

    try:
        addresses = socket.getaddrinfo(
            hostname,
            parsed.port or (443 if parsed.scheme == "https" else 80),
            type=socket.SOCK_STREAM,
        )
    except socket.gaierror:
        return False

    resolved_ips = {address[4][0] for address in addresses}
    return bool(resolved_ips) and all(_is_public_ip(ip) for ip in resolved_ips)


def read_page(url: str, *, phase: ResearchPhase = "initial") -> WebPage:
    original_url = url
    current_url = url
    headers = {"User-Agent": "ProductRadar/1.0 (+public-web-research)"}

    try:
        with httpx.Client(timeout=30.0, headers=headers) as client:
            for redirect_count in range(MAX_REDIRECTS + 1):
                if not _is_public_web_url(current_url):
                    return WebPage(url=original_url, error="unsafe_url")

                # 只对即将发生的网络读取计数；被安全规则拦截的 URL 不消耗预算。
                reserve_page_read(phase)
                with client.stream(
                    "GET",
                    current_url,
                    follow_redirects=False,
                ) as response:
                    if response.is_redirect:
                        if redirect_count == MAX_REDIRECTS:
                            return WebPage(
                                url=original_url,
                                error="too_many_redirects",
                            )
                        location = response.headers.get("location")
                        if not location:
                            return WebPage(
                                url=original_url,
                                error="invalid_redirect",
                            )
                        current_url = urljoin(current_url, location)
                        continue

                    response.raise_for_status()
                    content_type = response.headers.get("content-type", "")
                    if "text/html" not in content_type.lower():
                        return WebPage(
                            url=original_url,
                            error="unsupported_content",
                        )

                    content_length = response.headers.get("content-length")
                    if content_length is not None:
                        try:
                            if int(content_length) > MAX_RESPONSE_BYTES:
                                return WebPage(
                                    url=original_url,
                                    error="content_too_large",
                                )
                        except ValueError:
                            pass

                    body = bytearray()
                    for chunk in response.iter_bytes():
                        body.extend(chunk)
                        if len(body) > MAX_RESPONSE_BYTES:
                            return WebPage(
                                url=original_url,
                                error="content_too_large",
                            )
                    encoding = response.encoding or "utf-8"
                    html = bytes(body).decode(encoding, errors="replace")
                    break
            else:
                return WebPage(url=original_url, error="too_many_redirects")
    except httpx.TimeoutException:
        return WebPage(url=original_url, error="timeout")
    except httpx.HTTPStatusError:
        return WebPage(url=original_url, error="http_error")
    except httpx.RequestError:
        return WebPage(url=original_url, error="request_failed")

    soup = BeautifulSoup(html, "html.parser")
    for element in soup(["script", "style", "noscript"]):
        element.decompose()

    content = " ".join(soup.get_text(separator=" ", strip=True).split())
    if not content:
        return WebPage(url=original_url, error="empty_content")

    return WebPage(url=original_url, content=content)
