import csv
import hashlib
import itertools
import os
import pickle
import random
import re
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple, Union
from urllib.parse import parse_qs, urlencode, urlparse

import requests
from bs4 import BeautifulSoup
from crawl4ai import AsyncWebCrawler, CacheMode
from ragatouille import RAGPretrainedModel
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from sentence_transformers import CrossEncoder
from tqdm import tqdm
from webdriver_manager.chrome import ChromeDriverManager

#from modules.utils import read_collections_and_sources

global driver


def scrape_google_search_results_requests(
    query: str,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    sources: Optional[List[str]] = None,
    num_pages: int = 1,
    time_chunking: bool = False,
    debug: bool = False,
    include_metadata: bool = False,
    geolocation: Optional[str] = None,
    sleep: int = 0,
) -> Optional[Union[List[str], List[Dict[str, str]]]]:
    """
    Perform a Google search and return a list of main URLs scraped from the search results pages.
    This function is cached for repeated calls with the same query.

    Args:
        query: The search query.
        start_date: The start date for the search.
        end_date: The end date for the search.
        sources: A list of sites to filter the search results by.
        num_pages: The number of search result pages to scrape. Default is 1.
        time_chunking: If True, chunk requests for date ranges > 1 year. Default is False.
        debug: If True, print debug information. Default is False.
        include_metadata: If True, include title and description for each URL. Default is False.
        geolocation: The geolocation to use for the search. Default is None.
        sleep: The number of seconds to sleep between requests. Default is 0.
    Returns:
        A list of main URLs or a list of dictionaries containing URL, title, and description, or None if the request fails.
    """
    if debug:
        print(f"Debug: Starting scrape_google_search_results with query: {query}")
    base_url = "https://www.google.com/search"
    params = {"q": query}

    query_hash = hashlib.md5(query.encode()).hexdigest()
    progress_filename = f"search_progress_{query_hash}.pkl"

    results = []
    completed_chunks = set()
    if os.path.exists(progress_filename):
        with open(progress_filename, "rb") as f:
            progress_data = pickle.load(f)
            completed_chunks = progress_data["completed_chunks"]
            results = progress_data["results"]
        if debug:
            print(
                f"Debug: Loaded progress with {len(completed_chunks)} completed chunks"
            )

    source_chunks = []
    if sources:
        for i in range(0, len(sources), 5):
            source_chunks.append(sources[i : i + 5])
    else:
        source_chunks = [None]

    time_chunks = [(None, None)]
    if start_date and end_date and time_chunking and (end_date - start_date).days > 365:
        if debug:
            print("Debug: Using time chunking")
        total_days = (end_date - start_date).days
        chunk_size = total_days // ((total_days // 365) + 1)

        current_start = start_date
        while current_start < end_date:
            chunk_end = min(current_start + timedelta(days=chunk_size), end_date)
            time_chunks.append((current_start, chunk_end))
            current_start = chunk_end + timedelta(days=1)
    else:
        if debug:
            print("Debug: Not using time chunking")

    try:
        for source_chunk in source_chunks:
            chunk_base_params = params.copy()
            if source_chunk:
                chunk_base_params["q"] += f" site:{' OR site:'.join(source_chunk)}"
            if debug:
                print(
                    f"Debug: Processing source chunk with query: {chunk_base_params['q']}"
                )

            for chunk_start, chunk_end in time_chunks:
                chunk_id = f"{chunk_base_params['q']}_{chunk_start}_{chunk_end}"

                if chunk_id in completed_chunks:
                    if debug:
                        print(f"Debug: Skipping already processed chunk {chunk_id}")
                    continue

                chunk_params = chunk_base_params.copy()
                if chunk_start:
                    chunk_params["q"] += f" after:{chunk_start.strftime('%Y-%m-%d')}"
                if chunk_end:
                    chunk_params["q"] += f" before:{chunk_end.strftime('%Y-%m-%d')}"

                chunk_results = []
                for page in range(num_pages):
                    chunk_params["start"] = page * 10
                    search_url = f"{base_url}?{urlencode(chunk_params)}"
                    if geolocation:
                        search_url += f"&gl={geolocation}"
                    if debug:
                        print(f"Debug: Fetching page {page + 1} with URL: {search_url}")

                    try:
                        response = requests.get(
                            search_url, timeout=10, params={"pws": 0}
                        )
                        response.raise_for_status()
                        if sleep:
                            time.sleep(sleep * (1 + (random.random() - 0.5) * 0.5))
                    except requests.RequestException as e:
                        print(f"Error fetching search results for page {page + 1}: {e}")
                        continue

                    soup = BeautifulSoup(response.text, "html.parser")
                    page_results = []

                    for result in soup.find_all("div", class_="Gx5Zad"):
                        if "Pg70bf" not in result.get(
                            "class", []
                        ) and "OcpZAb" not in result.get("class", []):
                            if include_metadata:
                                try:
                                    title_element = result.find("h3")
                                    if not title_element:
                                        continue

                                    title = title_element.text

                                    url_element = result.find("a")
                                    if not url_element:
                                        continue

                                    url = url_element["href"]
                                    if url.startswith("/url?"):
                                        url = parse_qs(urlparse(url).query)["q"][0]

                                    if not url.startswith("http"):
                                        continue

                                    description_element = result.find(
                                        "div", class_="BNeawe s3v9rd AP7Wnd"
                                    )
                                    description = (
                                        description_element.text
                                        if description_element
                                        else ""
                                    )

                                    page_results.append(
                                        {
                                            "url": url,
                                            "title": title,
                                            "description": description,
                                        }
                                    )
                                except:
                                    continue
                            else:
                                try:
                                    url_element = result.find("a")
                                    if not url_element:
                                        continue

                                    url = url_element["href"]
                                    if url.startswith("/url?"):
                                        url = parse_qs(urlparse(url).query)["q"][0]
                                    if url.startswith("http"):
                                        page_results.append(url)
                                except:
                                    continue

                    if debug:
                        print(
                            f"Debug: Found {len(page_results)} results on page {page + 1}"
                        )

                    if len(page_results) < 9:
                        if debug:
                            print("Debug: Last page found. Stopping search.")
                        break

                    chunk_results.extend(page_results)

                results.extend(chunk_results)
                completed_chunks.add(chunk_id)
                with open(progress_filename, "wb") as f:
                    pickle.dump(
                        {"completed_chunks": completed_chunks, "results": results}, f
                    )
                if debug:
                    print(f"Debug: Saved progress after chunk {chunk_id}")

    finally:
        if debug:
            print(f"Debug: Total results found before deduplication: {len(results)}")
        if include_metadata:
            unique_results = {}
            for result in results:
                if isinstance(result, dict):
                    url = result["url"]
                    if url not in unique_results:
                        unique_results[url] = result
            results = list(unique_results.values())
        else:
            results = list(set(results))
        if debug:
            print(f"Debug: Total results found after deduplication: {len(results)}")

        if os.path.exists(progress_filename):
            os.remove(progress_filename)

        return results if results else None


def scrape_google_search_results(
    query: str,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    sources: Optional[List[str]] = None,
    num_pages: int = 1,
    time_chunking: bool = False,
    debug: bool = False,
    include_metadata: bool = False,
    geolocation: Optional[str] = None,
    sleep: int = 0,
    use_selenium: bool = False,
) -> Optional[Union[List[str], List[Dict[str, str]]]]:
    if use_selenium:
        return scrape_google_search_results_selenium(
            query,
            start_date,
            end_date,
            sources,
            num_pages,
            time_chunking,
            debug,
            include_metadata,
            geolocation,
            sleep,
        )
    else:
        return scrape_google_search_results_requests(
            query,
            start_date,
            end_date,
            sources,
            num_pages,
            time_chunking,
            debug,
            include_metadata,
            geolocation,
            sleep,
        )


def scrape_google_search_results_selenium(
    query: str,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    sources: Optional[List[str]] = None,
    num_pages: int = 1,
    time_chunking: bool = False,
    debug: bool = False,
    include_metadata: bool = False,
    geolocation: Optional[str] = None,
    sleep: int = 0,
) -> Optional[Union[List[str], List[Dict[str, str]]]]:
    """
    Perform a Google search and return a list of main URLs scraped from the search results pages.
    This function is cached for repeated calls with the same query.

    Args:
        query (str): The search query.
        start_date (Optional[datetime]): The start date for the search.
        end_date (Optional[datetime]): The end date for the search.
        sources (Optional[List[str]]): A list of sites to filter the search results by.
        num_pages (int): The number of search result pages to scrape. Default is 1.
        time_chunking (bool): If True, chunk requests for date ranges > 1 year. Default is False.
        debug (bool): If True, print debug information. Default is False.
        include_metadata (bool): If True, include title and description for each URL. Default is False.
        geolocation (Optional[str]): The geolocation to use for the search. Default is None.
        sleep (int): The number of seconds to sleep between requests. Default is 0.
    Returns:
        Optional[Union[List[str], List[Dict[str, str]]]]: A list of main URLs or a list of dictionaries containing URL, title, and description, or None if the request fails.
    """
    if debug:
        print(f"Debug: Starting scrape_google_search_results with query: {query}")

    progress_filename = (
        f"selenium_progress_{hashlib.md5(query.encode()).hexdigest()}.pkl"
    )

    completed_chunks = set()
    results = []
    if os.path.exists(progress_filename):
        with open(progress_filename, "rb") as f:
            saved_data = pickle.load(f)
            completed_chunks = saved_data["completed_chunks"]
            results = saved_data["results"]
            if debug:
                print(
                    f"Debug: Loaded {len(completed_chunks)} completed chunks and {len(results)} results"
                )

    options = webdriver.ChromeOptions()
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_argument("--blink-settings=imagesEnabled=false")

    if "driver" not in globals():
        driver = webdriver.Chrome(
            service=Service(ChromeDriverManager().install()), options=options
        )

    base_url = "https://www.google.com/search"
    params = {"q": query, "pws": 0}

    source_chunks = []
    if sources:
        for i in range(0, len(sources), 5):
            source_chunks.append(sources[i : i + 5])
    else:
        source_chunks = [[]]

    if debug and len(source_chunks) > 1:
        print(
            f"Debug: Split sources into {len(source_chunks)} chunks of max 5 sources each"
        )

    time_chunks = [(None, None)]
    if start_date and end_date and time_chunking and (end_date - start_date).days > 365:
        if debug:
            print("Debug: Using time chunking")
        total_days = (end_date - start_date).days
        chunk_size = total_days // ((total_days // 365) + 1)

        current_start = start_date
        while current_start < end_date:
            chunk_end = min(current_start + timedelta(days=chunk_size), end_date)
            time_chunks.append((current_start, chunk_end))
            current_start = chunk_end + timedelta(days=1)
    else:
        if debug:
            print("Debug: Not using time chunking")

    if debug:
        print(f"Debug: Number of time chunks: {len(time_chunks)}")

    try:
        for source_chunk in source_chunks:
            chunk_base_params = params.copy()
            if source_chunk:
                chunk_base_params["q"] += f" site:{' OR site:'.join(source_chunk)}"
            if debug:
                print(
                    f"Debug: Processing source chunk with query: {chunk_base_params['q']}"
                )

            for chunk_start, chunk_end in time_chunks:
                chunk_id = f"{chunk_base_params['q']}_{chunk_start}_{chunk_end}"

                if chunk_id in completed_chunks:
                    if debug:
                        print(f"Debug: Skipping already processed chunk {chunk_id}")
                    continue

                chunk_params = chunk_base_params.copy()
                if chunk_start:
                    chunk_params["q"] += f" after:{chunk_start.strftime('%Y-%m-%d')}"
                if chunk_end:
                    chunk_params["q"] += f" before:{chunk_end.strftime('%Y-%m-%d')}"
                if debug:
                    print(f"Debug: Processing chunk from {chunk_start} to {chunk_end}")

                chunk_results = []
                for page in range(num_pages):
                    chunk_params["start"] = page * 10
                    search_url = f"{base_url}?{urlencode(chunk_params)}"
                    if geolocation:
                        search_url += f"&gl={geolocation}"
                    if debug:
                        print(f"Debug: Fetching page {page + 1} with URL: {search_url}")

                    driver.get(search_url)
                    WebDriverWait(driver, 10).until(
                        EC.presence_of_element_located((By.ID, "search"))
                    )

                    if sleep:
                        time.sleep(sleep * (1 + (random.random() - 0.5) * 0.5))

                    search_results = driver.find_elements(By.CSS_SELECTOR, "div.g")

                    page_results = []
                    for result in search_results:
                        if include_metadata:
                            try:
                                title_element = result.find_element(
                                    By.CSS_SELECTOR, "h3.LC20lb"
                                )
                                title = title_element.text if title_element else ""

                                url_element = result.find_element(
                                    By.CSS_SELECTOR, "div.yuRUbf a"
                                )
                                url = (
                                    url_element.get_attribute("href")
                                    if url_element
                                    else ""
                                )

                                if not url or not url.startswith("http"):
                                    continue

                                description_element = result.find_element(
                                    By.CSS_SELECTOR, "div.VwiC3b"
                                )
                                description = (
                                    description_element.text
                                    if description_element
                                    else ""
                                )

                                page_results.append(
                                    {
                                        "url": url,
                                        "title": title,
                                        "description": description,
                                    }
                                )
                            except:
                                continue
                        else:
                            try:
                                url_element = result.find_element(
                                    By.CSS_SELECTOR, "div.yuRUbf a"
                                )
                                url = (
                                    url_element.get_attribute("href")
                                    if url_element
                                    else ""
                                )
                                if not url or not url.startswith("http"):
                                    continue
                                page_results.append(url)
                            except:
                                continue

                    if debug:
                        print(
                            f"Debug: Found {len(page_results)} results on page {page + 1}"
                        )

                    if len(page_results) == 0:
                        if debug:
                            print("Debug: Last page found. Stopping search.")
                        break

                    chunk_results.extend(page_results)

                results.extend(chunk_results)
                completed_chunks.add(chunk_id)
                with open(progress_filename, "wb") as f:
                    pickle.dump(
                        {"completed_chunks": completed_chunks, "results": results}, f
                    )
                if debug:
                    print(f"Debug: Saved progress after chunk {chunk_id}")

    finally:
        driver.quit()

    if debug:
        print(f"Debug: Total results found before deduplication: {len(results)}")

    if include_metadata:
        unique_results = {}
        for result in results:
            if isinstance(result, dict):
                url = result["url"]
                if url not in unique_results:
                    unique_results[url] = result
        results = list(unique_results.values())
    else:
        results = list(set(results))

    if debug:
        print(f"Debug: Total results found after deduplication: {len(results)}")

    if os.path.exists(progress_filename):
        os.remove(progress_filename)

    return results if results else None


def scrape_google_search_results_from_templates_and_values(
    templates: List[str],
    values: Dict[str, List[Union[str, Tuple[str, ...]]]],
    save_to_csv: bool = False,
    csv_filename: str = "search_results.csv",
    collect_dates: bool = False,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    sources: Optional[List[str]] = None,
    num_pages: int = 1,
    time_chunking: bool = False,
    debug: bool = False,
    include_metadata: bool = False,
    use_geolocation: bool = False,
    geolocation: Optional[str] = None,
    use_mc_collection: bool = False,
    sleep: int = 0,
    use_selenium: bool = False,
) -> Tuple[
    Dict[Tuple[Union[str, Tuple[str, ...]]], Set[str]],
    Optional[Dict[str, Dict[str, str]]],
    List[str],
]:
    """
    Scrape Google search results for each template and combination of values.

    Args:
        templates (List[str]): A list of templates in the format "query {placeholder_name}...".
        values (Dict[str, List[Union[str, Tuple[str, ...]]]]): A dictionary of values for each placeholder name.
                                                               Values can be strings or tuples of strings.
        save_to_csv (bool): If True, save the results to a CSV file. Also saves metadata and queries to a pickle file. Default is False.
        csv_filename (str): The name of the CSV file to save results to. Default is "search_results.csv".
        collect_dates (bool): If True, collect dates from the web pages. Default is False.
        start_date (Optional[datetime]): The start date for the search.
        end_date (Optional[datetime]): The end date for the search.
        sources (Optional[List[str]]): A list of sites to filter the search results by.
        num_pages (int): The number of search result pages to scrape. Default is 1.
        time_chunking (bool): If True, chunk requests for date ranges > 1 year. Default is False.
        debug (bool): If True, print debug information. Default is False.
        include_metadata (bool): If True, include title and description for each URL. Default is False.
        use_geolocation (bool): If True, predict the geolocation of each query and use it for the search. Default is False.
        geolocation (Optional[str]): The geolocation to use for the search. Overwritten by use_geolocation. Default is None.
        use_mc_collection (bool): If True, predict the MediaCloud collection of each query and use it for the search. Overwritten by sources. Default is False.
        sleep (int): The number of seconds to sleep between requests. Default is 0.
        use_selenium (bool): If True, use Selenium to scrape the search results. Default is False.
    Returns:
        Tuple[Dict[Tuple[Union[str, Tuple[str, ...]]], Set[str]], Optional[Dict[str, Dict[str, str]]], List[str]]:
            A tuple containing:
            1. A dictionary of unique URLs for each combination of values across all templates.
            2. If include_metadata is True, a dictionary of metadata for each URL, else None.
            3. A list of all queries used in the search.
    """
    url_dict = {}
    metadata_dict = {} if include_metadata else None
    placeholder_names = set()
    all_queries = []
    completed_combinations = {}

    pkl_filename = csv_filename.rsplit(".", 1)[0] + ".pkl"
    if save_to_csv and os.path.exists(csv_filename) and os.path.exists(pkl_filename):
        print(f"Restoring from existing files: {csv_filename} and {pkl_filename}")
        with open(csv_filename, "r", newline="", encoding="utf-8") as csvfile:
            reader = csv.reader(csvfile)
            header = next(reader)
            for row in reader:
                url = row[-2] if collect_dates else row[-1]
                values_in_row = row[:-2] if collect_dates else row[:-1]
                value_combo = tuple(values_in_row)
                if value_combo in url_dict:
                    url_dict[value_combo].add(url)
                else:
                    url_dict[value_combo] = {url}

        with open(pkl_filename, "rb") as f:
            saved_data = pickle.load(f)
            metadata_dict = saved_data["metadata_dict"] if include_metadata else None
            all_queries = saved_data["all_queries"]
            completed_combinations = saved_data.get("completed_combinations", {})

        print(f"Restored {len(completed_combinations)} completed combinations")

    if save_to_csv and not os.path.exists(csv_filename):
        os.makedirs(os.path.dirname(csv_filename) or ".", exist_ok=True)
        with open(csv_filename, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            header = []
            for name in placeholder_names:
                if isinstance(values[name][0], tuple):
                    header.extend([f"{name}_{i}" for i in range(len(values[name][0]))])
                else:
                    header.append(name)
            header.append("url")
            if collect_dates:
                header.append("date")
            writer.writerow(header)

    for template in templates:
        placeholder_pattern = r"\{([^}\[]+)(?:\[(\d+)\])?\}"
        placeholders = re.findall(placeholder_pattern, template)

        base_keys = set(key for key, _ in placeholders)
        placeholder_names.update(base_keys)

        value_combinations = itertools.product(*[values[name] for name in base_keys])
        print(value_combinations)
        for value_combo in tqdm(
            value_combinations, desc="Processing value combinations"
        ):
            if (
                value_combo in completed_combinations
                and template in completed_combinations[value_combo]
            ):
                if debug:
                    print(
                        f"Skipping already processed combination: {value_combo} with template: {template}"
                    )
                continue

            format_dict = {}
            for base_key, value in zip(base_keys, value_combo):
                if isinstance(value, tuple):
                    format_dict[base_key] = value
                else:
                    format_dict[base_key] = [value]

            formatted_query = template
            for key, index in placeholders:
                if index:
                    formatted_query = formatted_query.replace(
                        f"{{{key}[{index}]}}", str(format_dict[key][int(index)])
                    )
                else:
                    formatted_query = formatted_query.replace(
                        f"{{{key}}}", str(format_dict[key][0])
                    )

            query = formatted_query
            all_queries.append(query)
            print(f"Query: {query}")
            if use_geolocation:
                geolocation = predict_geolocation(query)
                print(f"Predicted geolocation: {geolocation}")
            if use_mc_collection:
                sources = get_mc_sources(query)
            search_results = scrape_google_search_results(
                query,
                start_date,
                end_date,
                sources,
                num_pages,
                time_chunking,
                debug,
                include_metadata,
                geolocation,
                sleep,
                use_selenium,
            )

            if include_metadata:
                urls = set()
                for result in search_results:
                    url = result["url"]
                    urls.add(url)
                    metadata_dict[url] = {
                        "title": result["title"],
                        "description": result["description"],
                    }
            else:
                urls = set(search_results)

            if value_combo in url_dict:
                url_dict[value_combo].update(urls)
            else:
                url_dict[value_combo] = urls

            if save_to_csv:
                with open(csv_filename, "a", newline="", encoding="utf-8") as csvfile:
                    writer = csv.writer(csvfile)
                    for url in urls:
                        row = []
                        for name in placeholder_names:
                            value = next(
                                (v for v in value_combo if v in values[name]), ""
                            )
                            if isinstance(value, tuple):
                                row.extend(value)
                            else:
                                row.append(value)
                        row.append(url)
                        if collect_dates:
                            row.append("")
                        writer.writerow(row)

                if value_combo in completed_combinations:
                    completed_combinations[value_combo].add(template)
                else:
                    completed_combinations[value_combo] = {template}

                data_to_save = {
                    "metadata_dict": metadata_dict,
                    "all_queries": all_queries,
                    "completed_combinations": completed_combinations,
                }
                with open(pkl_filename, "wb") as f:
                    pickle.dump(data_to_save, f)

    return url_dict, metadata_dict, all_queries


def rerank_urls(
    query: str,
    urls: List[str],
    metadata: Optional[List[str]] = None,
    sort_by_score: bool = True,
) -> List[Tuple[str, float]]:
    """
    Rerank the URLs based on their relevance to the query.

    Args:
        query: The search query
        urls: List of URLs to rerank
        metadata: Optional list of metadata for each URL to rerank by (impute missing values with url)
        sort_by_score: Whether to sort results by score. Defaults to True.

    Returns:
        List of tuples containing (url, score) pairs, sorted by score if sort_by_score is True
    """
    if not urls:
        return urls
    model = CrossEncoder("mixedbread-ai/mxbai-rerank-xsmall-v1")

    if metadata is None:
        documents = urls
    else:
        documents = [meta if meta else url for meta, url in zip(metadata, urls)]

    results = model.rank(query, documents, top_k=len(documents))
    sorted_results = sorted(results, key=lambda x: x["corpus_id"])
    url_scores = [
        (urls[result["corpus_id"]], result["score"]) for result in sorted_results
    ]

    if sort_by_score:
        url_scores = sorted(url_scores, key=lambda x: x[1], reverse=True)

    return url_scores


def predict_geolocation(query: str) -> str:
    """
    Predict the geolocation of a query using ColBERT.

    Args:
        query: The query to predict the geolocation of

    Returns:
        The predicted geolocation code if confidence is high enough, otherwise None
    """
    RAG = RAGPretrainedModel.from_index(
        "../.ragatouille/colbert/indexes/countries_mixedbread"
    )
    result = RAG.search(query)[0]

    if result["score"] > 19:
        country2code = {}
        with open("../data/country2code.csv", "r") as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                country2code[row[0]] = row[1]
        return country2code[result["content"]]
    return None


def get_mc_sources(query: str) -> List[str]:
    """
    Get the MediaCloud sources for a query.

    Args:
        query: The query to get the MediaCloud sources for

    Returns:
        List of MediaCloud sources if confidence is high enough, otherwise None
    """
    RAG = RAGPretrainedModel.from_index(
        "../.ragatouille/colbert/indexes/mc_collections_mixedbread"
    )
    result = RAG.search(query)[0]

    if result["score"] > 19:
        print(f"Predicted collection: {result['content']}")
        _, sources = read_collections_and_sources()
        return sources[result["content"]]
    return None


async def extract_links(urls: list[str]):
    """
    Extract all links from a list of URLs using AsyncWebCrawler.

    Args:
        urls: List of URLs to extract links from

    Returns:
        List of lists containing (url, text) tuples for each input URL
    """
    results = []
    async with AsyncWebCrawler(verbose=True) as crawler:
        for url in urls:
            results.append(
                await crawler.arun(
                    url=url,
                    verbose=True,
                    exclude_social_media_links=True,
                    exclude_external_links=False,
                    scan_full_page=True,
                    scroll_delay=2,
                    magic=True,
                    cache_mode=CacheMode.DISABLED,
                )
            )

    all_links = []
    for result, url in zip(results, urls):
        url_links = []
        if result.links:
            links = [
                (l["href"], l["text"])
                for l in result.links["internal"] + result.links["external"]
                if (l["href"].startswith("/") or l["href"].startswith("http"))
                and not "?" in l["href"]
            ]
            base_url = url.split("//")[-1].split("/")[0]
            url_links = [
                (f"https://{base_url}{l}" if l.startswith("/") else l, t)
                for l, t in links
            ]
        all_links.append(url_links)
    return all_links
