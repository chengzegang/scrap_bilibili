from selenium import webdriver
from selenium.webdriver.common.by import By
import duckdb
from tqdm.auto import tqdm
import os
from itertools import chain
import re
import numpy as np


def get_bv(url: str):
    pattern = r"BV(\w+)\/?"
    bvs = re.findall(pattern, url)
    if len(bvs) > 0:
        bv = bvs[0]
        bv = "BV" + bv
        return bv
    else:
        return None


def to_bilibili_url(bv: str):
    return "https://www.bilibili.com/video/" + bv


def get_related_videos(driver: webdriver.Chrome):
    try:
        for card in driver.find_elements(By.CLASS_NAME, "card-box"):
            link = card.find_element(By.CLASS_NAME, "info").find_element(
                By.TAG_NAME, "a"
            )
            url = link.get_attribute("href")
            if url is None:
                continue
            bv = get_bv(url)
            if bv is None:
                continue
            video_url = to_bilibili_url(bv)
            title = link.get_attribute("title")
            yield bv, title, video_url
    except Exception as e:
        print(e)


def rank_list_items(driver: webdriver.Chrome):
    for item in driver.find_elements(By.CLASS_NAME, "rank-item"):
        link = (
            item.find_element(By.CLASS_NAME, "content")
            .find_element(By.CLASS_NAME, "info")
            .find_element(By.TAG_NAME, "a")
        )
        url = link.get_attribute("href")
        if url is None:
            continue
        bv = get_bv(url)
        if bv is None:
            continue
        video_url = to_bilibili_url(bv)
        title = link.get_attribute("title")
        yield bv, title, video_url


def get_board_items(driver: webdriver.Chrome):
    for item in driver.find_elements(By.CLASS_NAME, "board-item-wrap"):
        href = item.get_attribute("href")
        if href is None:
            continue
        bv = get_bv(href)
        if bv is None:
            continue
        video_url = to_bilibili_url(bv)
        title = item.get_attribute("alt")
        yield bv, title, video_url


def get_video_cards(driver: webdriver.Chrome):
    video_cards = driver.find_elements(By.CLASS_NAME, "video-card")
    for card in tqdm(video_cards):
        url = (
            card.find_element(By.CLASS_NAME, "video-card__content")
            .find_element(By.TAG_NAME, "a")
            .get_attribute("href")
        )
        bv = get_bv(url)
        if bv is None:
            continue
        video_url = to_bilibili_url(bv)
        title = (
            card.find_element(By.CLASS_NAME, "video-card__info")
            .find_element(By.TAG_NAME, "p")
            .text
        )
        yield bv, title, video_url


def login_videos(urls):
    conn = duckdb.connect("bilibili.db")
    conn.sql("CREATE SEQUENCE IF NOT EXISTS bilibili_id START WITH 1")
    conn.sql(
        "CREATE TABLE IF NOT EXISTS bilibili (id INTEGER NOT NULL PRIMARY KEY DEFAULT NEXTVAL('bilibili_id'), bv VARCHAR(255),  title VARCHAR(255), url VARCHAR(255), UNIQUE(bv))"
    )
    conn.begin()
    stmt = "INSERT INTO bilibili (bv, title, url) VALUES ($1, $2, $3) ON CONFLICT DO NOTHING"
    driver = webdriver.Chrome()
    for url in tqdm(urls):
        driver.get(url)
        driver.implicitly_wait(5)
        data = chain(
            get_board_items(driver), get_video_cards(driver), rank_list_items(driver)
        )
        for bv, video_title, video_url in data:
            conn.sql(stmt, params=(bv, video_title, video_url))
    conn.commit()
    conn.close()
    driver.close()


def random_walk_scrap(num_walks: int = 5000):
    conn = duckdb.connect("bilibili.db")
    df = conn.sql("SELECT * FROM bilibili;").to_df()
    pool = df["url"].values.tolist()
    pool = set(pool)
    curr_url = pool.pop()
    driver = webdriver.Chrome()
    stmt = "INSERT INTO bilibili (bv, title, url) VALUES ($1, $2, $3) ON CONFLICT DO NOTHING"
    for _ in tqdm(range(num_walks)):
        driver.get(curr_url)
        driver.implicitly_wait(5)
        conn.begin()
        for bv, title, url in get_related_videos(driver):
            print(bv, url)
            pool.add(url)
            conn.sql(stmt, params=(bv, title, url))
        conn.commit()
        curr_url = pool.pop()


def scrap(num_walks: int):
    urls = [
        "https://www.bilibili.com/v/popular/history",
        "https://www.bilibili.com/v/popular/rank/all",
        "https://www.bilibili.com/",
        "https://www.bilibili.com/v/popular/drama/",
    ] + [
        "https://www.bilibili.com/v/popular/weekly?num={}".format(i)
        for i in range(1, 240)
    ]
    login_videos(urls)
    random_walk_scrap(num_walks)


if __name__ == "__main__":
    scrap(5000)
