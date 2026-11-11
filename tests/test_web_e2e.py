from __future__ import annotations

from pathlib import Path
import threading

import pytest
from playwright.sync_api import sync_playwright
from werkzeug.serving import make_server

from web.app import create_app


CHROME_CANDIDATES = (
    Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe"),
    Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
)


@pytest.fixture(scope="module")
def live_site():
    server = make_server("127.0.0.1", 0, create_app(), threaded=True)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        thread.join(timeout=5)


@pytest.fixture(scope="module")
def browser():
    executable = next((path for path in CHROME_CANDIDATES if path.exists()), None)
    if executable is None:
        pytest.skip("Chrome or Edge is required for browser E2E tests")
    with sync_playwright() as playwright:
        instance = playwright.chromium.launch(headless=True, executable_path=str(executable))
        try:
            yield instance
        finally:
            instance.close()


def test_dashboard_and_mobile_layout_do_not_overflow(browser, live_site):
    desktop = browser.new_page(viewport={"width": 1440, "height": 900})
    desktop.goto(f"{live_site}/#dashboard", wait_until="domcontentloaded")
    desktop.wait_for_selector("#dashboardMetrics strong")
    assert desktop.locator("#pageTitle").inner_text() == "仪表盘"
    assert desktop.evaluate("document.body.scrollWidth <= document.documentElement.clientWidth")
    desktop.wait_for_selector("#coverageChart canvas")
    canvas = desktop.locator("#coverageChart canvas")
    assert canvas.count() == 1
    assert canvas.evaluate("element => element.width > 0 && element.height > 0")
    desktop.close()

    mobile = browser.new_page(viewport={"width": 390, "height": 844})
    mobile.goto(f"{live_site}/#dashboard", wait_until="domcontentloaded")
    mobile.wait_for_selector("#dashboardMetrics strong")
    assert mobile.evaluate("document.body.scrollWidth <= document.documentElement.clientWidth")
    assert mobile.locator(".side-nav").evaluate("element => getComputedStyle(element).scrollbarWidth") == "none"
    mobile.close()


def test_stock_search_supports_chinese_name_and_numeric_code(browser, live_site):
    page = browser.new_page(viewport={"width": 1280, "height": 800})
    page.goto(f"{live_site}/#stocks", wait_until="domcontentloaded")
    search = page.get_by_placeholder("输入 601899 / 601899.SH / 紫金矿业")
    search.fill("紫金矿业")
    page.get_by_role("button", name="搜索", exact=True).click()
    page.wait_for_selector("#stockProfile:not([hidden])")
    assert page.locator("#stockName").inner_text() == "紫金矿业"
    assert page.locator("#stockDataDate").inner_text() == "20260716"
    assert page.locator("#stockRadar canvas").count() == 1

    search.fill("000001")
    page.get_by_role("button", name="搜索", exact=True).click()
    page.wait_for_function("document.querySelector('#stockName').textContent === '平安银行'")
    assert page.locator("#stockName").inner_text() == "平安银行"
    page.close()


def test_history_compare_cancelled_run_and_stale_result_clearing(browser, live_site):
    page = browser.new_page(viewport={"width": 1440, "height": 900})
    page.goto(f"{live_site}/#history", wait_until="domcontentloaded")
    page.wait_for_function("document.querySelector('#historyRows').textContent.includes('v11_top5_daily')")
    completed = page.locator("#historyRows tr", has_text="v11_top5_daily").filter(has_text="已完成")
    cancelled = page.locator("#historyRows tr", has_text="v11_top5_daily").filter(has_text="已取消")
    assert completed.count() == 1
    assert cancelled.count() == 1

    completed.locator('input[type="checkbox"]').check()
    cancelled.locator('input[type="checkbox"]').check()
    compare = page.get_by_role("button", name="对比所选", exact=True)
    assert compare.is_enabled()
    compare.click()
    page.wait_for_selector("#comparePanel:not([hidden])")
    assert "已取消" in page.locator("#comparePanel").inner_text()
    assert "已完成" in page.locator("#comparePanel").inner_text()

    completed.get_by_role("button", name="查看", exact=True).click()
    page.wait_for_selector("#dailyResult:not([hidden])")
    assert page.locator("#dailyEmpty").is_hidden()
    assert page.locator(".candidate-row").count() == 5
    page.get_by_label("交易日期", exact=True).fill("20260715")
    assert page.locator("#dailyResult").is_hidden()
    assert page.locator("#dailyEmpty").is_visible()
    assert page.locator(".candidate-row").count() == 0
    page.close()
