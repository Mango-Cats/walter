import csv
import sys
import time
import random

from playwright.sync_api import sync_playwright


FDA_URL = "https://verification.fda.gov.ph/"

REGISTERED_FILE = "registered.txt"
QUESTIONABLE_FILE = "registered_questionable.txt"
NOT_REGISTERED_FILE = "not_registered.txt"


def extract_unique_names(csv_file, column1, column2):
    names = []
    seen = set()

    with open(csv_file, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            for column in (column1, column2):
                name = row[column]

                if not name:
                    continue

                if name not in seen:
                    seen.add(name)
                    names.append(name)

    return names


def has_non_ascii(text):
    """Return True if the text contains any non-ASCII character."""
    return any(ord(char) > 127 for char in text)


def main():
    if len(sys.argv) != 4:
        print(
            "Usage: uv run scripts/fda_search.py <csv_file> <column1> <column2>"
        )
        print(
            "Example: uv run scripts/fda_search.py data/230.csv drug_a drug_b"
        )
        sys.exit(1)

    csv_file = sys.argv[1]
    column1 = sys.argv[2]
    column2 = sys.argv[3]

    queries = extract_unique_names(
        csv_file,
        column1,
        column2,
    )

    print(f"Unique queries: {len(queries)}")

    registered = []
    questionable = []
    not_registered = []

    with sync_playwright() as p:
        browser = p.firefox.launch(headless=True)

        page = browser.new_page()

        page.goto(FDA_URL)
        page.wait_for_load_state("networkidle")

        for i, query in enumerate(queries, start=1):
            print(f"\n[{i}/{len(queries)}] Searching: {query}")

            try:
                page.get_by_role("textbox").fill(query)

                with page.expect_response(
                    lambda response:
                        "/api/search" in response.url
                        and response.request.method == "GET",
                    timeout=15000,
                ) as response_info:

                    page.get_by_role(
                        "button",
                        name="Search",
                        exact=True
                    ).click()

                response = response_info.value

                print(f"HTTP {response.status}")

                if response.status != 200:
                    print(f"Unexpected HTTP status: {response.status}")
                    continue

                results_table = page.get_by_role(
                    "table"
                ).filter(
                    has_text="Registration Number"
                )

                rows = results_table.locator("tbody tr")
                result_count = rows.count()

                if result_count == 0:
                    not_registered.append(query)
                    print(f"NOT REGISTERED: {query}")

                else:
                    table_text = results_table.inner_text()

                    if has_non_ascii(table_text):
                        questionable.append(query)

                        print(
                            f"REGISTERED — QUESTIONABLE: {query}"
                        )

                        # Show the non-ASCII characters detected
                        non_ascii = sorted(
                            set(
                                char
                                for char in table_text
                                if ord(char) > 127
                            ),
                            key=ord,
                        )

                        print(
                            "Non-ASCII characters:",
                            " ".join(
                                f"{char} (U+{ord(char):04X})"
                                for char in non_ascii
                            ),
                        )

                    else:
                        registered.append(query)
                        print(f"REGISTERED: {query}")

            except Exception as e:
                print(f"ERROR: {query}: {e}")

            # Small delay between searches
            delay = random.uniform(0.5, 1.0)
            print(f"Waiting {delay:.2f}s...")
            time.sleep(delay)

        browser.close()

    with open(
        REGISTERED_FILE,
        "w",
        encoding="utf-8",
    ) as f:
        for name in registered:
            f.write(name + "\n")

    with open(
        QUESTIONABLE_FILE,
        "w",
        encoding="utf-8",
    ) as f:
        for name in questionable:
            f.write(name + "\n")

    with open(
        NOT_REGISTERED_FILE,
        "w",
        encoding="utf-8",
    ) as f:
        for name in not_registered:
            f.write(name + "\n")

    print("\nFinished.")
    print(f"Registered:             {len(registered)}")
    print(f"Registered questionable: {len(questionable)}")
    print(f"Not registered:         {len(not_registered)}")

    print(f"\nSaved:")
    print(f"  {REGISTERED_FILE}")
    print(f"  {QUESTIONABLE_FILE}")
    print(f"  {NOT_REGISTERED_FILE}")


if __name__ == "__main__":
    main()