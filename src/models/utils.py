import sys
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable
from tqdm import tqdm

from config.settings import MAX_RETRIES
from .base import ResponseIsNone


def parallel_generation(
        func: Callable[[Any], str],
        prompts: list[Any],
        max_connections: int
    ) -> list[str]:

    def wrapper(prompt: str) -> str:
        res = func(prompt)
        if not res:
            raise ResponseIsNone()
        return res

    def retry(prompt: str) -> str:
        for i in range(MAX_RETRIES):
            try: return wrapper(prompt)
            # TODO: stop excepting Exception, be more specific
            except Exception as ex:
                times = i + 1
                print("Error doing API call :", ex, file=sys.stderr)
                print(f"RETRY {times:02}/{MAX_RETRIES}", file=sys.stderr)

                # Check if it's a rate limit error (429)
                error_str = str(ex)
                if '429' in error_str or 'rate limit' in error_str.lower():
                    # For rate limiting, wait much longer (60 seconds on first retry)
                    wait_time = 30 * times
                    print(f"Rate limit hit, waiting {wait_time} seconds...", file=sys.stderr)
                    time.sleep(wait_time)
                else:
                    # For other errors, use shorter exponential backoff
                    time.sleep(1.5 * times)

        # TODO: property handle gemini 'request quota'
        return wrapper(prompt)

    with ThreadPoolExecutor(max_workers=max_connections) as executor:
        return [result for result in tqdm(executor.map(retry, prompts), total=len(prompts))]
