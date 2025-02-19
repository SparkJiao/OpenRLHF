import time
import ray
import requests
import torch

from openrlhf.utils.logging_utils import init_logger
from openrlhf.utils.reward_score.prime import compute_score

logger = init_logger(__name__)


def contrastive_compute_score(completions, references, tasks):
    scores = compute_score(completions, references, tasks)

    new_scores = []
    for s in scores:
        if s == 0.5:
            new_scores.append(0.0)
        elif s == 1.0:
            new_scores.append(1.0)
        else:
            new_scores.append(-1.0)
    return new_scores


local_rm_function = {
    "prime_reward_score": compute_score,
    "prime_reward_score_ctr": contrastive_compute_score,
}


def outcome_rewards_with_label(fn_name, queries, auxiliary):
    questions = auxiliary["questions"]
    labels = auxiliary["labels"]
    responses = auxiliary["responses"]

    score_fn = local_rm_function[fn_name]

    rewards = score_fn(responses, labels, ["math"] * len(queries))
    return torch.tensor(rewards)


def request_api_wrapper(url, data, score_key="rewards", try_max_times=5):
    """Synchronous request API wrapper"""
    headers = {
        "Content-Type": "application/json",
    }
    for _ in range(try_max_times):
        try:
            response = requests.post(url=url, json=data, headers=headers, timeout=180)
            response.raise_for_status()  # Raise an HTTPError for bad responses
            response = response.json()
            assert score_key in response, f"{score_key} not in {response}"
            return response.get(score_key)
        except requests.RequestException as e:
            logger.info(f"Request error, please check: {e}")
        except Exception as e:
            logger.info(f"Unexpected error, please check: {e}")
        time.sleep(1)

    raise Exception(f"Request error for {try_max_times} times, returning None. Please check the API server.")


def remote_rm_fn(api_url, queries, score_key="rewards"):
    """remote reward model API
    api_url: RM API, We assume that the API supports two modes: merging query + response and not merging
    queries: query+response with the template
    design is made optional.
    score_key: RM score key
    """
    scores = request_api_wrapper(api_url, {"query": queries}, score_key)
    return torch.tensor(scores)


@ray.remote
def remote_rm_fn_ray(api_url, queries, score_key="rewards", auxiliary=None):
    if api_url.startswith("local:"):
        return outcome_rewards_with_label(api_url[len("local:"):], queries, auxiliary)
    return remote_rm_fn(api_url, queries, score_key)


if __name__ == "__main__":
    # test utils
    url = "http:xxx/get_rm_score"
    score = remote_rm_fn(url, ["example query"], ["example response"])
    print(score)
