import argparse
import asyncio
import json
import logging
import os
import re
import traceback
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv
from tqdm.asyncio import tqdm_asyncio

from aworld.agents.llm_agent import Agent
from aworld.config.conf import AgentConfig, TaskConfig
from aworld.core.task import Task
from aworld.runner import Runners
from examples.gaia.prompt import system_prompt
from examples.gaia.utils import (
    add_file_path,
    load_dataset_meta,
    question_scorer,
    report_results,
)

# Create log directory if it doesn't exist
if not os.path.exists(os.getenv("AWORLD_WORKSPACE", "~")):
    os.makedirs(os.getenv("AWORLD_WORKSPACE", "~"))

parser = argparse.ArgumentParser()
parser.add_argument(
    "--start",
    type=int,
    default=0,
    help="Start index of the dataset",
)
parser.add_argument(
    "--end",
    type=int,
    default=20,
    help="End index of the dataset",
)
parser.add_argument(
    "--q",
    type=str,
    help="Question Index, e.g., 0-0-0-0-0. Highest priority: override other arguments if provided.",
)
parser.add_argument(
    "--skip",
    action="store_true",
    help="Skip the question if it has been processed before.",
)
parser.add_argument(
    "--split",
    type=str,
    default="validation",
    help="Split of the dataset, e.g., validation, test",
)
parser.add_argument(
    "--blacklist_file_path",
    type=str,
    nargs="?",
    help="Blacklist file path, e.g., blacklist.txt",
)
parser.add_argument(
    "--num-workers",
    type=int,
    default=1,
    help="Number of parallel workers to run.",
)
args = parser.parse_args()


def setup_logging():
    logging_logger = logging.getLogger()
    logging_logger.setLevel(logging.INFO)

    log_file_name = f"/super_agent_{args.q}.log" if args.q else f"/super_agent_{args.start}_{args.end}.log"
    file_handler = logging.FileHandler(
        os.getenv("AWORLD_WORKSPACE", "~") + log_file_name,
        mode="a",
        encoding="utf-8",
    )
    file_handler.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)

    logging_logger.addHandler(file_handler)


async def run_question(dataset_i, semaphore, super_agent, gaia_dataset_path, split):
    async with semaphore:
        try:
            logging.info(f"Start to process: {dataset_i['task_id']}")
            logging.info(f"Detail: {dataset_i}")
            logging.info(f"Question: {dataset_i['Question']}")
            logging.info(f"Level: {dataset_i['Level']}")
            logging.info(f"Tools: {dataset_i['Annotator Metadata']['Tools']}")

            question = add_file_path(dataset_i, file_path=gaia_dataset_path, split=split)["Question"]

            task = Task(input=question, agent=super_agent, conf=TaskConfig())
            result = await Runners.run_task(task=task)

            answer = ""
            match = re.search(r"<answer>(.*?)</answer>", result[task.id].answer)
            if match:
                answer = match.group(1)
                logging.info(f"Agent answer: {answer}")

                if question_scorer(answer, dataset_i["Final answer"]):
                    logging.info(f"Question {dataset_i['task_id']} Correct!")
                else:
                    logging.info(f"Question {dataset_i['task_id']} Incorrect!")
            else:
                logging.error(
                    "Could not parse answer from agent response. "
                    "The response is not in the expected format with <answer> tags."
                )
                logging.error(f"Full agent response: {result[task.id].answer}")
            logging.info(f"Ground truth answer: {dataset_i['Final answer']}")

            return {
                "task_id": dataset_i["task_id"],
                "level": dataset_i["Level"],
                "question": question,
                "answer": dataset_i["Final answer"],
                "response": answer,
                "is_correct": question_scorer(answer, dataset_i["Final answer"]),
            }
        except Exception:
            logging.error(f"Error processing {dataset_i['task_id']}: {traceback.format_exc()}")
            return None


async def main():
    load_dotenv()
    setup_logging()

    gaia_dataset_path = os.getenv("GAIA_DATASET_PATH", "./gaia_dataset")
    full_dataset = load_dataset_meta(gaia_dataset_path, split=args.split)
    logging.info(f"Total questions: {len(full_dataset)}")

    try:
        with open(Path(__file__).parent / "mcp.json", mode="r", encoding="utf-8") as f:
            mcp_config: dict[dict[str, Any]] = json.loads(f.read())
            available_servers: list[str] = list(
                server_name for server_name in mcp_config.get("mcpServers", {}).keys())
            logging.info(f"🔧 MCP Available Servers: {available_servers}")
    except json.JSONDecodeError as e:
        logging.error(f"Error loading mcp_collections.json: {e}")
        mcp_config = {}

    agent_config = AgentConfig(
        llm_provider=os.getenv("LLM_PROVIDER", "openai"),
        llm_model_name=os.getenv("LLM_MODEL_NAME", "gpt-4o"),
        llm_base_url=os.getenv("LLM_BASE_URL"),
        llm_api_key=os.getenv("LLM_API_KEY"),
        llm_temperature=float(os.getenv("LLM_TEMPERATURE", 0.0))
    )
    super_agent = Agent(
        conf=agent_config,
        name="gaia_super_agent",
        system_prompt=system_prompt,
        mcp_config=mcp_config,
        mcp_servers=available_servers,
    )

    # Pre-initialize the agent's tools to avoid a race condition in parallel execution.
    # The context is not used here since the agent is not configured with a sandbox.
    await super_agent.async_desc_transform(context=None)

    if os.path.exists(os.getenv("AWORLD_WORKSPACE", "~") + "/results.json"):
        with open(os.getenv("AWORLD_WORKSPACE", "~") + "/results.json", "r", encoding="utf-8") as results_f:
            results: List[Dict[str, Any]] = json.load(results_f)
    else:
        results: List[Dict[str, Any]] = []

    if args.blacklist_file_path and os.path.exists(args.blacklist_file_path):
        with open(args.blacklist_file_path, "r", encoding="utf-8") as f:
            blacklist = set(f.read().splitlines())
    else:
        blacklist = set()

    dataset_slice = (
        [dataset_record for dataset_record in full_dataset if dataset_record["task_id"] in args.q]
        if args.q is not None
        else full_dataset[args.start: args.end]
    )

    questions_to_run = []
    for dataset_i in dataset_slice:
        if args.q and args.q != dataset_i["task_id"]:
            continue
        if not args.q:
            if dataset_i["task_id"] in blacklist:
                continue
            if any(
                    (result["task_id"] == dataset_i["task_id"] and result["is_correct"])
                    for result in results
            ) or any(
                (result["task_id"] == dataset_i["task_id"] and not result["is_correct"] and dataset_i["Level"] == 3)
                for result in results
            ):
                continue
            if args.skip and any(
                    (result["task_id"] == dataset_i["task_id"])
                    for result in results
            ):
                continue
        questions_to_run.append(dataset_i)

    semaphore = asyncio.Semaphore(args.num_workers)
    tasks = [run_question(q, semaphore, super_agent, gaia_dataset_path, args.split) for q in questions_to_run]
    print(f"Total questions to run: {len(questions_to_run)}")

    new_results = await tqdm_asyncio.gather(*tasks, desc="Processing Questions", total=len(questions_to_run))

    for new_result in new_results:
        if new_result:
            existing_index = next(
                (i for i, result in enumerate(results) if result["task_id"] == new_result["task_id"]),
                None,
            )
            if existing_index is not None:
                results[existing_index] = new_result
                logging.info(f"Updated existing record for task_id: {new_result['task_id']}")
            else:
                results.append(new_result)
                logging.info(f"Added new record for task_id: {new_result['task_id']}")

    report_results(results)
    with open(os.getenv("AWORLD_WORKSPACE", "~") + "/results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass

