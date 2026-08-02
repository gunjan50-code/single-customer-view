"""
Run the whole pipeline end to end.

    python run_pipeline.py

Each stage writes its output to data/processed/ and the next stage reads it,
so you can also run any single stage on its own while iterating:

    python -m src.generate_data
    python -m src.standardize
    python -m src.blocking
    python -m src.features
    python -m src.train
    python -m src.decide
    python -m src.cluster

The LLM adjudicator is deliberately not part of this run -- it costs money
and needs an API key, so it stays an explicit choice:

    python -m src.llm_adjudicator --dry-run
    python -m src.llm_adjudicator
"""

import importlib
import sys
import time

STAGES = [
    ("src.generate_data", "Generate six messy source systems from known people"),
    ("src.standardize", "Clean and normalise every record"),
    ("src.blocking", "Cut ~482M possible pairs down to the ones worth scoring"),
    ("src.features", "Turn each candidate pair into similarity features"),
    ("src.train", "Train and calibrate the match classifier"),
    ("src.decide", "Apply auto-merge / review / auto-reject bands"),
    ("src.cluster", "Group into entities and build golden records"),
]


def main() -> None:
    started = time.time()
    for index, (module_name, description) in enumerate(STAGES, start=1):
        print("\n" + "#" * 74)
        print(f"# STAGE {index}/{len(STAGES)}  {module_name}")
        print(f"# {description}")
        print("#" * 74 + "\n")

        stage_started = time.time()
        module = importlib.import_module(module_name)
        try:
            module.main() if hasattr(module, "main") else module.generate()
        except Exception as error:  # noqa: BLE001 -- want the stage name in the message
            print(f"\nStage {module_name} failed: {error}", file=sys.stderr)
            raise
        print(f"\n[{module_name} finished in {time.time() - stage_started:.1f}s]")

    print("\n" + "#" * 74)
    print(f"# Pipeline complete in {time.time() - started:.1f}s")
    print("#" * 74)
    print("\nNext:")
    print("  python -m streamlit run app.py           review the ambiguous pairs")
    print("  python -m src.llm_adjudicator --dry-run  cost of LLM adjudication")
    print("  python -m src.inspect_sample             see the problem itself")


if __name__ == "__main__":
    main()
