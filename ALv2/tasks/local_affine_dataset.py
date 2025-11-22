# file: tasks/local_affine_dataset.py

"""Lightweight synthetic datasets that mimic the async R2Dataset interface.

These helpers let us instantiate the verified Affine env wrappers without
running the full Cloudflare R2 pipeline. Each dataset simply returns small,
deterministic Python programs or coding prompts via an async ``get`` method,
matching the contract expected by ``env.abd`` and ``env.ded``.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List


def _format_inputs(lines: List[str]) -> str:
    return "\n".join(lines)


@dataclass
class ABDSample:
    program: str
    inputs: str
    output: str


class SyntheticABDData:
    """Generates small programs whose stdin can be deduced analytically."""

    def __init__(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)

    async def get(self) -> Dict[str, str]:
        choice = self._rng.choice(["sum", "product", "string_repeat"])
        if choice == "sum":
            a, b = self._rng.randint(1, 20), self._rng.randint(1, 20)
            program = "a = int(input())\nb = int(input())\nprint(a + b)\n"
            inputs = f"{a}\n{b}"
            output = f"{a + b}\n"
        elif choice == "product":
            nums = [self._rng.randint(2, 9) for _ in range(3)]
            program = (
                "x = int(input())\n"
                "y = int(input())\n"
                "z = int(input())\n"
                "print(x * y * z)\n"
            )
            inputs = _format_inputs([str(n) for n in nums])
            output = f"{nums[0] * nums[1] * nums[2]}\n"
        else:  # string_repeat
            word = self._rng.choice(["hello", "agent", "affine"])
            k = self._rng.randint(2, 5)
            program = (
                "text = input().strip()\n"
                "k = int(input())\n"
                "print(text * k)\n"
            )
            inputs = _format_inputs([word, str(k)])
            output = f"{word * k}\n"

        sample = ABDSample(program=program, inputs=inputs, output=output)
        return {"program": sample.program, "inputs": sample.inputs, "output": sample.output}


class SyntheticDEDData:
    """Provides simple program-synthesis prompts with embedded tests."""

    def __init__(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)

    async def get(self) -> Dict[str, object]:
        spec = self._rng.choice(["sum_list", "factorial", "reverse_string"])
        if spec == "sum_list":
            numbers = [self._rng.randint(0, 20) for _ in range(5)]
            prompt = (
                "Write a program that reads a single line of integers separated by spaces "
                "and prints their sum as an integer."
            )
            tests = [
                {
                    "type": "stdin_stdout",
                    "input": " ".join(str(n) for n in numbers),
                    "output": f"{sum(numbers)}\n",
                }
            ]
        elif spec == "factorial":
            n = self._rng.randint(3, 6)
            prompt = "Read an integer n (0 <= n <= 8) and print n! on its own line."
            tests = [
                {"type": "stdin_stdout", "input": str(n), "output": f"{_factorial(n)}\n"},
                {"type": "stdin_stdout", "input": "0", "output": "1\n"},
            ]
        else:  # reverse_string
            prompt = "Read a line of text and output the string reversed (characters reversed)."
            tests = [
                {"type": "stdin_stdout", "input": "Affine\n", "output": "eniffA\n"},
                {"type": "stdin_stdout", "input": "RL\n", "output": "LR\n"},
            ]

        verification_info = {"test_cases": tests}
        sample = {
            "prompt": prompt,
            "verification_info": verification_info,
            "test_cases": verification_info,  # some legacy consumers expect this key
        }
        return sample


def _factorial(n: int) -> int:
    out = 1
    for i in range(2, n + 1):
        out *= i
    return out

