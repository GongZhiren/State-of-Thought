#!/usr/bin/env python3
"""Expose JUnit failures as GitHub Actions annotations."""

from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path


def _escape(value: str) -> str:
    return value.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("junit_xml", type=Path)
    args = parser.parse_args()

    root = ET.parse(args.junit_xml).getroot()
    for case in root.iter("testcase"):
        failure = case.find("failure")
        if failure is None:
            failure = case.find("error")
        if failure is None:
            continue
        classname = case.attrib.get("classname", "")
        path = classname.replace(".", "/") + ".py" if classname else "tests"
        title = f"{classname}.{case.attrib.get('name', 'test')}"
        details = failure.text or failure.attrib.get("message", "pytest failure")
        print(f"::error file={path},title={_escape(title)}::{_escape(details)}")


if __name__ == "__main__":
    main()
