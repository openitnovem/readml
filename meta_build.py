#!/usr/bin/env python3
# coding: utf8
import argparse
import re
import os

meta_file_name = "meta_build.json"
regex_dev = "^\d+\.\d+\.\d+.dev\d+$"
regex_release = "^\d+\.\d+\.\d+rc\d+$"
regex_prod = "^\d+\.\d+\.\d+$"

parser = argparse.ArgumentParser(description="generate build metadatas")
parser.add_argument("version", help="published version")
parser.add_argument(
    "pattern",
    help="""

version format:

format: regex_dev|regex_release|regex_prod

dev: %s
release: %s
prod: %s
"""
    % (regex_dev, regex_release, regex_prod),
)
args = parser.parse_args()

try:
    os.unlink(meta_file_name)
except:
    pass


def validate_pattern(pattern):
    """
    Compare the correct pattern and return associated regex
     :return regex pattern (String)
     :parameter pattern arg from jenkinsfile command line
    """

    if pattern == "regex_release":
        return regex_release
    elif pattern == "regex_prod":
        return regex_prod
    else:
        return regex_dev


def check_version_syntax(version):
    """
    Compare pattern and version, if matching is correct
     :return None if pattern not correponds to version and True if it's correct
     :parameter version (String)
    """
    print(validate_pattern(args.pattern))
    return re.match(validate_pattern(args.pattern), version)


if check_version_syntax(args.version) == None:
    print("error: pattern not match")
    exit(1)
"""
Writing a json file and add version
"""
with open(meta_file_name, "w") as jfd:
    jfd.write('{"version":"%s"}' % (args.version))

exit(0)
