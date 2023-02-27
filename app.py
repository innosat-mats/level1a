#!/usr/bin/env python3

from typing import Optional

import git
from git import TagReference
from aws_cdk import App

from stacks.level1a_stack import Level1AStack

app = App()
repo = git.Repo(".")

try:
    tag: Optional[TagReference] = repo.tags[-1]
except IndexError:
    tag = None

Level1AStack(
    app,
    "Level1AStackCCD",
    rac_bucket_name="ops-payload-level0-v0.2",
    platform_bucket_name="ops-platform-level1a-v0.3",
    output_bucket_name="ops-payload-level1a-v0.5",
    code_version=f"{tag} ({repo.head.commit})",
    data_prefix="CCD",
    read_htr=True,
)

Level1AStack(
    app,
    "Level1AStackPM",
    rac_bucket_name="ops-payload-level0-v0.2",
    platform_bucket_name="ops-platform-level1a-v0.3",
    output_bucket_name="ops-payload-level1a-PM-v0.1",
    code_version=f"{tag} ({repo.head.commit})",
    data_prefix="PM",
    read_htr=False,
)

app.synth()
