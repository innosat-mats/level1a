#!/usr/bin/env python3

import git
from aws_cdk import App

from stacks.level1a_stack import Level1AStack

app = App()
repo = git.Repo(".")

Level1AStack(
    app,
    "Level1AStack",
    rac_bucket_name="ops-payload-level0-v0.2",
    platform_bucket_name="ops-platform-level1a-v0.3",
    output_bucket_name="ops-payload-level1a-v0.5",
    code_version=f"{repo.tags[-1]} ({repo.head.commit})",
)

app.synth()
