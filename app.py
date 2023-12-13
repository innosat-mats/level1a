#!/usr/bin/env python3

from typing import Optional

import os

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

development = bool(os.environ.get("MATS_DEVELOPMENT", False))
if development:
    output_bucket_ccd = "dev-payload-level1a"
    output_bucket_pm = "dev-payload-level1a-pm"
else:
    output_bucket_ccd = "ops-payload-level1a-v0.7"
    output_bucket_pm = "ops-payload-level1a-pm-v0.4"
rac_bucket_name = "ops-payload-level0-v0.3"
platform_bucket_name = "ops-platform-level1a-v0.3"
mats_schedule_bucket_name = "ops-mats-schedule-v0.2"

Level1AStack(
    app,
    f"Level1AStackCCD{'Dev' if development else ''}",
    rac_bucket_name=rac_bucket_name,
    platform_bucket_name=platform_bucket_name,
    mats_schedule_bucket_name=mats_schedule_bucket_name,
    output_bucket_name=output_bucket_ccd,
    code_version=f"{tag} ({repo.head.commit})",
    data_prefix="CCD",
    time_column="EXPDate",
    read_htr=True,
    development=development,
)

Level1AStack(
    app,
    f"Level1AStackPM{'Dev' if development else ''}",
    rac_bucket_name=rac_bucket_name,
    platform_bucket_name=platform_bucket_name,
    mats_schedule_bucket_name=mats_schedule_bucket_name,
    output_bucket_name=output_bucket_pm,
    code_version=f"{tag} ({repo.head.commit})",
    data_prefix="PM",
    time_column="PMTime",
    read_htr=False,
    development=development,
)

app.synth()
