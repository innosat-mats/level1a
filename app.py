#!/usr/bin/env python3

from aws_cdk import App

from stacks.level1a_stack import Level1AStack

app = App()

Level1AStack(
    app,
    "Level1AStack",
    rac_bucket_name="mats-l0-artifacts",
    platform_bucket_name="mats-l1a-platform-parquet",
    output_bucket_name="mats-l1a",
)

app.synth()
