#!/usr/bin/env python3

from aws_cdk import App

from stacks.level1a_stack import Level1AStack

app = App()

Level1AStack(
    app,
    "Level1AStack",
    rac_bucket_name="ops-payload-level0-v0.1",
    platform_bucket_name="ops-platform-level1a-v0.1",
    output_bucket_name="ops-payload-level1a-v0.1",
)

app.synth()
