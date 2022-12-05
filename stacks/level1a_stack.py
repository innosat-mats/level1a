
from aws_cdk import Duration, Size, Stack
from aws_cdk.aws_events import Rule, Schedule
from aws_cdk.aws_lambda import Architecture, Runtime
from aws_cdk.aws_lambda_python_alpha import PythonFunction  # type: ignore
from aws_cdk.aws_s3 import Bucket
from constructs import Construct


class Level1AStack(Stack):
    def __init__(
        self,
        scope: Construct,
        id: str,
        rac_bucket_name: str,
        platform_bucket_name: str,
        output_bucket_name: str,
        lambda_schedule: Schedule = Schedule.rate(Duration.hours(12)),
        lambda_timeout: Duration = Duration.seconds(300),
        **kwargs
    ) -> None:
        super().__init__(scope, id, **kwargs)

        rac_bucket = Bucket.from_bucket_name(
            self,
            "Level1ARACBucket",
            rac_bucket_name,
        )

        platform_bucket = Bucket.from_bucket_name(
            self,
            "Level1APlatformBucket",
            platform_bucket_name,
        )

        output_bucket = Bucket.from_bucket_name(
            self,
            "Level1AOutputBucket",
            output_bucket_name,
        )

        level1a_lambda = PythonFunction(
            self,
            "Level1ALambda",
            entry="level1a",
            handler="lambda_handler",
            index="handlers/level1a.py",
            timeout=lambda_timeout,
            architecture=Architecture.X86_64,
            runtime=Runtime.PYTHON_3_9,
            memory_size=512,
            ephemeral_storage_size=Size.mebibytes(512),
            environment={
                "RAC_BUCKET": rac_bucket_name,
                "PLATFORM_BUCKET": platform_bucket_name,
                "OUTPUT_BUCKET": output_bucket_name,
            },
        )

        rule = Rule(
            self,
            "Level1ALambdaSchedule",
            schedule=lambda_schedule,
        )
        rule.add_target(level1a_lambda)

        output_bucket.grant_read_write(level1a_lambda)
        rac_bucket.grant_read(level1a_lambda)
        platform_bucket.grant_read(level1a_lambda)
