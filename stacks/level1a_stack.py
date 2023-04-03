from aws_cdk import Duration, RemovalPolicy, Size, Stack
from aws_cdk.aws_lambda import (
    Architecture,
    DockerImageCode,
    DockerImageFunction,
)
from aws_cdk.aws_lambda_event_sources import SqsEventSource
from aws_cdk.aws_s3 import Bucket, NotificationKeyFilter
from aws_cdk.aws_s3_notifications import SqsDestination
from aws_cdk.aws_sqs import DeadLetterQueue, Queue
from constructs import Construct


class Level1AStack(Stack):
    def __init__(
        self,
        scope: Construct,
        id: str,
        rac_bucket_name: str,
        platform_bucket_name: str,
        mats_schedule_bucket_name: str,
        output_bucket_name: str,
        data_prefix: str,
        time_column: str,
        read_htr: bool,
        lambda_timeout: Duration = Duration.minutes(15),
        queue_retention_period: Duration = Duration.days(14),
        message_timeout: Duration = Duration.hours(12),
        message_attempts: int = 4,
        code_version: str = "",
        development: bool = False,
        **kwargs
    ) -> None:
        super().__init__(scope, id, **kwargs)

        rac_bucket = Bucket.from_bucket_name(
            self,
            f"Level1ARACBucket{'Dev' if development else ''}",
            rac_bucket_name,
        )

        platform_bucket = Bucket.from_bucket_name(
            self,
            f"Level1APlatformBucket{'Dev' if development else ''}",
            platform_bucket_name,
        )

        mats_schedule_bucket = Bucket.from_bucket_name(
            self,
            f"Level1AMatsScheduleBucket{'Dev' if development else ''}",
            mats_schedule_bucket_name,
        )

        output_bucket = Bucket.from_bucket_name(
            self,
            f"Level1AOutputBucket{'Dev' if development else ''}",
            output_bucket_name,
        )

        environment = {
            "PLATFORM_BUCKET": platform_bucket_name,
            "MATS_SCHEDULE_BUCKET": platform_bucket_name,
            "OUTPUT_BUCKET": output_bucket_name,
            "L1A_VERSION": code_version,
            "DATA_PREFIX": data_prefix,
            "TIME_COLUMN": time_column,
        }
        if read_htr:
            environment["HTR_BUCKET"] = rac_bucket_name

        level1a_lambda = DockerImageFunction(
            self,
            f"Level1ALambda{data_prefix}{'Dev' if development else ''}",
            code=DockerImageCode.from_image_asset("."),
            timeout=lambda_timeout,
            architecture=Architecture.X86_64,
            memory_size=4096,
            ephemeral_storage_size=Size.mebibytes(512),
            environment=environment,
        )

        event_queue = Queue(
            self,
            f"Process{data_prefix}Queue{'Dev' if development else ''}",
            visibility_timeout=message_timeout,
            removal_policy=RemovalPolicy.RETAIN,
            dead_letter_queue=DeadLetterQueue(
                max_receive_count=message_attempts,
                queue=Queue(
                    self,
                    f"Failed{data_prefix}ProcessQueue{'Dev' if development else ''}",  # noqa: E501
                    retention_period=queue_retention_period,
                )
            )
        )

        rac_bucket.add_object_created_notification(
            SqsDestination(event_queue),
            NotificationKeyFilter(prefix=data_prefix),
        )

        level1a_lambda.add_event_source(SqsEventSource(
            event_queue,
            batch_size=1,
        ))

        output_bucket.grant_read_write(level1a_lambda)
        rac_bucket.grant_read(level1a_lambda)
        platform_bucket.grant_read(level1a_lambda)
        mats_schedule_bucket.grant_read(level1a_lambda)
