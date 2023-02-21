from aws_cdk import Duration, RemovalPolicy, Size, Stack
from aws_cdk.aws_lambda import Architecture, Runtime
from aws_cdk.aws_lambda_event_sources import SqsEventSource
from aws_cdk.aws_lambda_python_alpha import PythonFunction  # type: ignore
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
        output_bucket_name: str,
        lambda_timeout: Duration = Duration.seconds(900),
        queue_retention_period: Duration = Duration.days(14),
        code_version: str = "",
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
            memory_size=4096,
            ephemeral_storage_size=Size.mebibytes(512),
            environment={
                "PLATFORM_BUCKET": platform_bucket_name,
                "OUTPUT_BUCKET": output_bucket_name,
                "HTR_BUCKET": rac_bucket_name,
                "L1A_VERSION": code_version,
            },
        )

        event_queue = Queue(
            self,
            "ProcessCCDQueue",
            visibility_timeout=lambda_timeout,
            removal_policy=RemovalPolicy.RETAIN,
            dead_letter_queue=DeadLetterQueue(
                max_receive_count=1,
                queue=Queue(
                    self,
                    "FailedCCDProcessQueue",
                    retention_period=queue_retention_period,
                )
            )
        )

        rac_bucket.add_object_created_notification(
            SqsDestination(event_queue),
            NotificationKeyFilter(prefix="CCD"),
        )

        level1a_lambda.add_event_source(SqsEventSource(
            event_queue,
            batch_size=1,
        ))

        output_bucket.grant_read_write(level1a_lambda)
        rac_bucket.grant_read(level1a_lambda)
        platform_bucket.grant_read(level1a_lambda)
