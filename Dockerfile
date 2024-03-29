FROM public.ecr.aws/lambda/python:3.9

# Install the function's dependencies using file requirements.txt
# from your project folder.

COPY ./level1a/requirements.txt  .
RUN  pip3 install -r requirements.txt --target "${LAMBDA_TASK_ROOT}"

# Copy function code
COPY ./level1a/handlers/level1a.py ${LAMBDA_TASK_ROOT}
COPY ./level1a/handlers/mats_utils ${LAMBDA_TASK_ROOT}/mats_utils/

# Set the CMD to your handler (could also be done as a parameter override outside of the Dockerfile)
CMD [ "level1a.lambda_handler" ] 
