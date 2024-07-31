# Define variables
IMAGE_URI = 
FUNCTION_NAME = vision-process
ROLE_ARN = 

# Login to ECR
login-ecr:
	aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <account-id>.dkr.ecr.us-east-1.amazonaws.com

# Build the Docker image
build:
	docker buildx build --platform linux/amd64 -t vision-tools .

# Tag the Docker image
tag:
	docker tag vision-tools:latest $(IMAGE_URI)

# Push the Docker image to ECR
push:
	docker push $(IMAGE_URI)

update:
	aws lambda update-function-code --function-name $(FUNCTION_NAME) --image-uri $(IMAGE_URI)

# Create the Lambda function
create-function: login-ecr build tag push
    aws lambda create-function --function-name $(FUNCTION_NAME) \
      --package-type Image \
      --code ImageUri=$(IMAGE_URI) \
      --role $(ROLE_ARN)

# Update the Lambda function with the new image
update-function: login-ecr build tag push update

# Test the Lambda function
test-function:
	aws lambda invoke --function-name $(FUNCTION_NAME) \
      --payload '{"body": "{\"video_url\": \"https://<base-s3>.s3.amazonaws.com/video_test.mp4\"}"}' response.json --log-type Tail --query 'LogResult' --output text | base64 -d
	cat response.json

# PHONY target to avoid conflicts with filenames
.PHONY: login-ecr build tag push create-function update-function test-function
