import ray
from transformers import pipeline

# Initialize Ray
ray.init(address="auto")  # Connect to the Ray cluster

# Define a function for Q&A inference
@ray.remote(num_cpus=1, num_gpus=1)
def qna_inference(question, context):
    # Load the pre-trained model for Q&A
    qna_pipeline = pipeline("question-answering", model="Mistral-7B-Instruct-v0.1")

    # Perform Q&A inference
    result = qna_pipeline(question=question, context=context)

    return result

# Example question and context for inference
question = "What is the capital of France?"
context = "Paris is the capital of France."

# Perform distributed inference
result = ray.get(qna_inference.remote(question, context))

# Print the result
print(result)