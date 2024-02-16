import ray
from ray.util import accelerator

@ray.remote(num_cpus=1, num_gpus=1)
class InferenceWorker:
    def __init__(self, model_name, task_name):
        self.model_name = model_name
        self.task_name = task_name
        self.accelerator = accelerator.create(model_name=model_name, task_name=task_name)

    def inference(self, text):
        inputs = self.accelerator.prepare_inputs(text)
        outputs = self.accelerator.inference_step(inputs)
        return self.accelerator.postprocess_outputs(outputs)

ray.init(address="auto")  # Connect to the Ray cluster

# Initialize InferenceWorker instances on each GPU server
worker_instances = [InferenceWorker.remote("tuner007/mistral-7B", "text-generation") for _ in range(4)]

# Example text for inference
text = "Once upon a time, in a land far, far away..."

# Distribute the inference task among available workers
results = ray.get([worker.inference.remote(text) for worker in worker_instances])

# Do something with the results (e.g., aggregate or process them)
for result in results:
    print(result)
