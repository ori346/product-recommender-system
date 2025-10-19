## Deploy LLM model for review summarization.
For the model registry feature, we need to deploy an LLM model. We can do this using OpenShift AI tools.

1. **Deploy MinIO and Create a Data Connection**  
   Set up MinIO on your OpenShift cluster and establish a data connection:  
   👉 [Deploy MinIO and Create Data Connection](https://ai-on-openshift.io/tools-and-applications/minio/minio/)

2. **Download and upload the model to bucket**
Use the [Jupiter notebook](./model_to_bucket.ipynb) in OpenShift Workbench to download and upload your favorite ollama model to the MinIO bucket. You will need to provide a Hugging Face token if required.

3. Via OpenShift AI go to Data science projects &rarr; select your project &rarr; Models &rarr; Select single-model &rarr; Deploy model &rarr; Choose "Make deployed models available through an external route". It is recommended to add the argument to Configuration parameters `--max-model-len=8192`. Once you fill all the requierd you can press Deploy. 

4. Once the model deploy you can test that the model response by
```bash
curl -X POST http://<YOUR MODEL ENDPOINT>/chat/completions -H "Content-Type: application/json"   -d '{
    "model": "<MODEL NAME>",
    "messages": [
      { "role": "system", "content": "You are a helpful assistant." },
      { "role": "user", "content": "solve for x: x + 0.49 = 0.11" }
    ],
    "stream": false
  }'
```

The endpoint should be in the form: `http://<MODEL>-<NAMESPACE>.apps.ai-<CLUSTER>.kni.syseng.devcluster.openshift.com/v1`

Example for model name: `meta-llama/Llama-3.1-8B-Instruct`

5. When deploying the recommender system, add the flags `OLLAMA_MODEL=<MODEL NAME> MODEL_ENDPOINT=<MODEL ENDPOINT>` to the make command.
