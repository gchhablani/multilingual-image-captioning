# Multingual Image Captioning

Multilingual Image Captioning using ViT and mBART, pre-trained on WIT dataset.

This project is developed as a part of the 🤗 JAX/Flax community week!

## Summary
- Goal of the project: Present a visio-linguistic model that performs image captioning.
  - Side-goal (Optional): Use the model to test zero-shot/few-shot performance on the VizWiz Image Captioning dataset for the visually challenged.  
- Models we will use: FlaxViT for visual encoder, FlaxmBART for textual decoder
- Datasets will we use: WIT multilingual image-text dataset from Wikipedia. There are other datasets we are considering - GEM/COCO/COCO 12B/VizWiz/Flickr 8k/Flickr 30k
- Should we use a pre-trained model or train a model from scratch? Use the pre-trained encoder/decoders and fine-tune them for this dataset. Because the pre-training requires a huge amount of data for both the models.
- Training Scripts: [Flax Summarization](https://github.com/huggingface/transformers/tree/master/examples/flax/summarization)
- Demo: Possibly a streamlit or Gradio demo which performs image captioning provided with image and language.
- Work Division: No strict work division as such. Everyone is welcome to fix any issues/add features they like. In general, there are several sub-parts which include:
  -  Data loading and preprocessing
    - Cleaning/Image Normalizing
    - Collation/Tokenization 
  -  Model building and testing: (Bhavitvya, Gunjan)
    - Encoder-Decoder Flax? Or create own model?
    - Forward Pass
  -  Training script generation
    - Loss Functions (need to be defined) 
  -  Inference application creation

## Model(s)
- ViT
- mBART

## Dataset(s)

- [WIT](https://github.com/google-research-datasets/wit)
- [CoCo](https://cocodataset.org/#download)
- [VizWiz](https://vizwiz.org/workshops/2021-workshop/)
- [GEM](https://github.com/microsoft/GEM#get-dataset)
- [Common Crawl](https://colab.research.google.com/drive/1IuNfPyS29IBQ15veJc4KyCtBSNM36n7n?usp=sharing)


## Training Scripts
[Flax Summarization](https://github.com/huggingface/transformers/tree/master/examples/flax/summarization) should be close to what we are trying to achieve.

## Challenges
The main challenge of this project lies in deciding the languages of the models, and connecting the ViT model (786 hidden dim) to mBART decoder (1024 hidden dim). The plan is to use a linear projection with or without an activation for that same. Since we will be fine-tuning the model on a large dataset, that probably will be a decent way of doing this. Regarding the dataset, we are planning to go with mBART languages, since the tokenizer automatically those languages. 

## Desired Project Outcome

The model should be able to generate captions for images in multiple languages based on the specification. This can be showcased with a streamlit or gradio app.

Side-goal: Our end use case would be to run this model on a video clip or movie and make it like an accessibility tool for visually challenged people.

## So Far


## Resources and Links
### JAX/Flax tutorials
- [Flax Summarization](https://github.com/huggingface/transformers/tree/master/examples/flax/summarization)

### HuggingFace Instructions
- [JAX/Flax Community Week](https://github.com/huggingface/transformers/tree/master/examples/research_projects/jax-projects#how-to-submit-a-demo)

### Our Notebooks

### Papers

### Misc
- [Forum Post](https://discuss.huggingface.co/t/multilingual-image-captioning/7671)
