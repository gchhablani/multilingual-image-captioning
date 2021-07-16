import jax
from PIL import Image
import requests
from transformers import CLIPProcessor
from transformers import MBart50TokenizerFast
from models.flax_clip_vision_mbart.modeling_clip_vision_mbart import FlaxCLIPVisionMBartForConditionalGeneration

processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

inputs = processor(images=image, return_tensors="jax", padding=True)

tokenizer_de = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50", src_lang="en_XX", tgt_lang="de_DE")
with tokenizer_de.as_target_tokenizer():
    input_token = tokenizer_de("Neon-Kaffeeladen Schild auf einer dunklen Ziegelwand", max_length=64, padding="max_length", return_tensors="np", truncation=True)
# input_token = tokenizer_de(caption, max_length=64, padding="max_length", return_tensors="np", truncation=True)

model= FlaxCLIPVisionMBartForConditionalGeneration.from_clip_vision_mbart_pretrained('openai/clip-vit-base-patch32', 'facebook/mbart-large-50', mbart_from_pt=True)
outputs = model(inputs['pixel_values'], input_token['input_ids'], input_token['attention_mask'])
output_ids = model.generate(input_ids=inputs['pixel_values'], decoder_start_token_id=tokenizer_de.lang_code_to_id['de_DE'], max_length=64, num_beams=4)
# print(output_ids.shape)
# print(output_ids)
print(tokenizer_de.batch_decode(output_ids[0], skip_special_tokens=True, max_length=64))  # here 0 but change later

decoder_input_ids = shift_tokens_right_fn(jnp.array(input_token["input_ids"]), config.pad_token_id, config.decoder_start_token_id)
# print(outputs)
# print(outputs.keys())

# logits shape: (8, 64, 250054)
# labels shape: (8, 64)
