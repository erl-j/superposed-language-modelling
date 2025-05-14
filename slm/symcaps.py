#%%
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
import torch

df = pd.read_csv("../data/musiccaps-public.csv")

# Print 3 random captions from the dataset
print("3 random captions from MusicCaps:")
random_captions = df.sample(n=3)['caption']
for i, caption in enumerate(random_captions, 1):
    print(f"\n{i}. {caption}")

#%%

# Load the model and tokenizer
# model_name = "huihui-ai/Qwen2.5-72B-Instruct-abliterated"
model_name = "Qwen/Qwen2-7B-Instruct"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)



#%%
# Initialize conversation context
initial_messages = [
    {"role": "system", "content": "Your task is to rewrite this caption, removing references to mixing and production. Focus strictly on the musical content."}
]

# Create a list to store results
processed_captions = []

# Process each caption
for idx, row in df.iterrows():
    caption = row['caption']
    
    # Create the full message list
    messages = initial_messages + [
        {"role": "user", "content": caption}
    ]
    
    # Tokenize the messages
    input_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    input_ids = torch.tensor([input_ids], device=model.device)
    
    # Generate response
    outputs = model.generate(
        input_ids=input_ids,
        max_new_tokens=512,
        temperature=1.0,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id
    )
    
    # Decode response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Store results
    processed_captions.append({
        'original_caption': caption,
        'processed_caption': response
    })
    
    # Print progress
    if idx % 10 == 0:
        print(f"Processed {idx} captions")

    # Print example of original and processed caption
    print("\nExample of caption processing:")
    print("Original:", processed_captions[-1]['original_caption'])
    print("Processed:", processed_captions[-1]['processed_caption'])

# Convert results to DataFrame
results_df = pd.DataFrame(processed_captions)

# Save results
results_df.to_csv("../data/processed_captions.csv", index=False)

# %%
