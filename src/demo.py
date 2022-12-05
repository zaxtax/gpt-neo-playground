from transformers import AutoTokenizer, AutoModelForCausalLM

models = ["Salesforce/codegen-2B-mono", "facebook/incoder-1B", "EleutherAI/gpt-neo-2.7B", "EleutherAI/gpt-neo-125M"]

# Only load these models on a machine with lots of VRAM and disc space
big_models = ["Salesforce/codegen-6B-mono", "Salesforce/codegen-16B-mono", "facebook/incoder-6B"]

tokenizer = AutoTokenizer.from_pretrained(models[0])
model = AutoModelForCausalLM.from_pretrained(models[0])

def inference(prompt, temperature=0.2, top_p=0.95, max_length=128, device="cuda"):
    if not torch.cuda.is_available():
        device = "cpu"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    input_ids = input_ids.to(device)
    model.to(device)
    gen_tokens = model.generate(
        input_ids,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        max_length=max_length,
    )
    gen_text = tokenizer.batch_decode(gen_tokens)[0]
    return gen_text
