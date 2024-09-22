import os, json, requests, random, runpod

import torch
from diffusers import AutoencoderKL
from diffusers.models.model_loading_utils import load_state_dict
from controlnet_union import ControlNetModel_Union
from pipeline_fill_sd_xl import StableDiffusionXLFillPipeline
from PIL import Image, ImageDraw

with torch.inference_mode():
    config = ControlNetModel_Union.load_config("/content/model/union/config_promax.json")
    controlnet_model = ControlNetModel_Union.from_config(config)
    state_dict = load_state_dict("/content/model/union/diffusion_pytorch_model_promax.safetensors")
    model, _, _, _, _ = ControlNetModel_Union._load_pretrained_model(controlnet_model, state_dict, "/content/model/union/diffusion_pytorch_model_promax.safetensors", "/content/model/union")
    model.to(device="cuda", dtype=torch.float16)
    vae = AutoencoderKL.from_pretrained("/content/model/vae-fix", torch_dtype=torch.float16).to("cuda")
    pipe = StableDiffusionXLFillPipeline.from_pretrained("/content/model/lightning", torch_dtype=torch.float16, vae=vae, controlnet=model, variant="fp16").to("cuda")

def infer(image, width, height, overlap_width, num_inference_steps, prompt_input=None):
    source = image
    target_size = (width, height)
    overlap = overlap_width
    if source.width < target_size[0] and source.height < target_size[1]:
        scale_factor = min(target_size[0] / source.width, target_size[1] / source.height)
        new_width = int(source.width * scale_factor)
        new_height = int(source.height * scale_factor)
        source = source.resize((new_width, new_height), Image.LANCZOS)
    if source.width > target_size[0] or source.height > target_size[1]:
        scale_factor = min(target_size[0] / source.width, target_size[1] / source.height)
        new_width = int(source.width * scale_factor)
        new_height = int(source.height * scale_factor)
        source = source.resize((new_width, new_height), Image.LANCZOS)
    margin_x = (target_size[0] - source.width) // 2
    margin_y = (target_size[1] - source.height) // 2
    background = Image.new('RGB', target_size, (255, 255, 255))
    background.paste(source, (margin_x, margin_y))
    mask = Image.new('L', target_size, 255)
    mask_draw = ImageDraw.Draw(mask)
    mask_draw.rectangle([
        (margin_x + overlap, margin_y + overlap),
        (margin_x + source.width - overlap, margin_y + source.height - overlap)
    ], fill=0)
    cnet_image = background.copy()
    cnet_image.paste(0, (0, 0), mask)
    final_prompt = "high quality"
    if prompt_input and prompt_input.strip():
        final_prompt += ", " + prompt_input
    (
        prompt_embeds,
        negative_prompt_embeds,
        pooled_prompt_embeds,
        negative_pooled_prompt_embeds,
    ) = pipe.encode_prompt(final_prompt, "cuda", True)
    results = []
    for image in pipe(
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
        image=cnet_image,
        num_inference_steps=num_inference_steps
    ):
        results.append((cnet_image, image))
    image = image.convert("RGBA")
    cnet_image.paste(image, (0, 0), mask)
    results.append((background, cnet_image))
    return results

def download_file(url, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    file_name = url.split('/')[-1]
    file_path = os.path.join(save_dir, file_name)
    response = requests.get(url)
    response.raise_for_status()
    with open(file_path, 'wb') as file:
        file.write(response.content)
    return file_path

@torch.inference_mode()
def generate(input):
    values = input["input"]

    input_image = values['input_image']
    input_image = download_file(url=input_image, save_dir='/content')
    image = Image.open(input_image).convert("RGB")
    width = values['width']
    height = values['height']
    overlap_width = values['overlap_width']
    num_inference_steps = values['num_inference_steps']
    prompt_input = values['prompt_input']
    # width_height = values['width_height']
    # width, height = map(int, width_height.split('x'))
    output_image = infer(image, width, height, overlap_width, num_inference_steps, prompt_input)
    output_image[num_inference_steps+1][1].save("/content/outpaint_tost.png")

    result = "/content/outpaint_tost.png"
    try:
        notify_uri = values['notify_uri']
        del values['notify_uri']
        notify_token = values['notify_token']
        del values['notify_token']
        discord_id = values['discord_id']
        del values['discord_id']
        if(discord_id == "discord_id"):
            discord_id = os.getenv('com_camenduru_discord_id')
        discord_channel = values['discord_channel']
        del values['discord_channel']
        if(discord_channel == "discord_channel"):
            discord_channel = os.getenv('com_camenduru_discord_channel')
        discord_token = values['discord_token']
        del values['discord_token']
        if(discord_token == "discord_token"):
            discord_token = os.getenv('com_camenduru_discord_token')
        job_id = values['job_id']
        del values['job_id']
        default_filename = os.path.basename(result)
        with open(result, "rb") as file:
            files = {default_filename: file.read()}
        payload = {"content": f"{json.dumps(values)} <@{discord_id}>"}
        response = requests.post(
            f"https://discord.com/api/v9/channels/{discord_channel}/messages",
            data=payload,
            headers={"Authorization": f"Bot {discord_token}"},
            files=files
        )
        response.raise_for_status()
        result_url = response.json()['attachments'][0]['url']
        notify_payload = {"jobId": job_id, "result": result_url, "status": "DONE"}
        web_notify_uri = os.getenv('com_camenduru_web_notify_uri')
        web_notify_token = os.getenv('com_camenduru_web_notify_token')
        if(notify_uri == "notify_uri"):
            requests.post(web_notify_uri, data=json.dumps(notify_payload), headers={'Content-Type': 'application/json', "Authorization": web_notify_token})
        else:
            requests.post(web_notify_uri, data=json.dumps(notify_payload), headers={'Content-Type': 'application/json', "Authorization": web_notify_token})
            requests.post(notify_uri, data=json.dumps(notify_payload), headers={'Content-Type': 'application/json', "Authorization": notify_token})
        return {"jobId": job_id, "result": result_url, "status": "DONE"}
    except Exception as e:
        error_payload = {"jobId": job_id, "status": "FAILED"}
        try:
            if(notify_uri == "notify_uri"):
                requests.post(web_notify_uri, data=json.dumps(error_payload), headers={'Content-Type': 'application/json', "Authorization": web_notify_token})
            else:
                requests.post(web_notify_uri, data=json.dumps(error_payload), headers={'Content-Type': 'application/json', "Authorization": web_notify_token})
                requests.post(notify_uri, data=json.dumps(error_payload), headers={'Content-Type': 'application/json', "Authorization": notify_token})
        except:
            pass
        return {"jobId": job_id, "result": f"FAILED: {str(e)}", "status": "FAILED"}
    finally:
        if os.path.exists(result):
            os.remove(result)
        if os.path.exists(input_image):
            os.remove(input_image)

runpod.serverless.start({"handler": generate})