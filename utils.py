from processing import postprocess
from keybert_ import keybert

def text_to_prompt(text_prompts, from_keywords, keywords_weighted):
    if from_keywords:
        postprocess_keywords = postprocess(keybert(text_prompts, words=20), top_words=5)
        if keywords_weighted:
            text_prompts = ""
            for i, (key, val) in enumerate(postprocess_keywords.items()):
                text_prompts += '{}:{}|'.format(key, val)
            text_prompts = text_prompts[0:-1]
        else:
            text_prompts = ""
            for key, val in postprocess_keywords.items():
                text_prompts += key + ", "
            text_prompts = text_prompts.rstrip()[0:-1]
    return text_prompts

def concatenate_apply_static_w(title_prompt, text_prompt, emotion_prompt, w_title, w_text, w_emo):
    prompts = ""
    if title_prompt != "":
        if ":" not in title_prompt:
            prompts += title_prompt + ":{} |".format(w_title)
        else:
            prompts += title_prompt + " |"
    if text_prompt != "":
        if ":" not in text_prompt:
            prompts += text_prompt + ":{} |".format(w_text)
        else:
            prompts += text_prompt + " |"
    if emotion_prompt != "":
        if ":" not in emotion_prompt:
            prompts += emotion_prompt + ":{}".format(w_emo)
        else:
            prompts += emotion_prompt
    return prompts

def to_vqgan_prompts(text_prompts, target_images):
    text_prompts_vqgan = [phrase.strip() for phrase in text_prompts.split("|")]
    if text_prompts_vqgan == ['']:
        text_prompts_vqgan = []
    if target_images == "" or not target_images:
        target_images_vqgan = []
    else:
        target_images = target_images.split("|")
        target_images_vqgan = [image.strip() for image in target_images]
    return text_prompts_vqgan, target_images_vqgan