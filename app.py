import torch
import gradio as gr
from transformers import pipeline
from huggingface_hub import model_info


MODEL_NAME = "models/hakka_fsr" 
MODEL_NAME_PINYIN = "models/hakka_fsr_pinyin" 
lang = "zh"

device = 0 if torch.cuda.is_available() else "cpu"

pipe = pipeline(
    task="automatic-speech-recognition",
    model=MODEL_NAME,
    chunk_length_s=30,
    device=device,
)
pipe_pinyin = pipeline(
    task="automatic-speech-recognition",
    model=MODEL_NAME_PINYIN,
    chunk_length_s=30,
    device=device,
)



pipe.model.config.forced_decoder_ids = pipe.tokenizer.get_decoder_prompt_ids(language=lang, task="transcribe")
pipe_pinyin.model.config.forced_decoder_ids = pipe_pinyin.tokenizer.get_decoder_prompt_ids(language=lang, task="transcribe")

def transcribe( file_upload, mode,hakka):
    warn_output = ""
    if  (file_upload is None):
        return "ERROR: You have to either use the microphone or upload an audio file"

    file = file_upload
    if mode == "客語漢字":
        text = pipe(file, generate_kwargs={"language":"<|zh|>", "task":"transcribe"}, batch_size=16)["text"]
    else:
        text = pipe_pinyin(file, generate_kwargs={"language":"<|zh|>", "task":"transcribe"}, batch_size=16)["text"]

    return text

examples = [
    ["samples/F0100001D3038_90_02.wav","客語漢字", "早就翕在吾人生个相簿裡肚了。"],
    ["samples/M0130001D3046_19_02.wav","客語漢字", "看起來無打過个紙炮仔，"],
    ["samples/F0100001D3038_90_02.wav", "客語拼音", "zo31 qiu55 hib2 di55 nga24 ngin11 sen24 ge55 xiong55 pu24 di24 du31 le31 。"],
    ["samples/M0130001D3046_19_02.wav", "客語拼音", "kon55 hi31 loi11 mo11 da31 go55 ge55 zii31 pau55 e31 ，"],
    
]


demo = gr.Blocks()

mf_transcribe = gr.Interface(
    fn=transcribe,
    inputs=[
        #gr.inputs.Audio(source="microphone", type="filepath", optional=True),
        gr.inputs.Audio(source="upload", type="filepath", optional=True),
        gr.Textbox(placeholder="Enter a positive or negative sentence here..."),
        gr.Textbox(placeholder="Enter a positive or negative sentence here...")
    ],
    outputs="text",
    layout="horizontal",
    theme="huggingface",
    title="Transcribe Audio",
    examples=examples,
    description=(
        "Transcribe long-form microphone or audio inputs with the click of a button! Demo uses the the fine-tuned"

    ),
    allow_flagging="never",
)





with demo:
    gr.TabbedInterface([mf_transcribe], ["Trsudo anscribe Audio"])

demo.launch(enable_queue=True,server_name="0.0.0.0",server_port=6003)

