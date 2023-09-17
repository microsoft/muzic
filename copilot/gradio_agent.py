import uuid
import gradio as gr
import re
import requests
from agent import MusicCoplilotAgent
import soundfile
import argparse


all_messages = []
OPENAI_KEY = ""


def add_message(content, role):
    message = {"role": role, "content": content}
    all_messages.append(message)


def extract_medias(message):
    audio_pattern = re.compile(r"(http(s?):|\/)?([\.\/_\w:-])*?\.(flac|wav|mp3)")
    audio_urls = []
    for match in audio_pattern.finditer(message):
        if match.group(0) not in audio_urls:
            audio_urls.append(match.group(0))

    return audio_urls


def set_openai_key(openai_key):
    global OPENAI_KEY
    OPENAI_KEY = openai_key
    agent._init_backend_from_input(openai_key)
    return OPENAI_KEY, gr.update(visible=True), gr.update(visible=True)


def add_text(messages, message):
    if len(OPENAI_KEY) == 0 or not OPENAI_KEY.startswith("sk-"):
        return messages, "Please set your OpenAI API key first."
    add_message(message, "user")
    messages = messages + [(message, None)]
    audio_urls = extract_medias(message)

    for audio_url in audio_urls:
        if audio_url.startswith("http"):
            ext = audio_url.split(".")[-1]
            name = f"{str(uuid.uuid4()[:4])}.{ext}"
            response = requests.get(audio_url)
            with open(f"{agent.config['src_fold']}/{name}", "wb") as f:
                f.write(response.content)
        messages = messages + [((f"{audio_url} is saved as {name}",), None)]

    return messages, ""


def upload_audio(file, messages):
    file_name = str(uuid.uuid4())[:4]
    audio_load, sr = soundfile.read(file.name)
    soundfile.write(f"{agent.config['src_fold']}/{file_name}.wav", audio_load, samplerate=sr)

    messages = messages + [(None, f"** {file_name}.wav ** uploaded")]
    return messages
        

def bot(messages):
    if len(OPENAI_KEY) == 0 or not OPENAI_KEY.startswith("sk-"):
        return messages
    
    message = agent.chat(messages[-1][0])

    audio_urls = extract_medias(message)
    add_message(message, "assistant")
    messages[-1][1] = message
    for audio_url in audio_urls:
        if not audio_url.startswith("http") and not audio_url.startswith(f"{agent.config['src_fold']}"):
            audio_url = f"{agent.config['src_fold']}/{audio_url}"
        messages = messages + [((None, (audio_url,)))]

    return messages


def clear_all_history(messages):
    agent.clear_history()

    messages = messages + [(("All histories of LLM are cleared", None))]
    return messages

def parse_args():
    parser = argparse.ArgumentParser(description="A path to a YAML file")
    parser.add_argument("--config", type=str, help="a YAML file path.")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    agent = MusicCoplilotAgent(args.config, mode="gradio")

    with gr.Blocks() as demo:
        gr.Markdown("<h2><center>Music Pilot (Dev)</center></h2>")
        with gr.Row():
            openai_api_key = gr.Textbox(
                show_label=False,
                placeholder="Set your OpenAI API key here and press Enter",
                lines=1,
                type="password",
            )
            state = gr.State([])

        chatbot = gr.Chatbot([], elem_id="chatbot", label="music_pilot", visible=False).style(height=500)

        with gr.Row(visible=False) as text_input_raws:
            with gr.Column(scale=0.8):
                txt = gr.Textbox(show_label=False, placeholder="Enter text and press enter or click Run button").style(container=False)
            # with gr.Column(scale=0.1, min_width=0):
            #     run = gr.Button("🏃‍♂️Run")
            with gr.Column(scale=0.1, min_width=0):
                clear_txt = gr.Button("🔄Clear️")
            with gr.Column(scale=0.1, min_width=0):
                btn = gr.UploadButton("🖼️Upload", file_types=["audio"])

        openai_api_key.submit(set_openai_key, [openai_api_key], [openai_api_key, chatbot, text_input_raws])
        clear_txt.click(clear_all_history, [chatbot], [chatbot])

        btn.upload(upload_audio, [btn, chatbot], [chatbot])
        txt.submit(add_text, [chatbot, txt], [chatbot, txt]).then(
            bot, chatbot, chatbot
        )
        
        gr.Examples(
            examples=["Write a piece of lyric about the recent World Cup.",
                        "生成一首古风歌词的中文歌",
                        "Download a song by Jay Zhou for me and separate the vocals and the accompanies.",
                        "Convert the vocals in /b.wav to a violin sound.",
                        "Give me the sheet music and lyrics in the song /a.wav",
                        "近一个月流行的音乐类型",
                        "把c.wav中的人声搭配合适的旋律变成一首歌"
                        ],
            inputs=txt
        )

    demo.launch(share=True)
