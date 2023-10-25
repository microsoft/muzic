import uuid
import os
import gradio as gr
import re
import requests
from agent import MusicAgent
import soundfile
import argparse


all_messages = []
OPENAI_KEY = ""


def add_message(content, role):
    message = {"role": role, "content": content}
    all_messages.append(message)


def extract_medias(message):
    # audio_pattern = re.compile(r"(http(s?):|\/)?([\.\/_\w:-])*?\.(flac|wav|mp3)")
    audio_pattern = re.compile(r"(http(s?):|\/)?[a-zA-Z0-9\/.:-]*\.(flac|wav|mp3)")
    symbolic_button = re.compile(r"(http(s?):|\/)?[a-zA-Z0-9\/.:-]*\.(mid)")

    audio_urls = []
    for match in audio_pattern.finditer(message):
        if match.group(0) not in audio_urls:
            audio_urls.append(match.group(0))

    symbolic_urls = []
    for match in symbolic_button.finditer(message):
        if match.group(0) not in symbolic_urls:
            symbolic_urls.append(match.group(0))

    return list(set(audio_urls)), list(set(symbolic_urls))


def set_openai_key(openai_key):
    global OPENAI_KEY
    OPENAI_KEY = openai_key
    agent._init_backend_from_input(openai_key)
    if not OPENAI_KEY.startswith("sk-"):
        return "OpenAI API Key starts with sk-", gr.update(visible=False)

    return OPENAI_KEY, gr.update(visible=True)


def add_text(messages, message):
    add_message(message, "user")
    messages = messages + [(message, None)]
    audio_urls, _ = extract_medias(message)

    for audio_url in audio_urls:
        if audio_url.startswith("http"):
            ext = audio_url.split(".")[-1]
            name = f"{str(uuid.uuid4()[:4])}.{ext}"
            response = requests.get(audio_url)
            with open(f"{agent.config['src_fold']}/{name}", "wb") as f:
                f.write(response.content)
            messages = messages + [(None, f"{audio_url} is saved as {name}")]

    return messages, ""


def upload_audio(file, messages):
    file_name = str(uuid.uuid4())[:4]
    audio_load, sr = soundfile.read(file.name)
    soundfile.write(f"{agent.config['src_fold']}/{file_name}.wav", audio_load, samplerate=sr)

    messages = messages + [(None, f"Audio is stored in wav format as ** {file_name}.wav **"), 
                           (None, (f"{agent.config['src_fold']}/{file_name}.wav",))]
    return messages
        

def bot(messages):
    message, results = agent.chat(messages[-1][0])

    audio_urls, symbolic_urls = extract_medias(message)
    add_message(message, "assistant")
    messages[-1][1] = message
    for audio_url in audio_urls:
        if not audio_url.startswith("http") and not audio_url.startswith(agent.config['src_fold']):
            audio_url =  os.path.join(agent.config['src_fold'], audio_url)
        messages = messages + [(None, f"** {audio_url.split('/')[-1]} **"),
                                (None, (audio_url,))]
    
    for symbolic_url in symbolic_urls:
        if not symbolic_url.startswith(agent.config['src_fold']):
            symbolic_url = os.path.join(agent.config['src_fold'], symbolic_url)
        
        try:
            os.system(f"midi2ly {symbolic_url} -o {symbolic_url}.ly; lilypond -f png -o {symbolic_url} {symbolic_url}.ly")
        except:
            continue
        messages = messages + [(None, f"** {symbolic_url.split('/')[-1]} **")]
        
        if os.path.exists(f"{symbolic_url}.png"):
            messages = messages + [ (None, (f"{symbolic_url}.png",))]
        else:
            s_page = 1
            while os.path.exists(f"{symbolic_url}-page{s_page}.png"):
                messages = messages + [ (None, (f"{symbolic_url}-page{s_page}.png",))]
                s_page += 1
        
    def truncate_strings(obj, max_length=128):
        if isinstance(obj, str):
            if len(obj) > max_length:
                return obj[:max_length] + "..."
            else:
                return obj
        elif isinstance(obj, dict):
            return {key: truncate_strings(value, max_length) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [truncate_strings(item, max_length) for item in obj]
        else:
            return obj

    results = truncate_strings(results)
    results = sorted(results.items(), key=lambda x: int(x[0]))
    response = [(None, "\n\n".join([f"Subtask {r[0]}:\n{r[1]}" for r in results]))]

    return messages, response


def clear_all_history(messages):
    agent.clear_history()

    messages = messages + [((None, "All LLM history cleared"))]
    return messages

def parse_args():
    parser = argparse.ArgumentParser(description="music agent config")
    parser.add_argument("-c", "--config", type=str, help="a YAML file path.")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    agent = MusicAgent(args.config, mode="gradio")

    with gr.Blocks() as demo:
        gr.HTML("""
                <h1 align="center" style=" display: flex; flex-direction: row; justify-content: center; font-size: 25pt; ">🎧 Music Agent</h1>
                <h3>This is a demo page for Music Agent, a project that uses LLM to integrate music tools. For specific functions, please refer to the examples given below, or refer to the instructions in Github.</h3>
                <h3>Make sure the uploaded audio resource is in flac|wav|mp3 format.</h3>
                <h3>Due to RPM limitations, Music Agent requires an OpenAI key for the paid version.</h3>
                <div style="display: flex;"><a href='https://github.com/microsoft/muzic/tree/main/copilot'><img src='https://img.shields.io/badge/Github-Code-blue'></a></div>
                """)
        
        with gr.Row():
            openai_api_key = gr.Textbox(
                show_label=False,
                placeholder="Set your OpenAI API key here and press Enter",
                lines=1,
                type="password",
            )
            state = gr.State([])

        with gr.Row(visible=False) as interact_window:

            with gr.Column(scale=0.7, min_width=500):
                chatbot = gr.Chatbot([], elem_id="chatbot", label="Music-Agent Chatbot").style(height=500)

                with gr.Tab("User Input"):
                    with gr.Row(scale=1):
                        with gr.Column(scale=0.6):
                            txt = gr.Textbox(show_label=False, placeholder="Press ENTER or click the Run button. You can start by asking 'What can you do?'").style(container=False)
                        with gr.Column(scale=0.1, min_width=0):
                            run = gr.Button("🏃‍♂️Run")
                        with gr.Column(scale=0.1, min_width=0):
                            clear_txt = gr.Button("🔄Clear️")
                        with gr.Column(scale=0.2, min_width=0):
                            btn = gr.UploadButton("☁️Upload Audio", file_types=["audio"])

            with gr.Column(scale=0.3, min_width=300):
                with gr.Tab("Intermediate Results"):
                    response = gr.Chatbot([], label="Current Progress").style(height=400)

        openai_api_key.submit(set_openai_key, [openai_api_key], [openai_api_key, interact_window])
        clear_txt.click(clear_all_history, [chatbot], [chatbot])

        btn.upload(upload_audio, [btn, chatbot], [chatbot])
        run.click(add_text, [chatbot, txt], [chatbot, txt]).then(
            bot, chatbot, [chatbot, response]
        )
        txt.submit(add_text, [chatbot, txt], [chatbot, txt]).then(
            bot, chatbot, [chatbot, response]
        )
        
        gr.Examples(
            examples=["What can you do?",
                        "Write a piece of lyric about the recent World Cup.",
                        "生成一首古风歌词的中文歌",
                        "Download a song by Jay Chou for me and separate the vocals and the accompanies.",
                        "Convert the vocals in /b.wav to a violin sound.",
                        "Give me the sheet music and lyrics in the song /a.wav",
                        "近一个月流行的音乐类型",
                        "把c.wav中的人声搭配合适的旋律变成一首歌"
                        ],
            inputs=txt
        )

    demo.launch(share=True)
