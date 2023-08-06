""" Agent for CLI or APPs"""

import io
import os
import sys
import time
import re
import json
import logging
import yaml
import threading
import argparse
import pdb

import semantic_kernel as sk
from semantic_kernel.connectors.ai.open_ai import AzureTextCompletion, OpenAITextCompletion

from model_utils import lyric_format
from plugins import get_task_map, init_plugins

class MusicCoplilotAgent:
    """
    Attributes:
        config_path: A path to a YAML file, referring to the example config.yaml
        mode: Supports "cli" or "gradio", determining when to load the LLM backend.
    """
    def __init__(
            self,
            config_path: str,
            mode: str = "cli",
            ):
        self.config = yaml.load(open(config_path, "r"), Loader=yaml.FullLoader)
        os.makedirs("logs", exist_ok=True)
        self.src_fold = self.config["src_fold"]
        os.makedirs(self.src_fold, exist_ok=True)

        self._init_logger()
        self.kernel = sk.Kernel()

        self.task_map = get_task_map()
        self.pipes = init_plugins(self.config)

        if mode == "cli":
            self._init_backend_from_env()


    def _init_logger(self):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)

        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        if not self.config["debug"]:
            handler.setLevel(logging.CRITICAL)
        self.logger.addHandler(handler)

        log_file = self.config["log_file"]
        if log_file:
            filehandler = logging.FileHandler(log_file)
            filehandler.setLevel(logging.DEBUG)
            filehandler.setFormatter(formatter)
            self.logger.addHandler(filehandler)

    def _init_semantic_kernel(self):
        skills_directory = os.path.join(os.path.dirname(os.path.realpath(__file__)), "skills")
        copilot_funcs = self.kernel.import_semantic_skill_from_directory(skills_directory, "MusicCopilot")
        
        # task planning
        self.task_planner = copilot_funcs["TaskPlanner"]
        self.task_context = self.kernel.create_new_context()
        self.task_context["history"] = ""

        # model selection
        self.tool_selector = copilot_funcs["ToolSelector"]
        self.tool_context = self.kernel.create_new_context()
        self.tool_context["history"] = ""
        self.tool_context["tools"] = ""

        # response
        self.responder = copilot_funcs["Responder"]
        self.response_context = self.kernel.create_new_context()
        self.response_context["history"] = ""
        self.response_context["processes"] = ""

        # chat
        self.chatbot = copilot_funcs["ChatBot"]
        self.chat_context = self.kernel.create_new_context()
        self.chat_context["history"] = ""

    def clear_history(self):
        self.task_context["history"] = ""
        self.tool_context["history"] = ""
        self.response_context["history"] = ""
        self.chat_context["history"] = ""

    def _init_backend_from_env(self):
        # Configure AI service used by the kernel
        if self.config["use_azure_openai"]:
            deployment, api_key, endpoint = sk.azure_openai_settings_from_dot_env()
            self.kernel.add_text_completion_service("dv", AzureTextCompletion(deployment, endpoint, api_key))
        else:
            api_key, org_id = sk.openai_settings_from_dot_env()
            self.kernel.add_text_completion_service("dv", OpenAITextCompletion(self.config["model"], api_key, org_id))
        
        self._init_semantic_kernel()
        self._init_task_context()
        self._init_tool_context()

    def _init_backend_from_input(self, api_key):
        # Only OpenAI api is supported in Gradio demo
        self.kernel.add_text_completion_service("dv", OpenAITextCompletion(self.config["model"], api_key, ""))
        
        self._init_semantic_kernel()
        self._init_task_context()
        self._init_tool_context()

    def _init_task_context(self):
        self.task_context["tasks"] = json.dumps(list(self.task_map.keys()))

    def _init_tool_context(self):
        self.tool_context["tools"] = json.dumps(
            [{"id": pipe.id, "attr": pipe.get_attributes()} for pipe in self.pipes.values()]
        )

    def update_tool_attributes(self, pipe_id, **kwargs):
        self.pipes[pipe_id].update_attributes(kwargs)
        self._init_tool_context()

    def model_inference(self, model_id, command, device="cpu"):
        output = self.pipes[model_id].inference(command["args"], command["task"], device)
        
        locals = []
        for result in output:
            if "audio" in result or "sheet_music" in result:
                locals.append(result)
        
        if len(locals) > 0:
            self.task_context["history"] += f"In this task, <GENERATED>-{command['id']}: {json.dumps(locals)}. "

        return output
    
    def skillchat(self, input_text, chat_function, context):
        context["input"] = input_text
        answer = chat_function.invoke(context=context)
        answer = str(answer).strip()
        context["history"] += f"\nuser: {input_text}\nassistant: {answer}\n"

        # Manage history
        context["history"] = ' '.join(context["history"].split()[-self.config["history_len"]:])

        return answer
    
    def fix_depth(self, tasks):
        for task in tasks:
            task["dep"] = list(set(re.findall(r"<GENERATED>-([0-9]+)", json.dumps(task))))
            task["dep"] = [int(d) for d in task["dep"]]
            if len(task["dep"]) == 0:
                task["dep"] = [-1]
        
        return tasks

    def collect_result(self, command, choose, inference_result):
        result = {"task": command}
        result["inference result"] = inference_result
        result["choose model result"] = choose
        self.logger.debug(f"inference result: {inference_result}")
        return result

    def run_task(self, input_text, command, results):
        id = command["id"]
        args = command["args"]
        task = command["task"]
        deps = command["dep"]

        if deps[0] != -1:
            dep_tasks = [results[dep] for dep in deps]
        else:
            dep_tasks = []

        self.logger.debug(f"Run task: {id} - {task}")
        self.logger.debug("Deps: " + json.dumps(dep_tasks))

        inst_args = []
        for arg in args:
            for key in arg:
                if isinstance(arg[key], str):
                    if "<GENERATED>" in arg[key]:
                        dep_id = int(arg[key].split("-")[1])
                        for result in results[dep_id]["inference result"]:
                            if key in result:
                                tmp_arg = arg.copy()
                                tmp_arg[key] = result[key]
                                inst_args.append(tmp_arg)
                    else: 
                        tmp_arg = arg.copy()
                        inst_args.append(tmp_arg)

                elif isinstance(arg[key], list):
                    tmp_arg = arg.copy()
                    for t in range(len(tmp_arg[key])):
                        item = tmp_arg[key][t]
                        if "<GENERATED>" in item:
                            dep_id = int(item.split("-")[1])
                            for result in results[dep_id]["inference result"]:
                                if key in result:
                                    tmp_arg[key][t] = result[key]
                                    break
                                    
                    inst_args.append(tmp_arg)

        for arg in inst_args:
            for resource in ["audio", "sheet_music"]:
                if resource in arg:
                    if not arg[resource].startswith(self.config["src_fold"]) and not arg[resource].startswith("http") and len(arg[resource]) > 0:
                        arg[resource] = f"{self.config['src_fold']}/{arg[resource]}"

        command["args"] = inst_args

        self.logger.debug(f"parsed task: {command}")

        if task in ["lyric-generation"]: # ChatGPT Can do
            best_model_id = "ChatGPT"
            reason = "ChatGPT performs well on some NLP tasks as well."
            choose = {"id": best_model_id, "reason": reason}
            inference_result = []

            for arg in command["args"]:
                chat_input = f"[{input_text}] contains a task in JSON format {command}. Now you are a {command['task']} system, the arguments are {arg}. Just help me do {command['task']} and give me the resultwithout any additional description. The result must be in text form without any urls."
                response = self.skillchat(chat_input, self.chatbot, self.chat_context)
                inference_result.append({"lyric":lyric_format(response)})

        else:
            if task not in self.task_map:
                self.logger.warning(f"no available models on {task} task.")
                inference_result = [{"error": f"{command['task']} not found in available tasks."}]
                results[id] = self.collect_result(command, "", inference_result)
                return False

            candidates = [pipe_id for pipe_id in self.task_map[task] if pipe_id in self.pipes]
            candidates = candidates[:self.config["candidate_tools"]]
            self.logger.debug(f"avaliable models on {command['task']}: {candidates}")

            if len(candidates) == 0:
                self.logger.warning(f"unloaded models on {task} task.")
                inference_result = [{"error": f"models for {command['task']} are not loaded."}]
                results[id] = self.collect_result(command, "", inference_result)
                return False
            
            if len(candidates) == 1:
                best_model_id = candidates[0]
                reason = "Only one model available."
                choose = {"id": best_model_id, "reason": reason}
                self.logger.debug(f"chosen model: {choose}")
            else:
                self.tool_context["available"] = ', '.join([cand.id for cand in candidates])
                choose_str = self.skillchat(input_text, self.tool_selector, self.tool_context)
                self.logger.debug(f"chosen model: {choose_str}")
                choose = json.loads(choose_str)
                reason = choose["reason"]
                best_model_id = choose["id"]

            inference_result = self.model_inference(best_model_id, command, device=self.config["device"])

        results[id] = self.collect_result(command, choose, inference_result)
        return True

    def chat(self, input_text):
        start = time.time()
        self.logger.info(f"input: {input_text}")

        task_str = self.skillchat(input_text, self.task_planner, self.task_context)
        self.logger.info(f"plans: {task_str}")

        try:
            tasks = json.loads(task_str)
        except Exception as e:
            self.logger.debug(e)
            response = self.skillchat(input_text, self.chatbot, self.chat_context)
            return response
        
        if len(tasks) == 0:
            response = self.skillchat(input_text, self.chatbot, self.chat_context)
            return response
        
        tasks = self.fix_depth(tasks)
        results = {}
        threads = []
        d = dict()
        retry = 0
        while True:
            num_thread = len(threads)
            for task in tasks:
                # logger.debug(f"d.keys(): {d.keys()}, dep: {dep}")
                for dep_id in task["dep"]:
                    if dep_id >= task["id"]:
                        task["dep"] = [-1]
                        break
                dep = task["dep"]
                if dep[0] == -1 or len(list(set(dep).intersection(d.keys()))) == len(dep):
                    tasks.remove(task)
                    thread = threading.Thread(target=self.run_task, args=(input_text, task, d))
                    thread.start()
                    threads.append(thread)

            if num_thread == len(threads):
                time.sleep(0.5)
                retry += 1

            if retry > 120:
                self.logger.debug("User has waited too long, Loop break.")
                break

            if len(tasks) == 0:
                break

        for thread in threads:
            thread.join()
        
        results = d.copy()
        self.logger.debug("results: ", results)

        self.response_context["processes"] = str(results)
        response = self.skillchat(input_text, self.responder, self.response_context)
        
        end = time.time()
        during = end - start
        self.logger.info(f"time: {during}s")
        return response
    
def parse_args():
    parser = argparse.ArgumentParser(description="A path to a YAML file")
    parser.add_argument("--config", type=str, help="a YAML file path.")

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    agent = MusicCoplilotAgent(args.config, mode="cli")
    print("Input exit or quit to stop the agent.")
    while True:
        message = input("Send a message: ")
        if message in ["exit", "quit"]:
            break

        print(agent.chat(message))

        


