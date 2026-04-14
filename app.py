import os
import sys
import logging
import numpy as np
import torch
import gradio as gr
import soundfile as sf
import re
from typing import Optional, Tuple, List
from funasr import AutoModel
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import voxcpm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ---------- 【新增】路径配置 ----------
PERSONA_DIR = Path.cwd() / "personas"
PERSONA_DIR.mkdir(parents=True, exist_ok=True)

def get_persona_list():
    files = [f.stem for f in PERSONA_DIR.glob("*.wav")]
    return sorted(files) if files else ["(暂无音色)"]

# ---------- Inline i18n (完全还原原始所有文本) ----------

_USAGE_INSTRUCTIONS_EN = (
    "**VoxCPM2 — Three Modes of Speech Generation:**\n\n"
    "🎨 **Voice Design** — Create a brand-new voice  \n"
    "No reference audio required. Describe the desired voice characteristics "
    "(gender, age, tone, emotion, pace …) in **Control Instruction**, and VoxCPM2 "
    "will craft a unique voice from your description alone.\n\n"
    "🎛️ **Controllable Cloning** — Clone a voice with optional style guidance  \n"
    "Upload a reference audio clip, then use **Control Instruction** to steer "
    "emotion, speaking pace, and overall style while preserving the original timbre.\n\n"
    "🎙️ **Ultimate Cloning** — Reproduce every vocal nuance through audio continuation  \n"
    "Turn on **Ultimate Cloning Mode** and provide (or auto-transcribe) the reference audio's transcript. "
    "The model treats the reference clip as a spoken prefix and seamlessly **continues** from it, faithfully preserving every vocal detail."
    "Note: This mode will disable Control Instruction.\n\n"
    "📖 [**Cookbook: VoxCPM 2 Best Practices Guide**](https://voxcpm.readthedocs.io/en/latest/cookbook.html)"
)

_EXAMPLES_FOOTER_EN = (
    "---\n"
    "**💡 Voice Description Examples:**  \n"
    "Try the following Control Instructions to explore different voices:  \n\n"
    "**Example 1 — Gentle & Melancholic Girl**  \n"
    '`Control Instruction`: *"A young girl with a soft, sweet voice. '
    'Speaks slowly with a melancholic, slightly tsundere tone."*  \n'
    '`Target Text`: *"I never asked you to stay… It\'s not like I care or anything. '
    'But… why does it still hurt so much now that you\'re gone?"*  \n\n"'
    "**Example 2 — Laid-Back Surfer Dude**  \n"
    '`Control Instruction`: *"Relaxed young male voice, slightly nasal, '
    'lazy drawl, very casual and chill."*  \n'
    '`Target Text`: *"Dude, did you see that set? The waves out there are totally gnarly today. '
    "Just catching barrels all morning — it's like, totally righteous, you know what I mean?\"*"
)

_USAGE_INSTRUCTIONS_ZH = (
    "**VoxCPM2 — 三种语音生成方式：**\n\n"
    "🎨 **声音设计（Voice Design）**  \n"
    "无需参考音频。在 **Control Instruction** 中描述目标音色特征"
    "（性别、年龄、语气、情绪、语速等），VoxCPM2 即可为你从零创造独一无二的声音。\n\n"
    "🎛️ **可控克隆（Controllable Cloning）**  \n"
    "上传参考音频，同时可选地使用 **Control Instruction** 来指定情绪、语速、风格等表达方式，"
    "在保留原始音色的基础上灵活控制说话风格。\n\n"
    "🎙️ **极致克隆（Ultimate Cloning）**  \n"
    "开启 **极致克隆模式** 并提供参考音频的文字内容（可自动识别）。"
    "模型会将参考音频视为已说出的前文，以**音频续写**的方式完整还原参考音频中的所有声音细节。"
    "注意：该模式与可控克隆模式互斥，将禁用Control Instruction。\n\n"
    "📖 [**必读：VoxCPM 2 最佳实践指南**](https://voxcpm.readthedocs.io/zh-cn/latest/cookbook.html) \n"
)

_EXAMPLES_FOOTER_ZH = (
    "---\n"
    "**💡 声音描述示例（中英文均可）：**  \n\n"
    "**示例 1 — 深宫太后**  \n"
    '`Control Instruction`: *"中老年女性，声音低沉阴冷，语速缓慢而有力，'
    '字字深思熟虑，带有深不可测的城府与威慑感。"*  \n'
    '`Target Text`: *"哀家在这深宫待了四十年，什么风浪没见过？你以为瞒得过哀家？"*  \n\n'
    "**示例 2 — 暴躁驾校教练**  \n"
    '`Control Instruction`: *"暴躁的中年男声，语速快，充满无奈和愤怒"*  \n'
    '`Target Text`: *"踩离合！踩刹车啊！你往哪儿开呢？前面是树你看不见吗？'
    '我教了你八百遍了，打死方向盘！你是不是想把车给我开到沟里去？"*  \n\n'
    "---\n"
    "**🗣️ 方言生成指南：**  \n"
    "要生成地道的方言语音，请在 **Target Text** 中直接使用方言词汇和句式，"
    "并在 **Control Instruction** 中描述方言特征。  \n\n"
    "**示例 — 广东话**  \n"
    '`Control Instruction`: *"粤语，中年男性，语气平淡"*  \n'
    '✅ 正确（粤语表达）：*"伙計，唔該一個A餐，凍奶茶少甜！"*  \n'
    '❌ 错误（普通话原文）：*"伙计，麻烦来一个A餐，冻奶茶少甜！"*  \n\n'
    "**示例 — 河南话**  \n"
    '`Control Instruction`: *"河南话，接地气的大叔"*  \n'
    '✅ 正确（河南话表达）：*"恁这是弄啥嘞？晌午吃啥饭？"*  \n'
    '❌ 错误（普通话原文）：*"你这是在干什么呢？中午吃什么饭？"*  \n\n'
    "🤖 **小技巧：** 不知道方言怎么写？可以用豆包、DeepSeek、Kimi 等 AI 助手"
    "将普通话翻译为方言文本，再粘贴到 Target Text 中即可。  \n\n"
)

_I18N_TRANSLATIONS = {
    "en": {
        "tab_normal": "Normal Mode",
        "tab_studio": "🎬 Script Studio",
        "persona_name_label": "Persona Name",
        "btn_save_persona": "💾 Solidify Voice",
        "script_label": "Script Content",
        "script_placeholder": "Example: [RoleA](Happy) Hello!\n[RoleB] Who are you?",
        "btn_generate_script": "🚀 Generate Multi-Role Dialogue",
        "persona_list_label": "Persona Library (Click to tag)",
        "reference_audio_label": "🎤 Reference Audio (optional — upload for cloning)",
        "show_prompt_text_label": "🎙️ Ultimate Cloning Mode (transcript-guided cloning)",
        "show_prompt_text_info": "Auto-transcribes reference audio for every vocal nuance reproduced. Control Instruction will be disabled when active.",
        "prompt_text_label": "Transcript of Reference Audio (auto-filled via ASR, editable)",
        "prompt_text_placeholder": "The transcript of your reference audio will appear here …",
        "control_label": "🎛️ Control Instruction (optional — supports Chinese & English)",
        "control_placeholder": "e.g. A warm young woman / 年轻女性，温柔甜美 / Excited and fast-paced",
        "target_text_label": "✍️ Target Text — the content to speak",
        "generate_btn": "🔊 Generate Speech",
        "generated_audio_label": "Generated Audio",
        "advanced_settings_title": "⚙️ Advanced Settings",
        "ref_denoise_label": "Reference audio enhancement",
        "ref_denoise_info": "Apply ZipEnhancer denoising to the reference audio before cloning",
        "normalize_label": "Text normalization",
        "normalize_info": "Normalize numbers, dates, and abbreviations via wetext",
        "cfg_label": "CFG (guidance scale)",
        "cfg_info": "Higher → closer to the prompt / reference; lower → more creative variation",
        "dit_steps_label": "LocDiT flow-matching steps",
        "dit_steps_info": "LocDiT flow-matching steps — more steps → maybe better audio quality, but slower",
        "usage_instructions": _USAGE_INSTRUCTIONS_EN,
        "examples_footer": _EXAMPLES_FOOTER_EN,
    },
    "zh-CN": {
        "tab_normal": "普通模式",
        "tab_studio": "🎬 剧本工坊",
        "persona_name_label": "音色名称",
        "btn_save_persona": "💾 固化到音色库",
        "script_label": "剧本内容",
        "script_placeholder": "格式示例：[角色A](开心) 你好啊！\n[角色B] 你是谁？（无括号则自动启用极致模式）",
        "btn_generate_script": "🚀 生成多人对话",
        "persona_list_label": "音色库 (点击选择角色)",
        "reference_audio_label": "🎤 参考音频（可选 — 上传后用于克隆）",
        "show_prompt_text_label": "🎙️ 极致克隆模式（基于文本引导的极致克隆）",
        "show_prompt_text_info": "自动识别参考音频文本，完整还原音色、节奏、情感等全部声音细节。开启后 Control Instruction 将暂时禁用",
        "prompt_text_label": "参考音频内容文本（ASR 自动填充，可手动编辑）",
        "prompt_text_placeholder": "参考音频的文字内容将自动识别并显示在此处 …",
        "control_label": "🎛️ Control Instruction（可选 — 支持中英文描述）",
        "control_placeholder": "如：年轻女性，温柔甜美 / A warm young woman / 暴躁老哥，语速飞快",
        "target_text_label": "✍️ Target Text — 要合成的目标文本",
        "generate_btn": "🔊 开始生成",
        "generated_audio_label": "生成结果",
        "advanced_settings_title": "⚙️ 高级设置",
        "ref_denoise_label": "参考音频降噪增强",
        "ref_denoise_info": "克隆前使用 ZipEnhancer 对参考音频进行降噪处理",
        "normalize_label": "文本规范化",
        "normalize_info": "自动规范化数字、日期及缩写（基于 wetext）",
        "cfg_label": "CFG（引导强度）",
        "cfg_info": "数值越高 → 越贴合提示/参考音色；数值越低 → 生成风格更自由",
        "dit_steps_label": "LocDiT 流匹配迭代步数",
        "dit_steps_info": "LocDiT 流匹配生成迭代步数 — 步数越多 → 可能生成更好的音频质量，但速度变慢",
        "usage_instructions": _USAGE_INSTRUCTIONS_ZH,
        "examples_footer": _EXAMPLES_FOOTER_ZH,
    },
}
_I18N_TRANSLATIONS["zh-Hans"] = _I18N_TRANSLATIONS["zh-CN"]
_I18N_TRANSLATIONS["zh"] = _I18N_TRANSLATIONS["zh-CN"]

for _d in _I18N_TRANSLATIONS.values():
    if _d is not None:
        for _k, _v in _I18N_TRANSLATIONS["en"].items():
            _d.setdefault(_k, _v)

I18N = gr.I18n(**_I18N_TRANSLATIONS)

DEFAULT_TARGET_TEXT = (
    "VoxCPM2 is a creative multilingual TTS model from ModelBest, "
    "designed to generate highly realistic speech."
)

_CUSTOM_CSS = """
.logo-container { text-align: center; margin: 0.5rem 0 1rem 0; }
.logo-container img { height: 80px; width: auto; max-width: 200px; display: inline-block; }
.switch-toggle { padding: 8px 12px; border-radius: 8px; background: var(--block-background-fill); }
.switch-toggle input[type="checkbox"] { appearance: none; -webkit-appearance: none; width: 44px; height: 24px; background: #ccc; border-radius: 12px; position: relative; cursor: pointer; transition: background 0.3s ease; flex-shrink: 0; }
.switch-toggle input[type="checkbox"]::after { content: ""; position: absolute; top: 2px; left: 2px; width: 20px; height: 20px; background: white; border-radius: 50%; transition: transform 0.3s ease; box-shadow: 0 1px 3px rgba(0,0,0,0.2); }
.switch-toggle input[type="checkbox"]:checked { background: var(--color-accent); }
.switch-toggle input[type="checkbox"]:checked::after { transform: translateX(20px); }
"""

_APP_THEME = gr.themes.Soft(
    primary_hue="blue", secondary_hue="gray", neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "Arial", "sans-serif"],
)

# ---------- Model ----------

class VoxCPMDemo:
    def __init__(self, model_id: str = "./pretrained_models/VoxCPM2") -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Running on device: {self.device}")

        self.asr_model_id = "./pretrained_models/SenseVoiceSmall"
        self.asr_model: Optional[AutoModel] = AutoModel(
            model=self.asr_model_id, disable_update=True, log_level="DEBUG",
            device="cuda:0" if self.device == "cuda" else "cpu",
        )
        self.voxcpm_model: Optional[voxcpm.VoxCPM] = None
        self._model_id = model_id

    def get_or_load_voxcpm(self) -> voxcpm.VoxCPM:
        if self.voxcpm_model is not None: return self.voxcpm_model
        logger.info(f"Loading model: {self._model_id}")
        self.voxcpm_model = voxcpm.VoxCPM.from_pretrained(self._model_id, optimize=True)
        return self.voxcpm_model

    def prompt_wav_recognition(self, prompt_wav: Optional[str]) -> str:
        if prompt_wav is None: return ""
        res = self.asr_model.generate(input=prompt_wav, language="auto", use_itn=True)
        return res[0]["text"].split("|>")[-1]

    def _build_generate_kwargs(
        self, *, final_text: str, audio_path: Optional[str], prompt_text_clean: Optional[str],
        cfg_value_input: float, do_normalize: bool, denoise: bool, inference_timesteps: int = 10,
    ) -> dict:
        generate_kwargs = dict(
            text=final_text, reference_wav_path=audio_path, cfg_value=float(cfg_value_input),
            inference_timesteps=inference_timesteps, normalize=do_normalize, denoise=denoise,
        )
        if prompt_text_clean and audio_path:
            generate_kwargs["prompt_wav_path"] = audio_path
            generate_kwargs["prompt_text"] = prompt_text_clean
        return generate_kwargs

    def generate_tts_audio(
        self, text_input: str, control_instruction: str = "", reference_wav_path_input: Optional[str] = None,
        prompt_text: str = "", cfg_value_input: float = 2.0, do_normalize: bool = True,
        denoise: bool = True, inference_timesteps: int = 10,
    ) -> Tuple[int, np.ndarray]:
        current_model = self.get_or_load_voxcpm()
        text = (text_input or "").strip()
        if len(text) == 0: raise ValueError("Please input text to synthesize.")
        control = (control_instruction or "").strip()
        final_text = f"({control}){text}" if control else text
        audio_path = reference_wav_path_input if reference_wav_path_input else None
        prompt_text_clean = (prompt_text or "").strip() or None

        generate_kwargs = self._build_generate_kwargs(
            final_text=final_text, audio_path=audio_path, prompt_text_clean=prompt_text_clean,
            cfg_value_input=cfg_value_input, do_normalize=do_normalize, denoise=denoise,
            inference_timesteps=inference_timesteps,
        )
        wav = current_model.generate(**generate_kwargs)
        return (current_model.tts_model.sample_rate, wav)

# ---------- 【新增/修改】业务逻辑 ----------

def fn_save_persona(name, audio_data, text_content):
    if not name or audio_data is None: return "❌ 失败：请输入名称并确保已有生成结果"
    sr, wav = audio_data
    out_path = PERSONA_DIR / f"{name}.wav"
    txt_path = PERSONA_DIR / f"{name}.txt"
    sf.write(str(out_path), wav, sr)
    # 同时保存文本内容，用于后期极致克隆
    with open(str(txt_path), "w", encoding="utf-8") as f:
        f.write(text_content if text_content else "")
    return f"✅ 音色 [{name}] 已成功固化（含参考文本）！"

def fn_script_studio(demo, script, cfg, norm, denoise, steps):
    lines = script.strip().split('\n')
    combined_wav = []
    sr_final = 48000
    for line in lines:
        # 匹配格式 [角色](指令)文本 或 [角色]文本
        match = re.match(r"\[([^\]]+)\](?:\(([^)]+)\))?\s*(.*)", line)
        if match:
            role, emotion, content = match.groups()
            ref_path = PERSONA_DIR / f"{role}.wav"
            txt_path = PERSONA_DIR / f"{role}.txt"
            if not ref_path.exists(): continue
            
            p_text = ""
            ctrl_ins = ""
            
            # 判断逻辑：如果有括号指令，使用可控克隆；如果没有括号，尝试使用极致克隆
            if emotion:
                ctrl_ins = emotion
                logger.info(f"[剧本模式] 角色:{role} -> 使用可控模式 (指令:{ctrl_ins})")
            else:
                if txt_path.exists():
                    with open(str(txt_path), "r", encoding="utf-8") as f:
                        p_text = f.read().strip()
                    logger.info(f"[剧本模式] 角色:{role} -> 发现固化文本，使用极致模式")
                else:
                    logger.info(f"[剧本模式] 角色:{role} -> 未发现固化文本，使用普通克隆")

            sr, wav = demo.generate_tts_audio(
                text_input=content, control_instruction=ctrl_ins,
                reference_wav_path_input=str(ref_path), prompt_text=p_text,
                cfg_value_input=cfg, do_normalize=norm, denoise=denoise, 
                inference_timesteps=int(steps)
            )
            combined_wav.append(wav)
            combined_wav.append(np.zeros(int(sr * 0.9))) # 0.3s pause
            sr_final = sr
            
    if not combined_wav: return None, "❌ 匹配失败：请检查剧本格式或音色库"
    return (sr_final, np.concatenate(combined_wav)), "✅ 生成成功！"

# ---------- UI ----------

def create_demo_interface(demo: VoxCPMDemo):
    script_dir = Path(__file__).parent.absolute()
    gr.set_static_paths(paths=[script_dir / "assets"])

    def _generate(text, control_instruction, ref_wav, use_prompt_text, prompt_text_value, cfg_value, do_normalize, denoise, dit_steps):
        actual_prompt_text = prompt_text_value.strip() if use_prompt_text else ""
        actual_control = "" if use_prompt_text else control_instruction
        return demo.generate_tts_audio(
            text_input=text, control_instruction=actual_control, reference_wav_path_input=ref_wav,
            prompt_text=actual_prompt_text, cfg_value_input=cfg_value, do_normalize=do_normalize,
            denoise=denoise, inference_timesteps=int(dit_steps)
        )

    def _on_toggle_instant(checked):
        if checked: return (gr.update(visible=True, value="", placeholder="Recognizing reference audio..."), gr.update(visible=False))
        return (gr.update(visible=False), gr.update(visible=True, interactive=True))

    def _run_asr_if_needed(checked, audio_path):
        if not checked or not audio_path: return gr.update()
        try:
            asr_text = demo.prompt_wav_recognition(audio_path)
            return gr.update(value=asr_text)
        except: return gr.update(value="")

    # 快捷功能：插入带括号的指令到剧本
    def add_instruction_tag(script, instruction):
        if not instruction: return script
        return f"{script.rstrip()}({instruction}) ", ""

    with gr.Blocks(css=_CUSTOM_CSS, theme=_APP_THEME) as interface:
        gr.HTML("""
            <div style="text-align: center; margin: 20px 0; font-family: 'Inter', sans-serif;">
                <h1 style="font-size: 72px; font-weight: 800; color: white; margin-bottom: 10px; line-height: 1;">VoxCPM2</h1>
                <div style="font-size: 38px; font-weight: 700; background: linear-gradient(90deg, #FFFFFF 0%, #A594F9 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; line-height: 1.2;">
                    多人对话剧本工坊版
                </div>
            </div>
        """)
        gr.Markdown(I18N("usage_instructions"))

        with gr.Tabs():
            # ---------- 普通模式 (完全还原原版) ----------
            with gr.Tab(I18N("tab_normal")):
                with gr.Row():
                    with gr.Column():
                        reference_wav = gr.Audio(sources=["upload", "microphone"], type="filepath", label=I18N("reference_audio_label"))
                        show_prompt_text = gr.Checkbox(value=False, label=I18N("show_prompt_text_label"), info=I18N("show_prompt_text_info"), elem_classes=["switch-toggle"])
                        prompt_text = gr.Textbox(label=I18N("prompt_text_label"), placeholder=I18N("prompt_text_placeholder"), lines=2, visible=False)
                        control_instruction = gr.Textbox(label=I18N("control_label"), placeholder=I18N("control_placeholder"), lines=2)
                        text = gr.Textbox(value=DEFAULT_TARGET_TEXT, label=I18N("target_text_label"), lines=3)

                        with gr.Accordion(I18N("advanced_settings_title"), open=False):
                            DoDenoise = gr.Checkbox(value=False, label=I18N("ref_denoise_label"))
                            DoNorm = gr.Checkbox(value=False, label=I18N("normalize_label"))
                            cfg_val = gr.Slider(1.0, 3.0, 2.0, step=0.1, label=I18N("cfg_label"))
                            dit_steps = gr.Slider(1, 50, 10, step=1, label=I18N("dit_steps_label"))

                        run_btn = gr.Button(I18N("generate_btn"), variant="primary", size="lg")
                    
                    with gr.Column():
                        audio_output = gr.Audio(label=I18N("generated_audio_label"))
                        # 【新增】音色固化区
                        with gr.Row():
                            p_name = gr.Textbox(label=I18N("persona_name_label"), placeholder="如：御姐A")
                            save_p_btn = gr.Button(I18N("btn_save_persona"))
                        p_status = gr.Textbox(label="状态", interactive=False)
                        gr.Markdown(I18N("examples_footer"))

                # Event Logic
                show_prompt_text.change(_on_toggle_instant, [show_prompt_text], [prompt_text, control_instruction]).then(
                    _run_asr_if_needed, [show_prompt_text, reference_wav], [prompt_text]
                )
                run_btn.click(_generate, [text, control_instruction, reference_wav, show_prompt_text, prompt_text, cfg_val, DoNorm, DoDenoise, dit_steps], [audio_output])
                # 修改：保存时同时传入目标文本
                save_p_btn.click(fn_save_persona, [p_name, audio_output, text], [p_status])

            # ---------- 剧本工坊 (新增) ----------
            with gr.Tab(I18N("tab_studio")):
                with gr.Row():
                    with gr.Column(scale=2):
                        script_input = gr.TextArea(label=I18N("script_label"), placeholder=I18N("script_placeholder"), lines=12, value="[旁白](缓缓地说) 韩立默默退到了众人身后。\n[韩立] 我叫历飞雨。")
                        # 【新增】指令输入窗口，回车插入剧本
                        script_ins = gr.Textbox(label=I18N("control_label"), placeholder="输入指令后回车插入剧本，如：四川话，无奈地说")
                        btn_script = gr.Button(I18N("btn_generate_script"), variant="primary")
                    with gr.Column(scale=1):
                        p_list = gr.Dropdown(label=I18N("persona_list_label"), choices=get_persona_list())
                        btn_refresh = gr.Button("🔄 刷新音色库")
                        script_audio = gr.Audio(label=I18N("generated_audio_label"))
                        script_msg = gr.Textbox(label="状态")
                
                # 交互逻辑
                def add_role_tag(s, r): return f"{s.rstrip()}\n[{r}] "
                p_list.input(add_role_tag, [script_input, p_list], [script_input])
                # 回车插入指令逻辑
                script_ins.submit(add_instruction_tag, [script_input, script_ins], [script_input, script_ins])
                
                btn_refresh.click(lambda: gr.update(choices=get_persona_list()), None, p_list)
                btn_script.click(lambda *args: fn_script_studio(demo, *args), [script_input, cfg_val, DoNorm, DoDenoise, dit_steps], [script_audio, script_msg])

    return interface

def run_demo(server_name="0.0.0.0", server_port=8808, show_error=True, model_id="./pretrained_models/VoxCPM2"):
    demo = VoxCPMDemo(model_id=model_id)
    interface = create_demo_interface(demo)
    interface.queue(max_size=10).launch(server_name=server_name, server_port=server_port, show_error=show_error, i18n=I18N, theme=_APP_THEME, css=_CUSTOM_CSS, inbrowser=True)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", type=str, default="./pretrained_models/VoxCPM2")
    parser.add_argument("--port", type=int, default=8808)
    args = parser.parse_args()
    run_demo(model_id=args.model_id, server_port=args.port)