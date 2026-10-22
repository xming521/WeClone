@echo off
rem ============================================================
rem  复制这个文件为 secrets.local.cmd，然后填入你自己的密钥。
rem  secrets.local.cmd 已被 .gitignore 忽略，不会进入仓库。
rem ============================================================

rem DeepSeek API Key（打分 / 人设 / 对话都用它）
set "WECLONE_API_KEY=sk-替换成你的key"

rem 本机分身接口的鉴权口令（LangBot 里要填同一个值）
set "WECLONE_SERVE_KEY=local-key"
