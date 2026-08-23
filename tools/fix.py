import re

def replace_in_file(filepath, pattern, repl):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    new_content = re.sub(pattern, repl, content)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(new_content)

replace_in_file('scripts/phase2_smoke.py', r'Path\("\."\)\.resolve\(\)', 'Path.cwd()')
replace_in_file('scripts/phase2_smoke.py', r'except Exception as e:', 'except Exception as e:  # noqa: BLE001')

replace_in_file('src/models/state_inferencer.py', r'r\\\'\\\\b\(already closed\|not closed\|remained open\)\\\\b\\\'', r"r'\b(already closed|not closed|remained open)\b'")

replace_in_file('tests/test_s05_segment.py', r'assert disp == 0\.1', 'assert _disp == 0.1')
