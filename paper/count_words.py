# -*- coding: utf-8 -*-
"""Count main-text characters (excluding abstract & references) of manuscript.md."""
import re
import sys

text = open(sys.argv[1] if len(sys.argv) > 1 else 'manuscript.md', encoding='utf-8').read()
main = text.split('## 参考文献')[0]
# 去掉摘要区（--- 分隔线之前的标题/摘要/关键词）
parts = re.split(r'^---$', main, flags=re.M)
main = max(parts, key=len)
# 去掉行内公式
main = re.sub(r'\$[^$]*\$', '', main)
# 去掉标题标记与表格行
main = re.sub(r'#+ ', '', main)
main = re.sub(r'\|[^\n]*\|', '', main)
# 去掉 Markdown 符号
main = re.sub(r'[|*_\-\n\r]', '', main)
clean = main.replace(' ', '')
print('main-body chars (with punct):', len(clean))
print('hanzi count:', len(re.findall(r'[一-鿿]', clean)))
print('segments after --- split:', len(parts))
