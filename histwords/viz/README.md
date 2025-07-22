本目录包含用于可视化 histwords 词向量的代码和脚本。

# png 图片

scripts/ 目录下有用于生成 .png 图像的脚本。运行 `python scripts/closest_over_time_with_anns.py awful` 会在 viz/output/ 目录下生成图片文件。

注意：如果你使用的是 Mac OS，若 python 命令无法运行，可以尝试使用 'pythonw'。

# Web 探索器

web 目录下包含一个自包含的 Web 服务器，可用于交互式探索 histwords 词向量。运行 `python viz/web/main.py`，然后打开 http://localhost:5000 并输入你要查询的词。

## 多词探索

如需同时查询多个词，用冒号分隔，例如：
"awful:terrible"。当输入多个查询词时，结果会根据每个查询词用不同颜色标记。将鼠标悬停在某个词上，可以看到该词与哪些查询词相关。

如果你输入了很多词，可能需要等待较长时间！一般来说，同时查询 2~3 个词大约需要 5 秒以上。
