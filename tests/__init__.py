"""测试包标记。

存在此文件后 pytest 会把 tests 视为包，`from .helpers import ...`
这类相对导入才能正常工作（pytest.ini 已用 `pythonpath = .` 把项目根
加入 sys.path）。
"""
