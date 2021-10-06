####  实验室发布测试

解决jupyter内核经常断开连接，并且命令行报错ImportError: cannot import name 'create_prompt_application'

可能导致此问题的原因：jupyter与pycharm混用导致的pycharm自动更新了一些包，从而包之间的相互依赖出现版本问题

解决方法：prompt执行

```
pip3 install --upgrade --force jupyter-console
```

若报错

Cannot uninstall 'entrypoints'. It is a distutils installed project and thus we cannot accurately determine which files belong to it which would lead to only a partial uninstall.

改成

pip3 install --upgrade --ignore-installed --force jupyter-console