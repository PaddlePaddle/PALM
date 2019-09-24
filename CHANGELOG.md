Changelog
===
以下记录了项目中所有值得关注的变更内容，其格式基于[Keep a Changelog]。

本项目版本遵守[Semantic Versioning]和[PEP-440]。

v1.1 - 2019-9-21
---
v1.1版是在不更改核心设计的前提下提高库的易用性和容错性，明确用户自定义接口规范，新增模型backbone “ERNIE”和一个问答匹配预置任务，并支持了对主任务的预测。

具体更新内容如下：

1. 新增ERNIE为预置模型backbone

2. 新增了answer_matching预置任务

3. 新增对主任务的预测功能，用户可通过设置do_predict为True来完成对主任务的预测

4. 修复了BERT finetuning精度bug

5. 提高容错性，如为预训练模型的载入、ckpt保存与载入引入检查机制等

6. 优化了MRC任务的prediction流程，避免反复的数据读取和处理

7. 不影响理解的情况下简化和规范化模块命名，如`backbone_representation` -> `backbone`, `train_multi_task.py`-> `mtl_run.py`等

8. 减少目录层级，降低自定义复杂度，如backbone下描述BERT所需开发完成的接口

   *before：*

   ```
   ├───backbone_representation
      ├────bert
          ├────bert.py
          └────bert_model.py
   ```

   *after：*

   ```
   ├───backbone
      └────bert.py
   ```

8. 优化了日志打印（数量、排版和位置）

9. 优化了配置文件必选字段，例如

   *before:*

   ```
   backbone_model:backbone_representation/bert:bert_model:ModelBERT
   checkpoints: models/save_models/firstrun
   init_checkpoint: models/save_models/firstrun/params
   ```

   *after:*

   ```
   backbone: bert_model
   checkpoint_path: output_model/firstrun
   ```


1.0 - 2019-08-27
---
### Added
- 创建项目


[Unreleased]: http://icode.baidu.com/repos/baidu/personal-code/multi-task-learning/merge/0.1.0...master

[Keep a Changelog]: https://keepachangelog.com/zh-CN/1.0.0/
[Semantic Versioning]: https://semver.org/lang/zh-CN/
[PEP-440]: https://www.python.org/dev/peps/pep-0440/