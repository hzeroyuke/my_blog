针对具身场景数据调一下prompt，整合了一下代码

**整体流程**

数据组织格式

```txt
- real
	- motor
		- motor_longlive.mp4
		- motor_gt.mp4
		- prompt.txt
	- driving
		- driving_longlive.mp4
		- driving_gt.mp4
		- prompt.txt
- robotics
	- 
- gaming
```

数据预处理

- 先VLM判断识别的目标
- SAM模型 -> bounding box + mask

VLM 判断

- 先判断是否related
- 再判断是否符合规律

**新的功能**

计时

- Long Live 1 min video sam分割处理 约 1000 frames ：大概需要260 seconds about 4-5min ，考虑到gt video可以先处理好，就不计算两倍的时间
- vlm call带video输入 以 gemini flash model 为例子 大概需要 6-12 seconds（和API供应也有关系）

重试功能

- vlm call输出可能因为网络问题或者vlm本身的问题导致出错，需要增加重试功能
- 最多重试三次

**results** 

| question       | motor       | motor_gt    | zigbag      | zigbag_gt    |
| -------------- | ----------- | ----------- | ----------- | ------------ |
| gravity        | false       | true        | not related | not related  |
| buoyancy       | not related | not related | not related | not related  |
| compression    | false       | true        | false       | true         |
| impact         | false       | true        | not related | not related  |
| melting        | not related | not related | not related | not related  |
| sublimation    | not related | not related | not related | not related  |
| vaporization   | not related | not related | not related | not related  |
| condensation   | not related | not related | not related | not related  |
| deposition     | not related | not related | not related | not related  |
| freezing       | not related | not related | not related | not related  |
| color_mixing   | not related | not related | not related | not related  |
| solubility     | not related | not related | not related | not related  |
| hardness       | not related | not related | false       | true         |
| combustibility | not related | not related | not related | not raelated |

**数据清洗**

- 下载数据的脚本和查找tasks!=1的脚本
- 筛掉所有fisheyes的脚本（不需要筛掉，只从任务内容上判断）


**World Arena**

- 均匀采样16frames
- Qwen3-VL

```bash
1. Interaction_Quality (Quality of robot-object interactions)
- Score 1: Objects pass through robot or other objects; no proper contact
- Score 2: Contact exists but interaction is unrealistic (e.g., sliding without friction, incorrect force response)
- Score 3: Mostly plausible interactions with minor issues (e.g., slight penetration, imperfect grasping)
- Score 4: Realistic contact physics (proper friction, force transfer, object deformation)
- Score 5: Perfect interaction physics; indistinguishable from real robot manipulation
2. PERSPECTIVITY (3D consistency and camera geometry)
- Score 1: Scene has no coherent 3D structure; objects float inconsistently
- Score 2: 3D structure is unstable (e.g., scale changes, incorrect occlusion)
- Score 3: Reasonable 3D consistency with minor issues (e.g., slight perspective drift)
- Score 4: Stable camera perspective with consistent depth relationships
- Score 5: Perfect camera geometry and 3D consistency
3. INSTRUCTION FOLLOWING (Adherence to given instruction)
- HALLUCINATION CHECK: If the video shows human hands instead of robotic arms, score <= 2 immediately
- Score 1: Completely different from instruction (wrong action, wrong objects, wrong scene)
- Score 2: Partially related but major errors (e.g., wrong target object, incorrect manipulation type)
- Score 3: Follows general intent but with execution errors (e.g., correct action sequence but imprecise)
- Score 4: Mostly correct with minor deviations (e.g., slight position error, extra unnecessary motion)
- Score 5: Perfect execution of all specified elements (action, object, scene, outcome)
```


另外还有一批关于robotics的检查

```bash
SPECIFIC ROBOT-RELATED CHECKS:
- Robotic arm should have mechanical appearance, NOT human limbs
- End-effector (gripper) should maintain consistent form throughout interaction
- Robot motion should show appropriate joint movement and kinematics
- Object manipulation should respect object mass and inertia
- Contact should be maintained appropriately during grasping/lifting
```


打数据打了一半，预计周二周三的样子可以交付

- 去掉失败的例子
- 去掉大量人员走动
- 去掉极端的摄像头方向的