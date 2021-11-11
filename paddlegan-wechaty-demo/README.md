# PaddleGAN-WeChaty-Demo

本示例将展示如何在[Wechaty](https://github.com/Wechaty/wechaty)中使用PaddleGAN的多种能力。

基本原理：通过[Wechaty](https://github.com/Wechaty/wechaty)获取微信接收的消息，然后使用PaddleGAN中的人脸动作迁移算法`first order motion`模型，将静态照片转换成动态趣味视频，最终以微信消息的形式发送。

## 风险提示

本项目采用的api为第三方——Wechaty提供，**非微信官方api**，用户需承担来自微信方的使用风险。  
在运行项目的过程中，建议尽量选用**新注册的小号**进行测试，不要用自己的常用微信号。

## Wechaty

关于Wechaty和python-wechaty，请查阅以下官方repo：
- [Wechaty](https://github.com/Wechaty/wechaty)
- [python-wechaty](https://github.com/wechaty/python-wechaty)
- [python-wechaty-getting-started](https://github.com/wechaty/python-wechaty-getting-started/blob/master/README.md)


## 环境准备

- 系统环境：Linux, MacOS, Windows
-  python3.7+


## 安装和使用

1. 安装PaddleGAN，详情请见[安装方式](https://github.com/PaddlePaddle/PaddleGAN/blob/develop/docs/zh_CN/install.md)

   ```shell
   git clone https://github.com/PaddlePaddle/paddlegan
   cd paddlegan-wechaty-demo
   ```

2. 安装依赖 —— paddlepaddle, ppgan, wechaty

   ```shell
   pip install -r requirements.txt
   ```

3. 安装项目所需的PaddleGAN的module

    此demo以`first order motion`为示例，其他module根据项目所需安装，更多的模型请查阅[PaddleGAN模型API接口说明](https://github.com/PaddlePaddle/PaddleGAN/blob/develop/docs/en_US/apis/apps.md)。

4. Set token for your bot

    在当前系统的环境变量中，配置以下与`WECHATY_PUPPET`相关的两个变量。
    关于其作用详情和TOKEN的获取方式，请查看[Wechaty Puppet Services](https://wechaty.js.org/docs/puppet-services/)。

    ```shell
    export WECHATY_PUPPET=wechaty-puppet-service
    export WECHATY_PUPPET_SERVICE_TOKEN=your_token_at_here
    ```

    [Paimon](https://wechaty.js.org/docs/puppet-services/paimon/)的短期TOKEN经测试可用，比赛期间将提供选手一个可使用1个月的token，大家可自行使用。

4. Run the bot

   ```shell
   python examples/paddleGAN_fom.py
   ```
   运行后，可以通过微信移动端扫码登陆，登陆成功后则可正常使用。

## 运行效果

在`examples/paddleGAN_fom.py`中，通过以下几行代码即可实例化一个`first order motion`的模型。

```python
# Initialize a PaddleGAN first order motion model
from ppgan.apps import FirstOrderPredictor
animate = FirstOrderPredictor(output="test_fom", filename="result.mp4",\
    relative=True, adapt_scale=True)
```

`on_message`方法是接收到消息时的回调函数，可以通过自定义的条件(譬如消息类型、消息来源、消息文字是否包含关键字、是否群聊消息等等)来判断是否回复信息，消息的更多属性和条件可以参考[Class Message](https://github.com/Wechaty/wechaty#3-class-message)。  

本示例中的`on_message`方法的代码如下，

```python
async def on_message(msg: Message):
    """
    Message Handler for the Bot
    """
    ### PaddleGAN fom

    global fom, source, driving

    if isinstance(msg.text(), str) and len(msg.text()) > 0 \
        and msg._payload.type == MessageType.MESSAGE_TYPE_TEXT \
        and "fom" in msg.text():
        bot_response = u"好嘞, 给我发个图片和驱动视频吧"
        fom = True
        await msg.say(bot_response)

    if fom and msg._payload.type == MessageType.MESSAGE_TYPE_IMAGE:
        fileBox = await msg.to_file_box()
        await fileBox.to_file("test_fom/source.jpg", True)

        bot_response = u"好嘞, 收到图片"
        await msg.say(bot_response)

        source = True

    if fom and msg._payload.type == MessageType.MESSAGE_TYPE_VIDEO:
        fileBox = await msg.to_file_box()
        await fileBox.to_file("test_fom/driving.mp4", True)

        bot_response = u"好嘞, 收到驱动视频"
        await msg.say(bot_response)

        driving = True

    if source and driving:
        bot_response = u"都收到啦，稍等一下嘿嘿"
        await msg.say(bot_response)
        source = False
        driving = False
        fom = False
        animate.run("test_fom/source.jpg", "test_fom/driving.mp4")
        file_box = FileBox.from_file("test_fom/result.mp4")
        await msg.say(file_box)

    ###  

```

脚本成功运行后，所登陆的账号即可作为一个Chatbot，下图左侧的内容由Chatbot生成和回复。
<div align='center'>
  <img src='https://user-images.githubusercontent.com/48054808/124779361-4ca4c800-df74-11eb-9a45-e4c82bab346b.jpeg'width='300'/>
</div>
