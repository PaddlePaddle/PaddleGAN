from collections import deque
import os
import asyncio

from wechaty import (
    Contact,
    FileBox,
    Message,
    Wechaty,
    ScanStatus,
)
from wechaty_puppet import MessageType

# Initialize a PaddleGAN fom model
from ppgan.apps import FirstOrderPredictor
animate = FirstOrderPredictor(output="test_fom", filename="result.mp4",\
    relative=True, adapt_scale=True)
fom = False
source = False
driving = False


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
        

async def on_scan(
        qrcode: str,
        status: ScanStatus,
        _data,
):
    """
    Scan Handler for the Bot
    """
    print('Status: ' + str(status))
    print('View QR Code Online: https://wechaty.js.org/qrcode/' + qrcode)


async def on_login(user: Contact):
    """
    Login Handler for the Bot
    """
    print(user)
    # TODO: To be written


async def main():
    """
    Async Main Entry
    """
    #
    # Make sure we have set WECHATY_PUPPET_SERVICE_TOKEN in the environment variables.
    #
    if 'WECHATY_PUPPET_SERVICE_TOKEN' not in os.environ:
        print('''
            Error: WECHATY_PUPPET_SERVICE_TOKEN is not found in the environment variables
            You need a TOKEN to run the Python Wechaty. Please goto our README for details
            https://github.com/wechaty/python-wechaty-getting-started/#wechaty_puppet_service_token
        ''')

    bot = Wechaty()

    bot.on('scan',      on_scan)
    bot.on('login',     on_login)
    bot.on('message',   on_message)

    await bot.start()


asyncio.run(main())
